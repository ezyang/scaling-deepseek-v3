// Event-driven simulator: turns (config, refinement level) into a trace.
// Refinements are cumulative and ordered by (rough) impact on step walltime.

import * as M from './model.js';
import { memoryUsage, layerAnalysis } from './memory.js';

export const LEVELS = [
  { id: 'roofline', title: 'Roofline', desc: 'total FLOPs ÷ peak FLOP/s, nothing else' },
  { id: 'compute', title: 'Kernel costs', desc: 'per-op roofline: GEMM efficiency, memory-bound ops, flash attention' },
  { id: 'pipeline', title: '1F1B pipeline', desc: 'warmup/cooldown bubbles, stage imbalance' },
  { id: 'expert-comm', title: 'Expert-parallel all-to-all', desc: 'MoE dispatch/combine on NVLink + IB' },
  { id: 'dp-comm', title: 'FSDP & pipeline comm', desc: 'all-gather, reduce-scatter, p2p sends, optimizer step' },
  { id: 'overhead', title: 'Launch overhead', desc: 'per-kernel launch cost + host gaps between microbatches' },
  { id: 'jitter', title: 'Jitter & stragglers', desc: 'kernel time noise + Python GC pauses convoying collectives' },
];

export function refinementsAt(level) {
  return {
    ops: level >= 1, pipeline: level >= 2, epComm: level >= 3,
    dpComm: level >= 4, overhead: level >= 5, jitter: level >= 6,
  };
}

export function defaultConfig(overrides = {}) {
  const cfg = {
    hardware: 'h800', dtype: 'bf16',
    gpus: 256,                           // total cluster size; dp derives as gpus/pp
    pp: 16, ep: 64,                      // ep is clamped to dp
    microbatches: 24, mbs: 1, seqLen: 4096,
    globalBatch: null,                   // sequences/step; when set: mbs = 1, m = GBS / dp
    nodeLimit: 4,                        // max nodes a token's experts may span
    dpRanksToSim: 1,                     // simulated dp ranks per stage (raise to see stragglers)
    granularity: 'layer',                // 'phase' | 'layer' | 'op'
    level: 4,
    zero: 3,                             // 0 replicate | 1 shard optimizer | 3 FSDP (shard params+grads too)
    recompute: 'selective',              // 'none' | 'selective' | 'full' (memory model)
    recipe: null,                        // per-matmul precision preset (memory.js RECIPES); null = derive from dtype
    matmuls: null,                       // per-matmul dtype overrides, e.g. {o_proj:'bf16'}
    saved: null,                         // per-op SAVE marks over the recompute preset, e.g. {x1: true} (unlisted = recompute)
    kernelOverheadUs: 5, hostGapUs: 40,
    gcProb: 0.02, gcMs: 150, opSigma: 0.03,
    seed: 42,
    ...overrides,
  };
  // dp derives from the cluster size (an explicit dp override wins and sets gpus)
  if (overrides.dp != null && overrides.gpus == null) cfg.gpus = cfg.pp * overrides.dp;
  cfg.dp = overrides.dp ?? Math.max(1, Math.round(cfg.gpus / cfg.pp));
  cfg.ep = Math.min(cfg.ep, cfg.dp);
  // batching derives from the global batch when given: one sequence per
  // microbatch (mbs = 1), m = GBS / dp
  if (cfg.globalBatch != null) {
    cfg.mbs = 1;
    cfg.microbatches = Math.max(1, Math.round(cfg.globalBatch / cfg.dp));
  }
  // cumulative level sets the base; individual refinement toggles override it
  cfg.refinements = { ...refinementsAt(cfg.level), ...(overrides.refinements ?? {}) };
  return cfg;
}

export function simulate(overrides = {}) {
  const cfg = defaultConfig(overrides);
  const hw = M.HARDWARE[cfg.hardware];
  const a = M.DSV3;
  const R = cfg.refinements;
  const tokens = cfg.mbs * cfg.seqLen;
  const gpus = cfg.gpus;
  const stepTokens = cfg.dp * cfg.microbatches * tokens;
  const modelFlops = M.modelFlopsPerToken(a, cfg.seqLen) * stepTokens;
  const peak = M.peakFlops(hw, cfg.dtype);

  if (cfg.level === 0) {
    const dur = modelFlops / gpus / peak * 1e6;
    const stats = finishStats(cfg, dur, dur, 0, modelFlops, gpus, peak, stepTokens, {});
    return {
      cfg, stats,
      trace: {
        meta: { cfg, stats },
        ranks: [{
          label: `all ${gpus} GPUs (idealized)`,
          tracks: [{
            name: 'compute', lanes: 1, slices: [{
              name: 'one step at the roofline', cat: 'gemm', ts: 0, dur, depth: 0,
              args: { flops: M.fmtNum(modelFlops / gpus) + ' FLOPs/GPU', peak: M.fmtNum(peak) + ' FLOP/s', note: 'no memory, no comm, no bubbles' },
            }],
          }],
        }],
      },
    };
  }

  // ---- pre-compute per-stage op tables --------------------------------------
  const stages = [];
  for (let s = 0; s < cfg.pp; s++) {
    const layers = M.stageLayerKinds(a, cfg.pp, s).map(l => ({
      ...l,
      specs: M.layerOps(l.kind, a, cfg.seqLen).map(spec => ({
        spec,
        F: M.opTimeUs(spec, tokens, cfg, hw, 'F'),
        B: M.opTimeUs(spec, tokens, cfg, hw, 'B'),
      })),
    }));
    if (s === 0) layers.unshift(wrapExtra('embed', M.EMBED_OP(a), tokens, cfg, hw));
    if (s === cfg.pp - 1) layers.push(wrapExtra('head', null, tokens, cfg, hw, M.HEAD_OPS(a)));
    stages.push(layers);
  }
  // Recompute replay: ops marked `recompute` rerun their forward during
  // backward — extra compute (and a2a, under full checkpointing) per layer.
  const replay = {};
  for (const kind of ['moe', 'dense']) {
    const ana = layerAnalysis(kind, cfg);
    let us = 0;
    for (const id of ana.replayed) {
      const n = ana.byId[id];
      if (n.opKind === 'comm') continue; // charged below at a2a cost
      const rate = n.opKind === 'attn' ? hw.flops.bf16 * hw.attnEff
        : n.opKind === 'matmul' ? M.gemmRate(hw, cfg.dtype)
          : null;                        // vector: bandwidth-bound
      us += rate ? n.flopsTok * tokens / rate * 1e6
        : 3 * n.outBytes * tokens / (hw.hbm * hw.hbmEff) * 1e6;
    }
    replay[kind] = { us, commIds: ana.replayComm, flops: ana.replayFlopsTok * tokens };
  }

  const a2aF = M.a2aTimeUs(tokens, M.actBytes(cfg.dtype), cfg, hw); // dispatch (fp8 if mxfp8)
  const a2aC = M.a2aTimeUs(tokens, 2, cfg, hw);                     // combine & grads in bf16
  const p2p = M.p2pTimeUs(tokens, cfg, hw);

  // ---- op graph generation ---------------------------------------------------
  const ops = [];             // {id, rank, track, name, cat, us, flops, deps, group, groupSize, mb, phase, layer, gapBeforeUs}
  const queues = new Map();   // `${rank}|${track}` -> [opId]
  const keyIds = new Map();
  const nr = cfg.dpRanksToSim;
  const add = (rank, track, o) => {
    const id = ops.length;
    ops.push({ id, rank, track, deps: [], ...o });
    const qk = `${rank}|${track}`;
    if (!queues.has(qk)) queues.set(qk, []);
    queues.get(qk).push(id);
    if (o.key) keyIds.set(o.key, id);
    return id;
  };

  for (let s = 0; s < cfg.pp; s++) {
    for (let r = 0; r < nr; r++) {
      const rank = s * nr + r;
      const rng = M.mulberry32(cfg.seed + rank * 7919);
      const jit = (us) => {
        let t = us;
        if (R.jitter) t *= Math.max(0.3, 1 + cfg.opSigma * (rng() + rng() + rng() + rng() - 2) * 1.73);
        if (R.overhead) t += cfg.kernelOverheadUs;
        return t;
      };
      const rsIds = [];

      if (R.dpComm && cfg.zero >= 3) { // FSDP all-gathers queue up front; layer k's first use waits on AG k
        stages[s].forEach((l, li) => {
          const { dense, expert } = bucketParams(a, cfg, s, l, li, stages[s].length);
          const us = M.collTimeUs(dense, M.actBytes(cfg.dtype), cfg.dp, hw)
            + M.collTimeUs(expert, M.actBytes(cfg.dtype), cfg.dp / cfg.ep, hw);
          add(rank, 'coll', {
            name: `allgather ${l.label ?? 'L' + l.index}`, cat: 'fsdp', us: jit(us),
            group: `AG:${s}:${li}`, groupSize: nr, key: `AG:${s}:${r}:${li}`,
          });
        });
      }

      const emitPhase = (phase, mb) => {
        const layers = phase === 'F' ? stages[s] : [...stages[s]].reverse();
        let pending = []; // cross-track deps to attach to the next compute op
        let first = true, lastId = -1;
        // p2p / cross-stage dependency
        if (R.pipeline) {
          const upKey = phase === 'F' ? (s > 0 && `F:${s - 1}:${r}:${mb}`) : (s < cfg.pp - 1 && `B:${s + 1}:${r}:${mb}`);
          if (upKey) {
            if (R.dpComm) pending.push(add(rank, 'pp', { name: `recv ${phase}${mb}`, cat: 'p2p', us: jit(p2p), deps: [upKey], mb, phase }));
            else pending.push(upKey);
          }
        }
        for (const l of layers) {
          if (phase === 'B' && replay[l.kind]) {
            const rp = replay[l.kind];
            const commUs = R.epComm && l.kind === 'moe'
              ? rp.commIds.reduce((t, c) => t + (c === 'dispatch' ? a2aF : a2aC), 0) : 0;
            if (rp.us + commUs > 1) {
              lastId = add(rank, 'compute', {
                name: 'recompute (fwd replay)', cat: 'gemm', us: jit(rp.us + commUs),
                flops: rp.flops, deps: pending, mb, phase, layer: l.label ?? l.index,
              });
              pending = [];
            }
          }
          const specs = phase === 'F' ? l.specs : [...l.specs].reverse();
          for (const { spec, F, B } of specs) {
            const us = phase === 'F' ? F : B;
            if (us <= 0) continue;
            const moe = l.kind === 'moe' && R.epComm;
            if (moe && spec.name === 'routed_experts' && phase === 'B') {
              // bwd of combine: scatter output grads back to experts
              const cg = add(rank, 'a2a', {
                name: `a2a combine-grad L${l.index}`, cat: 'a2a', us: jit(a2aC),
                deps: [...pending, ...(lastId >= 0 ? [lastId] : [])],
                group: `cg:${s}:${mb}:${l.index}`, groupSize: nr, mb, phase,
              });
              pending = [cg];
            }
            const deps = pending; pending = [];
            const mult = phase === 'B' ? (spec.bwdMult ?? 2) : 1;
            lastId = add(rank, 'compute', {
              name: spec.name, cat: spec.cat, us: jit(us), flops: spec.ftok * tokens * mult,
              deps, mb, phase, layer: l.label ?? l.index,
              gapBeforeUs: first && R.overhead ? cfg.hostGapUs : 0,
            });
            first = false;
            if (R.dpComm && cfg.zero >= 3 && mb === 0 && phase === 'F') {
              ops[lastId].deps.push(`AG:${s}:${r}:${layers.indexOf(l)}`);
            }
            if (moe && phase === 'F' && spec.name === 'router') {
              pendingDispatch = add(rank, 'a2a', {
                name: `a2a dispatch L${l.index}`, cat: 'a2a', us: jit(a2aF), deps: [lastId],
                group: `fd:${s}:${mb}:${l.index}`, groupSize: nr, mb, phase,
              });
            }
            if (moe && spec.name === 'routed_experts') {
              if (phase === 'F') {
                ops[lastId].deps.push(pendingDispatch);
                pending.push(add(rank, 'a2a', {
                  name: `a2a combine L${l.index}`, cat: 'a2a', us: jit(a2aC), deps: [lastId],
                  group: `fc:${s}:${mb}:${l.index}`, groupSize: nr, mb, phase,
                }));
              } else {
                pending.push(add(rank, 'a2a', {
                  name: `a2a dispatch-grad L${l.index}`, cat: 'a2a', us: jit(a2aC), deps: [lastId],
                  group: `dg:${s}:${mb}:${l.index}`, groupSize: nr, mb, phase,
                }));
              }
            }
          }
          if (phase === 'B' && R.dpComm && mb === cfg.microbatches - 1) {
            const li = stages[s].indexOf(l);
            const { dense, expert } = bucketParams(a, cfg, s, l, li, stages[s].length);
            // FSDP reduce-scatters grads; ZeRO-0/1 ring all-reduce costs 2x the traffic
            const us = (M.collTimeUs(dense, 2, cfg.dp, hw) + M.collTimeUs(expert, 2, cfg.dp / cfg.ep, hw))
              * (cfg.zero >= 3 ? 1 : 2);
            rsIds.push(add(rank, 'coll', {
              name: `${cfg.zero >= 3 ? 'reducescatter' : 'allreduce'} ${l.label ?? 'L' + l.index}`,
              cat: 'fsdp', us: jit(us),
              deps: [lastId], group: `RS:${s}:${li}`, groupSize: nr, mb, phase,
            }));
          }
        }
        // if the stage ends in a MoE layer, a trailing a2a op is the true output
        for (const p of pending) if (typeof p === 'number') lastId = p;
        keyIds.set(`${phase}:${s}:${r}:${mb}`, lastId);
      };
      let pendingDispatch = -1;

      // 1F1B schedule for this stage
      const m = cfg.microbatches, wu = Math.min(cfg.pp - 1 - s, m);
      const items = [];
      for (let j = 0; j < wu; j++) items.push(['F', j]);
      for (let j = wu; j < m; j++) items.push(['F', j], ['B', j - wu]);
      for (let j = Math.max(m - wu, 0); j < m; j++) items.push(['B', j]);
      for (const [phase, mb] of items) {
        if (R.jitter && rng() < cfg.gcProb) {
          add(rank, 'compute', { name: 'python GC pause', cat: 'stall', us: cfg.gcMs * 1000 * (0.5 + rng()), mb, phase });
        }
        emitPhase(phase, mb);
      }
      if (R.dpComm) {
        const p = M.stageParams(a, cfg.pp, cfg.ep, s);
        const shard = cfg.zero >= 1
          ? p.dense / cfg.dp + p.expert / Math.max(1, cfg.dp / cfg.ep)
          : p.dense + p.expert;
        add(rank, 'compute', {
          name: 'optimizer step (adamw)', cat: 'optimizer',
          us: jit(shard * 20 / (hw.hbm * hw.hbmEff) * 1e6), deps: rsIds.slice(),
          mb: -1, phase: 'O',
        });
      }
    }
  }

  // resolve symbolic deps
  for (const op of ops) {
    op.deps = op.deps.map(d => {
      if (typeof d === 'number') return d;
      const id = keyIds.get(d);
      if (id === undefined) throw new Error(`unresolved dep ${d}`);
      return id;
    });
  }

  schedule(ops, queues);
  const trace = buildTrace(ops, cfg, nr);
  const { stepUs, busyUs, stallUs, byCat } = measure(trace);
  const stats = finishStats(cfg, stepUs, busyUs, stallUs, modelFlops, gpus, peak, stepTokens, byCat);
  trace.meta = { cfg, stats };
  return { cfg, trace, stats };
}

function wrapExtra(label, single, tokens, cfg, hw, list) {
  const specs = (list ?? [single]).map(spec => ({
    spec,
    F: M.opTimeUs(spec, tokens, cfg, hw, 'F'),
    B: M.opTimeUs(spec, tokens, cfg, hw, 'B'),
  }));
  return { label, index: label, kind: label, specs };
}

// FSDP bucket sizes: split stage params evenly by layer bucket (embed/head
// buckets carry their own vocab-sized matrices via stageParams already).
function bucketParams(a, cfg, s, l, li, nBuckets) {
  if (l.kind === 'embed') return { dense: a.hidden * a.vocab, expert: 0 };
  if (l.kind === 'head') return { dense: a.hidden * a.vocab + a.hidden, expert: 0 };
  const dense = M.attnLayerParams(a) + M.layerNormParams(a) + (l.kind === 'dense'
    ? 3 * a.hidden * a.denseInter
    : (a.hidden + 1) * a.routedExperts + a.sharedExperts * 3 * a.hidden * a.moeInter);
  const expert = l.kind === 'moe' ? (a.routedExperts / cfg.ep) * 3 * a.hidden * a.moeInter : 0;
  return { dense, expert };
}

// ---- scheduler ---------------------------------------------------------------
function schedule(ops, queues) {
  const trackTime = new Map();
  const heads = new Map();
  const groups = new Map();
  let remaining = ops.length;
  while (remaining > 0) {
    let progress = false;
    for (const [qk, q] of queues) {
      let idx = heads.get(qk) ?? 0;
      while (idx < q.length) {
        const op = ops[q[idx]];
        if (op.end !== undefined) { idx++; continue; } // scheduled via group fire
        if (op.deps.some(d => ops[d].end === undefined)) break;
        const depEnd = op.deps.reduce((t, d) => Math.max(t, ops[d].end), 0);
        const localStart = Math.max((trackTime.get(qk) ?? 0) + (op.gapBeforeUs ?? 0), depEnd);
        if (op.group) {
          let g = groups.get(op.group);
          if (!g) groups.set(op.group, g = new Map());
          if (!g.has(op.id)) { g.set(op.id, localStart); progress = true; }
          if (g.size < op.groupSize) break; // barrier: wait for peers
          const start = Math.max(...g.values());
          for (const oid of g.keys()) {
            const o = ops[oid];
            o.start = start; o.end = start + o.us;
            trackTime.set(`${o.rank}|${o.track}`, o.end);
            remaining--;
          }
          idx++; progress = true;
        } else {
          op.start = localStart; op.end = localStart + op.us;
          trackTime.set(qk, op.end);
          idx++; remaining--; progress = true;
        }
      }
      heads.set(qk, idx);
    }
    if (!progress) throw new Error('scheduler deadlock (bad dependency graph)');
  }
}

// ---- slice building ----------------------------------------------------------
const TRACK_ORDER = ['compute', 'a2a', 'coll', 'pp'];
function buildTrace(ops, cfg, nr) {
  const byRank = new Map();
  for (const op of ops) {
    if (!byRank.has(op.rank)) byRank.set(op.rank, new Map());
    const tr = byRank.get(op.rank);
    if (!tr.has(op.track)) tr.set(op.track, []);
    tr.get(op.track).push(op);
  }
  const ranks = [];
  for (const [rank, tracks] of [...byRank.entries()].sort((x, y) => x[0] - y[0])) {
    const s = Math.floor(rank / nr), r = rank % nr;
    const out = { label: `pp${s}` + (nr > 1 ? ` dp${r}` : ''), tracks: [] };
    for (const tname of TRACK_ORDER) {
      if (!tracks.has(tname)) continue;
      const tops = tracks.get(tname).sort((x, y) => x.start - y.start);
      out.tracks.push(tname === 'compute'
        ? { name: 'compute', lanes: cfg.granularity === 'phase' ? 1 : 2, slices: computeSlices(tops, cfg) }
        : { name: tname, lanes: 1, slices: tops.map(o => sliceOf(o, o.name, 0)) });
    }
    ranks.push(out);
  }
  return { ranks };
}

function sliceOf(o, name, depth) {
  const args = { mb: o.mb, phase: o.phase };
  if (o.flops) args.tflops = +(o.flops / o.us / 1e6).toFixed(1);
  return { name, cat: o.cat, ts: o.start, dur: o.end - o.start, depth, args };
}

function computeSlices(tops, cfg) {
  const slices = [];
  const leafDepth = cfg.granularity === 'phase' ? 0 : 1;
  const keyOf = (o) => o.cat === 'stall' || o.cat === 'optimizer' || cfg.granularity === 'op' ? `o${o.id}`
    : cfg.granularity === 'layer' ? `${o.mb}:${o.phase}:${o.layer}` : `${o.mb}:${o.phase}`;
  const nameOf = (o) => cfg.granularity === 'op' || o.cat === 'stall' || o.cat === 'optimizer' ? o.name
    : cfg.granularity === 'layer' ? (typeof o.layer === 'number' ? `L${o.layer}` : o.layer) : `${o.phase}${o.mb}`;
  let cur = null;
  const flush = () => {
    if (!cur) return;
    // merged slices take the dominant category; the mix goes in the tooltip
    const mix = Object.entries(cur.catUs).sort((a, b) => b[1] - a[1]);
    cur.cat = mix[0][0];
    if (mix.length > 1) cur.args.mix = mix.map(([c, us]) => `${c} ${Math.round(us / cur.dur * 100)}%`).join(' · ');
    slices.push(cur); cur = null;
  };
  for (const o of tops) {
    const k = keyOf(o);
    if (cur && cur.key === k && o.start - (cur.ts + cur.dur) < 0.5) {
      cur.dur = o.end - cur.ts;
      cur.flops += o.flops ?? 0;
      cur.catUs[o.cat] = (cur.catUs[o.cat] ?? 0) + o.us;
      cur.args.tflops = +(cur.flops / cur.dur / 1e6).toFixed(1);
    } else {
      flush();
      cur = { key: k, flops: o.flops ?? 0, catUs: { [o.cat]: o.us }, ...sliceOf(o, nameOf(o), leafDepth) };
    }
  }
  flush();
  // phase-level parent slices spanning each microbatch F/B (includes waits)
  if (cfg.granularity !== 'phase') {
    const phases = new Map();
    for (const o of tops) {
      if (o.mb < 0 || o.cat === 'stall') continue;
      const k = `${o.phase}${o.mb}`;
      const p = phases.get(k) ?? { ts: Infinity, end: 0, busy: 0 };
      p.ts = Math.min(p.ts, o.start); p.end = Math.max(p.end, o.end); p.busy += o.us;
      phases.set(k, p);
    }
    for (const [name, p] of phases) {
      slices.push({
        name, cat: 'phase', ts: p.ts, dur: p.end - p.ts, depth: 0,
        args: { busy: M.fmtUs(p.busy), waiting: M.fmtUs(p.end - p.ts - p.busy) },
      });
    }
  }
  return slices.map(({ key, flops, catUs, ...s }) => s).sort((a, b) => a.ts - b.ts || b.dur - a.dur);
}

// ---- stats -------------------------------------------------------------------
function measure(trace) {
  let t0 = Infinity, t1 = 0, busyUs = 0, stallUs = 0, nCompute = 0;
  const byCat = {};
  for (const rank of trace.ranks) {
    for (const track of rank.tracks) {
      for (const s of track.slices) {
        t0 = Math.min(t0, s.ts); t1 = Math.max(t1, s.ts + s.dur);
        if (s.cat === 'phase') continue;
        byCat[s.cat] = (byCat[s.cat] ?? 0) + s.dur;
        if (track.name === 'compute') {
          if (s.cat === 'stall') stallUs += s.dur;
          else busyUs += s.dur;
        }
      }
      if (track.name === 'compute') nCompute++;
    }
  }
  return { stepUs: t1 - t0, busyUs: busyUs / Math.max(1, nCompute), stallUs, byCat };
}

function finishStats(cfg, stepUs, busyUs, stallUs, modelFlops, gpus, peak, stepTokens, byCat) {
  return {
    stepUs,
    mfu: modelFlops / gpus / (stepUs * 1e-6) / peak,
    tokPerSecPerGpu: stepTokens / gpus / (stepUs * 1e-6),
    bubbleFrac: 1 - busyUs / stepUs,
    stallUs, byCat,
    modelFlops, gpus, stepTokens,
    mem: memoryUsage(cfg),
  };
}
