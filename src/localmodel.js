// The local-lens model: what ONE GPU holds under the essay's fiat
// parallelism — PP stage layout, virtual-stage placement, 1F1B/DualPipeV
// in-flight admission, per-component byte rates, and the activation buckets
// the fit charts price from. Pure module (no DOM): Node consumers (sanity,
// goldens) import it directly; the renderers in viewer.js build on it.
import { DSV3 } from './model.js';
import { PARAMS } from './params.js';
import { resolveMatmuls, MATMULS } from './recipes.js';
import { blockGraph, analyze, RECOMPUTE_PRESETS } from './blockgraph.js';

// byte components of the per-op strips (optim / consolidated variants), in the
// memory-bars stacked order with the memory-bars segment colors — the strips
// pre-teach the bar. bpp = bytes per parameter.
export const BYTE_COMPS = [
  // zthresh: the ZeRO level at which this component shards over its
  // replication group (1 = optimizer, 2 = + gradients, 3 = + weights/FSDP)
  { prop: 'showWeights', color: '#2a78d6', bpp: 2, zthresh: 3, label: 'weights (2 B/param)' },
  { prop: 'showGrads', color: '#eb6834', bpp: 4, zthresh: 2, label: 'gradients (fp32, 4 B/param)' },
  { prop: 'showOptim', color: '#1baf7a', bpp: 8, zthresh: 1, label: 'optimizer states (8 B/param)' },
];

// per-op activation buckets for the fit chart's acts breakdown — the same
// stashes the wire chips draw, grouped Haziza-style. 'other' is a catch-all
// remainder, so the list always PARTITIONS savedBytes (the visual audit's
// decomposition rule enforces the sum exactly).
export const ACT_BUCKETS = [
  { label: 'x0, x1 (residual)', ids: ['x0', 'x1'] },
  { label: 'norm outs', ids: ['norm1', 'norm2'] },
  { label: 'mla latents', ids: ['qkv_down', 'q_norm', 'kv_norm'] },
  { label: 'q · k,v', ids: ['q_up', 'kv_up', 'rope_q', 'rope_kv'] },
  { label: 'attention out', ids: ['attn'] },
  { label: 'router state', ids: ['router'] },
  { label: 'dispatched tokens', ids: ['dispatch'] },
  // the stash is the QUANTIZED copy (the 'quant' node); the GEMM's own bf16
  // output is never stashed (its mark is tied) — listed so the bucket partitions
  { label: 'gate, up (routed+sh)', ids: ['quant', 'gate_up'] },
  { label: 'swiglu out', ids: ['swiglu'] },
  { label: 'other', ids: [] },
];
// bucket an analysis's per-tensor stash bytes; the remainder lands in 'other'
export const actBucketsOf = (ana2) => {
  const named = ACT_BUCKETS.map((b) => b.ids.reduce((t, id) => t + (ana2.savedById?.[id] ?? 0), 0));
  named[ACT_BUCKETS.length - 1] = ana2.savedBytes - named.reduce((a, b) => a + b, 0);
  return named;
};

// fiat parallelism for the `local` variant (what one GPU holds): 2048 GPUs;
// PP degree, EP width, and ZeRO-1 are the layer's knobs. Layers split over
// the PP stages by a contiguous floor split: stage 0 gets the embedding (+
// the dense blocks while they last), the last stage gets final norm + lm head.
// PP is {1, 8} ONLY: PP8 under DualPipeV = 16 virtual chunks — the
// DualPipeV-equivalent of DSv3's published 16 pipeline stages (half the
// ranks, same virtual depth). Intermediate depths are jettisoned: a good
// zero-bubble partition is sensitive to the front-of-pipe imbalance
// (dense layers, emb/head), so "the PP4 schedule" isn't even well-defined
// without choices this essay doesn't want to defend.
export const PP_CHOICES = [1, 8];
export const LOCAL_PAR = { world: 2048, pp: 8 };   // .pp = the default degree
// an AUTHORED config (snapshot from/to, deck steps) is complete by fiat:
// unlisted knobs mean these neutral nothing-applied defaults, never
// "whatever the widget's live defaults happen to be" — a published figure
// must not drift when the interactive defaults do
export const CFG_DEFAULTS = { world: 2048, pp: 1, ep: 1, tp: 1, zero: 0, sched: '1f1b', hw: 'h100', a2a: false, gradB: 4, fp8Params: false };   // vpp/fold are derived from pp (DualPipeV); sched 'interleaved' takes vpp + layout
// virtual-stage placement: with VPP = vpp chunks per rank the chain is
// vpp·pp virtual stages deep; 'wrap' places stage v on rank v mod pp
// (Megatron interleaving), 'reflect' bounces each pass (ZB-V / DualPipeV:
// rank 0 → pp−1 → 0 → …), so with even vpp the chain both starts AND ends
// on rank 0. DualPipeV ≡ vpp 2 + reflect.
export const vstagesOf = (r, pp, vpp = 1, fold = 'reflect') =>
  Array.from({ length: vpp }, (_, c) =>
    c * pp + (fold === 'reflect' && c % 2 ? pp - 1 - r : r));
// ---- Megatron pipeline layouts (the Blackwell post's schedule) ------------
// Megatron-Core's --pipeline-model-parallel-layout grammar: E = embedding,
// t = transformer layer, m = MTP layer, L = loss (final norm + lm head);
// '|' ends a chunk (virtual stage), X*N repeats a token, (…)*N repeats a
// group, separators included — "Et|(tt|)*30L" is 32 chunks. Chunks are
// dealt round-robin: chunk i lives on pp rank i mod PP as its virtual
// stage ⌊i/PP⌋ (the 'wrap' fold). Parsed chunks: {emb, layers, mtp, head}.
export const parseLayout = (str) => {
  let s = String(str).replace(/\s+/g, '');
  s = s.replace(/\(([^()]*)\)\*(\d+)/g, (_, g, n) => g.repeat(+n));
  s = s.replace(/([EtmL])\*(\d+)/g, (_, t, n) => t.repeat(+n));
  const parts = s.split('|');
  if (parts[parts.length - 1] === '') parts.pop();   // a trailing '|' ends the last chunk
  return parts.map((c) => ({
    emb: c.includes('E'), layers: (c.match(/t/g) ?? []).length,
    mtp: (c.match(/m/g) ?? []).length, head: c.includes('L'),
  }));
};
// the layout a Megatron config means: an explicit string, else NVIDIA's
// Megatron-Bridge default for DeepSeek-V3 (set_deepseek_v3_pipeline_model_
// parallel_layout, recipes/deepseek/h100/deepseek_v3.py:50-68): 16 chunks =
// "Et*4|(t*4|)*14tmL" (E + 4 layers · fourteen × 4 · 1 layer + MTP + loss);
// 32 chunks = the H100 recipes' "Et|(tt|)*30mL"; other depths (no Bridge
// default) deal the 64 slots E · 61 t · m · L as evenly as PP·VP allows
export const MEGATRON_LAYOUTS = { 16: 'Et*4|(t*4|)*14tmL', 32: 'Et|(tt|)*30mL' };
export const megatronLayout = (pp, vpp) => {
  const D = pp * vpp;
  if (MEGATRON_LAYOUTS[D]) return MEGATRON_LAYOUTS[D];
  const per = Math.floor(64 / D);
  let rem = 64 - per * D;
  const toks = ['E', ...Array(61).fill('t'), 'm', 'L'];
  const out = []; let k = 0;
  for (let i = 0; i < D; i++) { const n = per + (rem > 0 ? 1 : 0); rem--; out.push(toks.slice(k, k + n).join('')); k += n; }
  return out.join('|');
};
// schedule GEOMETRY from a config: DualPipeV (02's law) derives vpp/fold from
// pp — pp > 1 → 2 chunks per rank, reflect; Megatron's interleaved 1F1B (or
// any config already on the wrap fold, e.g. its ×1 mb) takes vpp + a layout
// (default megatronLayout) with the wrap fold, plus the a2a-overlap flag. An
// unknown layout for a new depth re-derives the default.
export const schedGeom = (c) => {
  if (!(c.sched === 'interleaved' || c.fold === 'wrap')) return { vpp: c.pp > 1 ? 2 : 1, fold: 'reflect', layout: null, a2a: false };
  const vpp = c.pp === 1 ? 1 : (c.vpp ?? 1);
  let layout = c.layout ?? null;
  if (layout && parseLayout(layout).length !== c.pp * vpp) layout = null;
  return { vpp, fold: 'wrap', layout: layout ?? megatronLayout(c.pp, vpp), a2a: !!c.a2a };
};
// Megatron's interleaved 1F1B, as the sequence of LIVE stashes on one rank.
// The per-rank program is static (schedules.py: forwards chunk-major in
// groups of PP microbatches; warmup W = 2(PP−r−1) + (VP−1)·PP chunk-forwards,
// +1 with the 1F1B all-to-all overlap NVIDIA's MXFP8 recipes enable; then
// strict 1F1B pairs; then cooldown), and a stash is born at this rank's own
// F and freed at its own B — so the live multiset after each op depends only
// on the op ORDER, never on cross-rank timing. Returns one live-per-chunk
// vector per op. VP1 = the plain schedule (warmup PP−r−1); PP1 = no pipeline.
export const ilvLive = (r, pp, vpp = 1, a2a = false, m = pp * (vpp + 3)) => {
  const r2 = Math.min(r, pp - 1);
  if (pp === 1) return [Array(vpp).fill(1)];
  const total = m * vpp, G = pp * vpp;
  const W = Math.min(vpp === 1 ? pp - r2 - 1 : (pp - r2 - 1) * 2 + (vpp - 1) * pp + (a2a ? 1 : 0), total);
  const chunkOf = (k, fwd) => { const c = Math.floor((k % G) / pp); return fwd ? c : vpp - 1 - c; };
  const live = Array(vpp).fill(0), out = [];
  const F = (k) => { live[chunkOf(k, true)]++; out.push([...live]); };
  const B = (k) => { live[chunkOf(k, false)]--; out.push([...live]); };
  for (let k = 0; k < W; k++) F(k);
  for (let i = 0; i < total - W; i++) { F(W + i); B(i); }
  for (let i = total - W; i < total; i++) B(i);
  return out;
};
// the PEAK moment on rank r: the op after which Σ_c live_c × (its chunk's
// MoE layers × rateM + dense layers × rateD) is largest (a light last chunk —
// 1 layer + MTP + loss — makes the byte peak and the count peak differ).
// Returns the per-chunk live counts there and each layer KIND's mean over
// its own layers — the numbers the cells charge. Without a layout every
// chunk weighs the same (the count peak: PP·VP + PP − 2r chunk-stashes with
// the a2a overlap, one fewer without).
export const ilvPeak = (r, pp, vpp = 1, layout = null, { a2a = false, rateM = 1, rateD = 1 } = {}) => {
  const segs = layout ? ppStage(r, pp, vpp, 'wrap', layout).segs : Array(vpp).fill({ moe: 1, dense: 0 });
  const w = segs.map((sg) => sg.moe * rateM + sg.dense * rateD);
  let best = -1, live = null;
  for (const l of ilvLive(r, pp, vpp, a2a)) {
    const v = l.reduce((t, n, c) => t + n * w[c], 0);
    if (v > best) { best = v; live = l; }
  }
  const mean = (k) => { const L = segs.reduce((t, sg) => t + sg[k], 0); return L ? segs.reduce((t, sg, c) => t + sg[k] * live[c], 0) / L : null; };
  const n = live.reduce((a, b) => a + b, 0);
  return { live, n, moe: mean('moe') ?? n / vpp, dense: mean('dense') ?? n / vpp, segs };
};
const layoutStage = (s, pp, vpp, layout) => {
  const ch = parseLayout(layout);
  const D = pp * vpp;
  if (ch.length !== D) throw new Error(`layout "${layout}" has ${ch.length} chunks; PP${pp}×VP${vpp} needs ${D}`);
  const starts = []; let acc = 0;
  for (const c of ch) { starts.push(acc); acc += c.layers; }
  const seg = (i) => {
    const lo = starts[i], hi = lo + ch[i].layers;
    const dense = Math.max(0, Math.min(hi, 3) - Math.min(lo, 3));
    return { lo, hi, layers: hi - lo, dense, moe: hi - lo - dense, mtp: ch[i].mtp, emb: ch[i].emb, head: ch[i].head };
  };
  const vs = vstagesOf(Math.min(s, pp - 1), pp, vpp, 'wrap');
  const segs = vs.map(seg);
  const sum = (k) => segs.reduce((t, g) => t + g[k], 0);
  return { segs, layers: sum('layers'), dense: sum('dense'), moe: sum('moe'), mtp: sum('mtp'),
    emb: segs.some((g) => g.emb), head: segs.some((g) => g.head) };
};
export const ppStage = (s, pp = LOCAL_PAR.pp, vpp = 1, fold = 'reflect', layout = null) => {
  if (layout) return layoutStage(s, pp, vpp, layout);
  const D = vpp * pp;
  // SLOT model (author's fiat): the embedding and the lm head each count a
  // layer's worth when balancing, so 63 slots (emb + 61 layers + head) are
  // split contiguously with the +1 remainders handed out FRONT-first — at
  // D=16 that is exact with no odd chunk out: chunk 0 = emb + 3 layers,
  // fourteen interiors × 4, and the last = 2 layers + head (the head's
  // chunk gives up the layer). Every slot is in exactly one chunk, so the
  // partition stays exact.
  const sizes = Array(D).fill(Math.floor(63 / D));
  let rem = 63 - Math.floor(63 / D) * D;
  for (let k = 0; k < D && rem > 0; k++) { sizes[k]++; rem--; }
  const bounds = [0];
  for (const z of sizes) bounds.push(bounds[bounds.length - 1] + z);
  const seg = (i) => {
    const s0 = bounds[i], s1 = bounds[i + 1];             // slot range
    const lo = Math.max(0, s0 - 1), hi = Math.max(lo, Math.min(61, s1 - 1));   // layer range (slot 0 = emb, slot 62 = head)
    const dense = Math.max(0, Math.min(hi, 3) - Math.min(lo, 3));
    return { lo, hi, layers: hi - lo, dense, moe: hi - lo - dense,
      emb: s0 === 0 && s1 > 0, head: s1 === 63 && s0 < 63 };
  };
  const vs = vstagesOf(s, pp, vpp, fold);
  const segs = vs.map(seg);
  const sum = (k) => segs.reduce((t, g) => t + g[k], 0);
  return { segs, layers: sum('layers'), dense: sum('dense'), moe: sum('moe'),
    emb: vs.includes(0), head: vs.includes(D - 1) };
};

// save-everything bf16 activation bytes for ONE layer × one 4096-token
// microbatch (the local model's activation quantum), per KIND — a dense
// layer stashes no router state or dispatched tokens, so pricing every
// layer at the MoE rate would overstate the dense front
const ACT_LAYER_B = {};
export const actLayerBytes = (kind = 'moe') => ACT_LAYER_B[kind] ??=
  analyze(blockGraph(kind, DSV3, resolveMatmuls({ recipe: 'bf16' }), 4096), RECOMPUTE_PRESETS.none, false).savedBytes * 4096;
// 1F1B admission per virtual stage: stage v of a D-deep chain holds D − v
// forward chunk-stashes at steady state (warmup depth; assumes ≥ D
// microbatches per step). A rank's residency in microbatch-equivalents is
// the sum over its hosted virtual stages, ÷ vpp (each chunk is 1/vpp of the
// rank's layers). vpp 1 → the plain 1F1B staircase pp − s; vpp 2 + reflect
// (DualPipeV) → a UNIFORM pp + ½ on every rank (the two depths always sum
// to 2pp+1 — 8.5 at PP8, vs the DSv3 paper's coarse PP+1 bound); wrap
// (Megatron interleaving) concentrates at rank 0: pp(vpp+1)/2 − s.
// 'one' = a single microbatch in flight.
// 'interleaved' (Megatron): ilvPeak's numbers — the rank law
// (PP·VP + PP − 2r [−1 without a2a overlap]) / VP in microbatch-equivalents
// without a layout, and with a layout AND a layer kind the per-kind mean at
// the byte-peak moment (the surplus stashes sit on the early chunks, and
// the kinds are not spread evenly over them), the number the cells charge.
export const inflightOf = (sched, s, pp, vpp = 1, fold = 'reflect', layout = null, kind = null, opts = {}) => {
  if (sched === 'one') return 1;
  if (sched === 'interleaved') {
    const pk = ilvPeak(s, pp, vpp, layout, { rateM: actLayerBytes('moe'), rateD: actLayerBytes('dense'), ...opts });
    return kind ? pk[kind] : pk.n / vpp;
  }
  const D = vpp * pp, s2 = Math.min(s, pp - 1);
  return vstagesOf(s2, pp, vpp, fold).reduce((t, v) => t + (D - v), 0) / vpp;
};

// the PP stage holding the most resident bytes under the local model (all
// components on, vocab counted on the end stages, activations under the
// schedule) — the default stage to show: the fully loaded rank.
export const peakStage = (pp, ep, zero, world = LOCAL_PAR.world, sched = '1f1b', vpp = 1, fold = 'reflect', layout = null, a2a = false, tp = 1) => {
  const dp = world / pp / tp;   // TP shards the non-expert params; its stashes divide by TP too (uniformly — the peak pick is unaffected)
  const bpp = (cls) => BYTE_COMPS.reduce((t, c) =>
    t + (zero >= c.zthresh ? c.bpp / (cls === 'e' ? dp / ep : dp) : c.bpp), 0);
  const moeExp = PARAMS.expert * DSV3.routedExperts;
  const dB = PARAMS.denseBlock * bpp('d');
  const mB = (PARAMS.moeBlock - moeExp) * bpp('d') + (moeExp / ep) * bpp('e');
  let best = 0, bestV = -1;
  for (let s2 = 0; s2 < pp; s2++) {
    const g = ppStage(s2, pp, vpp, fold, layout);
    const v = g.dense * dB + g.moe * mB
      + (((g.emb ? 1 : 0) + (g.head ? 1 : 0)) * PARAMS.embed * bpp('d'))
      + g.dense * actLayerBytes('dense') * inflightOf(sched, s2, pp, vpp, fold, layout, 'dense', { a2a })
      + g.moe * actLayerBytes('moe') * inflightOf(sched, s2, pp, vpp, fold, layout, 'moe', { a2a });
    if (v > bestV) { bestV = v; best = s2; }
  }
  return best;
};

// Daniel Haziza's roofline config (the analysis the essay credits) as a
// one-click cross-check preset for the full model: dsv3-style fp8 GEMMs
// with a BF16 attn-out stash (no E5M6), e4m3+ᵀ-RESIDENT params, and his
// exact stash policy — FFN outputs + attn out + norm2 kept; the latents,
// x1 and the router state replayed.
// canonical state signatures (recipe recognition + the Haziza button)
export const mmSig = (m) => MATMULS.map((x) => m[x.id]).concat(m.swiglu_in ?? 'bf16').join(',');
export const markSig = (m) => Object.keys(m).filter((k) => m[k] === true).sort().join(',');
export const HAZIZA_CFG = {
  matmuls: { qkv_down: 'e4m3', q_up: 'e4m3', kv_up: 'e4m3', attn: 'bf16', o_proj: 'bf16',
    router: 'fp32', ffn_gate_up: 'e4m3', ffn_down: 'e4m3', lm_head: 'bf16', swiglu_in: 'e4m3' },
  marks: { gate_up: true, ffn_down: true, combine: true, moe_add: true, attn: true, norm2: true, dispatch: true },
  transposed: false, fp8Params: true,
  ep: 64, pp: 8, zero: 1, world: 2048, stage: 1, sched: '1f1b',
};
