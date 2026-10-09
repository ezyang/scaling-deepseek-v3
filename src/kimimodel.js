// Kimi K3: the pure model behind src/kimi.js's widgets and the roofline
// sheet's generator (no DOM, node-importable): architecture constants, the
// exact parameter tally, the block as an op graph for recompute planning
// (planned by src/blockgraph.js analyze) and Block AttnRes across a pipeline.
//
// Sources: moonshotai/Kimi-K3 config.json + modeling_kimi_linear.py (tensor
// shapes), the K3 tech report (topology, §5 recompute), the AttnRes paper
// (arXiv 2603.15031, §4.1 pipeline caching).

import { analyze, DTYPE_BYTES } from './blockgraph.js';
export const KIMI = {
  layers: 93, kdaLayers: 69, mlaLayers: 24, denseLayers: 1,   // layer 1 dense; MLA every 4th + layer 93
  hidden: 7168, vocab: 163840,
  heads: 96, headDim: 128,                    // KDA q/k/v and MLA nope/v head dims
  qkNope: 128, qkRope: 64, vHead: 128,        // MLA (NoPE: the 64 "rope" dims are never rotated)
  qRank: 1536, kvRank: 512,
  convK: 4, decayRank: 128,                   // KDA short conv + low-rank decay bottleneck
  denseInter: 33792,
  moeInter: 3072, moeLatent: 3584,            // routed experts live in the 3584-wide latent
  routedExperts: 896, topk: 16, sharedExperts: 2,
  attnResBlock: 12,
};

const A = KIMI;
const PJ = A.heads * A.headDim;   // 12288, the q/k/v/gate projection width

// exact parameter quantities, derived once (shapes from modeling_kimi_linear.py)
export const KPARAMS = (() => {
  const d = A.hidden;
  const qkv = 3 * d * PJ;
  const conv = 3 * PJ * A.convK;                                  // depthwise short convs, no bias
  const decay = d * A.decayRank + A.decayRank * PJ + PJ + A.heads; // f_a + f_b + dt_bias + A_log
  const beta = d * A.heads;
  const gate = d * PJ;                                            // full-rank output gate (KDA & MLA)
  const gateNorm = A.headDim;                                     // KDA's gated RMSNorm weight
  const oProj = PJ * d;
  const kdaAttn = qkv + conv + decay + beta + gate + gateNorm + oProj;
  const qDown = d * A.qRank, qNorm = A.qRank, qUp = A.qRank * A.heads * (A.qkNope + A.qkRope);
  const kvDown = d * (A.kvRank + A.qkRope), kvNorm = A.kvRank, kvUp = A.kvRank * A.heads * (A.qkNope + A.vHead);
  const mlaAttn = qDown + qNorm + qUp + kvDown + kvNorm + kvUp + gate + oProj;
  const routerWeight = A.routedExperts * d, routerBias = A.routedExperts;
  const expert = 3 * A.moeLatent * A.moeInter;
  const latDown = d * A.moeLatent, latUp = A.moeLatent * d, latNorm = A.moeLatent;
  const shared = 3 * d * (A.moeInter * A.sharedExperts);          // one fused MLP of 2× width
  const experts = A.routedExperts * expert + shared;
  const activeExperts = A.topk * expert + shared;
  const latents = latDown + latNorm + latUp;
  const moeFfn = routerWeight + routerBias + experts + latents;
  const activeMoeFfn = routerWeight + routerBias + activeExperts + latents;
  const denseFfn = 3 * d * A.denseInter;
  const normsBlk = 2 * d;                                         // input + post-attention RMSNorm
  const attnRes = 4 * d;                                          // 2 res norms + 2 pseudo-query projs
  const misc = normsBlk + attnRes;
  const embed = A.vocab * d;
  const kdaDenseBlock = kdaAttn + denseFfn + misc;
  const kdaMoeBlock = kdaAttn + moeFfn + misc;
  const mlaMoeBlock = mlaAttn + moeFfn + misc;
  const finalNorms = 3 * d;                                       // final RMSNorm + output attn-res norm+proj
  const nKdaMoe = A.kdaLayers - A.denseLayers, nMla = A.mlaLayers;
  const total = embed + kdaDenseBlock + nKdaMoe * kdaMoeBlock + nMla * mlaMoeBlock + finalNorms + embed;
  // "active / token" follows DeepSeek's corrected accounting (the convention
  // K3 descends from): the embedding lookup is not counted, the head is.
  const active = kdaDenseBlock + nKdaMoe * (kdaAttn + activeMoeFfn + misc)
    + nMla * (mlaAttn + activeMoeFfn + misc) + finalNorms + embed;
  return {
    qkv, conv, decay, beta, gate, gateNorm, oProj, kdaAttn,
    qDown, qNorm, qUp, kvDown, kvNorm, kvUp, mlaAttn,
    routerWeight, routerBias, expert, latDown, latUp, latNorm, latents, shared,
    experts, activeExperts, moeFfn, activeMoeFfn, denseFfn, normsBlk, attnRes, misc,
    embed, kdaDenseBlock, kdaMoeBlock, mlaMoeBlock, finalNorms,
    nKdaMoe, nMla, total, active,
  };
})();
const K = KPARAMS;

// ---- the block as an op graph, for recompute planning ----------------------
// Same node format and semantics as src/blockgraph.js (whose analyze() does
// the planning): marks {id: true} = save the op's output, anything unmarked
// replays in backward. Stash FORMATS are fixed at K3's recipe so that only
// the recompute choice varies: GEMM outputs and GEMM inputs block-wise FP8
// (MXFP8 bytes), attention-kernel inputs/outputs and the residual stream
// BF16, router state FP32. Under the K3 preset this reproduces the roofline
// sheet's policy-A stash row for row (σᴬ: tests/kimi.js pins it), plus the
// decay bottleneck's rank-128 intermediate, which the sheet drops.
// Outside the graph: the AttnRes block store (the mixes' other input, kept
// per microbatch, not per layer) and the chunked kernels' own state recompute.
const B8 = DTYPE_BYTES.mxfp8;
export function kimiGraph(kindA, kindF) {
  const h = A.hidden, nh = A.heads, l = A.moeLatent, I = A.moeInter, k = A.topk;
  const qk = A.qkNope + A.qkRope, moe = kindF === 'moe';
  const N = (id, opKind, inputs, tensor, elems, bytesPer, flopsTok, opts = {}) => ({
    id, label: tensor, opKind, inputs, tensor, elems, outBytes: elems * bytesPer, flopsTok,
    bucket: opts.bucket ?? 'mla', always: !!opts.always, nomark: !!opts.nomark,
    needsOwnOutput: !!opts.needsOwnOutput, aux: opts.aux ?? null, bwdNeeds: opts.bwdNeeds ?? null,
    dt: bytesPer === 2 ? 'bf16' : bytesPer === 4 ? 'fp32' : 'mxfp8',
  });
  const rstd = (n = 1) => ({ name: 'rstd', bytes: 4 * n });
  const attn = kindA === 'kda' ? [
    N('qkv', 'matmul', ['norm1'], 'q, k, v (pre-conv)', 3 * PJ, B8, 2 * h * 3 * PJ),
    N('conv', 'vector', ['qkv'], 'q, k, v', 3 * PJ, 2, 2 * 3 * PJ * A.convK),
    // the f_a → f_b bottleneck: f_b's wgrad needs the rank-128 intermediate (aux)
    N('decay', 'matmul', ['norm1'], 'α (decay)', PJ, B8, 2 * (h * A.decayRank + A.decayRank * PJ),
      { aux: { name: 'f_a out', bytes: 2 * A.decayRank } }),
    N('beta', 'matmul', ['norm1'], 'β', nh, B8, 2 * h * nh),
    N('gate1', 'matmul', ['norm1'], 'gate', PJ, B8, 2 * h * PJ),
    // chunk_kda takes the gate activation, L2Norm and β sigmoid in-kernel; its
    // backward re-derives the chunk states from q, k, v, g, β (not its output)
    N('kda_scan', 'attn', ['conv', 'decay', 'beta'], 'scan out', PJ, 2,
      nh * (6 * A.headDim * A.headDim + 10 * 64 * A.headDim)),
    N('gate_mul1', 'vector', ['kda_scan', 'gate1'], 'gated out', PJ, B8, 8 * PJ, { aux: rstd(nh) }),
  ] : [
    N('q_down', 'matmul', ['norm1'], 'q latent', A.qRank, B8, 2 * h * A.qRank),
    N('kv_down', 'matmul', ['norm1'], 'kv latent', A.kvRank + A.qkRope, B8, 2 * h * (A.kvRank + A.qkRope)),
    N('q_norm', 'vector', ['q_down'], 'norm(q latent)', A.qRank, B8, 8 * A.qRank, { aux: rstd() }),
    N('kv_norm', 'vector', ['kv_down'], 'norm(kv latent)', A.kvRank, B8, 8 * A.kvRank, { aux: rstd() }),
    N('q_up', 'matmul', ['q_norm'], 'q', nh * qk, 2, 2 * A.qRank * nh * qk),
    // k counts per head with the shared 64-dim slice expanded, as attention reads it
    N('kv_up', 'matmul', ['kv_norm'], 'k, v', nh * (qk + A.vHead), 2, 2 * A.kvRank * nh * (A.qkNope + A.vHead)),
    N('attn', 'attn', ['q_up', 'kv_up'], 'attn out', nh * A.vHead, 2, 2 * nh * (qk + A.vHead) * 8192 / 2,
      { needsOwnOutput: true, aux: { name: 'lse', bytes: 4 * nh } }),
    N('gate1', 'matmul', ['norm1'], 'gate', PJ, B8, 2 * h * PJ),
    N('gate_mul1', 'vector', ['attn', 'gate1'], 'gated out', PJ, B8, PJ),
  ];
  const ffn = moe ? [
    N('router', 'matmul', ['norm2'], 'router state (logits, k probs)', A.routedExperts + k, 4, 2 * h * A.routedExperts, { bucket: 'moe' }),
    N('lat_down', 'matmul', ['norm2'], 'z (latent)', l, B8, 2 * h * l, { bucket: 'moe' }),
    N('dispatch', 'comm', ['lat_down', 'router'], 'dispatched tokens', k * l, B8, 0, { bucket: 'moe' }),
    N('gate_up', 'matmul', ['dispatch'], 'gate, up (routed)', k * 2 * I, B8, 2 * k * l * 2 * I, { bucket: 'moe' }),
    // the top-k weight multiplies before the (linear) down-proj, so the probs'
    // gradient reads the activation, not the expert outputs (K3 §4's rewrite)
    N('situ', 'vector', ['gate_up', 'router'], 'act out (routed)', k * I, B8, 8 * k * I, { bucket: 'moe' }),
    N('ffn_down', 'matmul', ['situ'], 'expert outputs', k * l, 2, 2 * k * I * l, { bucket: 'moe' }),
    N('combine', 'comm', ['ffn_down'], 'combined latent', l, B8, 0, { bucket: 'moe', bwdNeeds: [] }),
    N('lat_norm', 'vector', ['combine'], 'norm(latent)', l, B8, 8 * l, { bucket: 'moe', aux: rstd() }),
    N('lat_up', 'matmul', ['lat_norm'], 'moe out', h, 2, 2 * l * h, { bucket: 'moe' }),
    N('shared_gu', 'matmul', ['norm2'], 'gate, up (shared)', 2 * I * A.sharedExperts, B8, 2 * h * 2 * I * A.sharedExperts, { bucket: 'moe' }),
    N('shared_situ', 'vector', ['shared_gu'], 'act out (shared)', I * A.sharedExperts, B8, 8 * I * A.sharedExperts, { bucket: 'moe' }),
    N('shared_down', 'matmul', ['shared_situ'], 'shared out', h, 2, 2 * I * A.sharedExperts * h, { bucket: 'moe' }),
    N('moe_add', 'add', ['lat_up', 'shared_down'], 'ffn out', h, 2, 0, { bucket: 'moe' }),
  ] : [
    N('ffn_gate_up_d', 'matmul', ['norm2'], 'gate, up', 2 * A.denseInter, B8, 2 * h * 2 * A.denseInter, { bucket: 'moe' }),
    N('situ_d', 'vector', ['ffn_gate_up_d'], 'act out', A.denseInter, B8, 8 * A.denseInter, { bucket: 'moe' }),
    N('ffn_down_d', 'matmul', ['situ_d'], 'ffn out', h, 2, 2 * A.denseInter * h, { bucket: 'moe' }),
  ];
  return [
    // the partial sum entering the layer: the checkpoint anchor (AttnRes keeps
    // the residual stream as a within-block partial sum, reset every 12 layers)
    N('x0', 'boundary', [], 'x0 (partial sum in)', h, 2, 0, { bucket: 'residual', always: true }),
    N('attnres1', 'vector', ['x0'], 'attn mix out', h, 2, 0, { bucket: 'residual' }),
    // MoE: the router reads BF16, the latent/shared GEMMs an FP8 copy
    N('norm1', 'vector', ['attnres1'], 'norm1 out', h, B8, 8 * h, { aux: rstd() }),
    ...attn,
    N('o_proj', 'matmul', ['gate_mul1'], 'attn proj out', h, 2, 2 * PJ * h),
    N('res1', 'add', ['o_proj', 'x0'], 'x1 (partial sum)', h, 2, 0, { bucket: 'residual' }),
    N('attnres2', 'vector', ['res1'], 'ffn mix out', h, 2, 0, { bucket: 'residual' }),
    N('norm2', 'vector', ['attnres2'], 'norm2 out', h, moe ? 2 + B8 : B8, 8 * h, { bucket: 'moe', aux: rstd() }),
    ...ffn,
    N('res2', 'add', [moe ? 'moe_add' : 'ffn_down_d', 'res1'], 'x2 → next layer', h, 2, 0, { bucket: 'residual', nomark: true }),
  ];
}
// every markable op across the four block kinds
// the briefed recipe's COMPUTE dtype per op (null = not a GEMM): every
// linear MXFP8 except the routers (BF16 in, FP32 logits out); both attention
// cores BF16. One rule for the sheet's replay split and the diagram's tints.
export const k3ComputeDt = (n) => n.opKind === 'attn' || n.id === 'router' ? 'bf16' : n.opKind === 'matmul' ? 'mxfp8' : null;
export const MARKABLE_K3 = [...new Set(['kda', 'mla'].flatMap((a) => ['moe', 'dense'].flatMap((f) =>
  kimiGraph(a, f).filter((n) => !n.always && !n.nomark).map((n) => n.id))))];
const saveExcept = (redo) => Object.fromEntries(MARKABLE_K3.filter((i) => !redo.includes(i)).map((i) => [i, true]));
export const K3_RECOMPUTE = {
  none: saveExcept([]),
  // K3 §5: element-wise operators recompute, AttnRes runs wholly under
  // checkpointing, the group GEMM's input comes back by re-dispatching; GEMM
  // outputs stay (block-wise FP8) — the sheet's policy A
  k3: saveExcept(['attnres1', 'norm1', 'conv', 'gate_mul1', 'q_norm', 'kv_norm', 'attnres2', 'norm2',
    'situ', 'lat_norm', 'shared_situ', 'situ_d', 'dispatch']),
  full: {},
};
export const K3_POLICIES = [['none', 'no recompute'], ['k3', 'K3'], ['full', 'recompute all']];
// whole-model stash per token: the three block kinds × their counts
export const K3_KINDS = [['kda', 'dense', A.denseLayers], ['kda', 'moe', A.kdaLayers - A.denseLayers], ['mla', 'moe', A.mlaLayers]];
export function k3Stash(marks) {
  return K3_KINDS.reduce((t, [a, f, n]) => t + n * analyze(kimiGraph(a, f), marks).savedBytes, 0);
}

// ---- the pipeline split ------------------------------------------------------
// K3's report publishes no PP degree, VP count or layer split, so the split is
// ours: contiguous chunks cut to balance ROOFLINE TIME per rank, priced as the
// sheet prices compute — MXFP8 GEMMs (6 × the layer's active params bar the
// router) at the FP8 rate, the BF16 router and attention kernels (MLA's
// softmax attention at sequence length S, KDA's chunked scan) at the BF16 rate
// — with the LM head (one more layer's worth of GEMM) on the last chunk and the
// embedding lookup free. Integer layers leave a residual imbalance λ = the
// slowest rank / the mean: the sheet multiplies each schedule's compute (and
// its bubble) by it. Times are in FP8-FLOP units: fp8 + ratio · bf16, ratio =
// the FP8 rate over the BF16 rate (2 at the sheet's defaults).
export function k3LayerCost(S = 8192) {
  const nh = A.heads, dh = A.headDim, dR = A.qkRope;
  const sdpa = nh * S * (4 * (dh + dR) + 3 * dh);            // per token, fwd + bwd: 4 QKᵀ-shaped + 3 PV-shaped passes, causal
  const kda = 3 * nh * (6 * dh * dh + 10 * 64 * dh);          // fwd + 2× bwd
  const layers = Array.from({ length: A.layers }, (_, l) => {
    const mla = (l + 1) % 4 === 0 || l === A.layers - 1, dense = l < A.denseLayers;
    const att = mla ? K.mlaAttn : K.kdaAttn, ffn = dense ? K.denseFfn : K.activeMoeFfn - K.routerWeight;
    return { f8: 6 * (att + ffn + K.misc + (dense ? 0 : K.routerBias)), bf: (dense ? 0 : 6 * K.routerWeight) + (mla ? sdpa : kda) };
  });
  return { layers, head: { f8: 6 * K.embed + 6 * K.finalNorms, bf: 0 } };
}
export function k3Split(P, sched = 'v', V = 2, S = 8192, ratio = 2) {
  const nc = P * (sched === 'chain' ? 1 : sched === 'v' ? 2 : V);
  const rank = (c) => sched === 'v' && c >= P ? nc - 1 - c : c % P;
  const { layers, head } = k3LayerCost(S);
  const t = layers.map((x) => x.f8 + ratio * x.bf), th = head.f8 + ratio * head.bf, L = t.length;
  const cum = [0]; for (let l = 0; l < L; l++) cum.push(cum[l] + t[l]);
  const total = cum[L] + th;
  const loads = (cuts) => {
    const r = new Array(P).fill(0);
    for (let c = 0; c < nc; c++) r[rank(c)] += cum[cuts[c + 1]] - cum[cuts[c]];
    r[rank(nc - 1)] += th;
    return r;
  };
  // start at the time quantiles, then nudge single cuts while the slowest rank improves
  const cuts = Array.from({ length: nc + 1 }, (_, c) => {
    if (c === 0 || c === nc) return c ? L : 0;
    let l = 0; const want = total * c / nc;
    while (l < L && cum[l + 1] < want) l++;
    return want - cum[l] < cum[l + 1] - want ? l : l + 1;
  });
  // accept a one-layer move when the rank loads, sorted worst first, improve lexicographically
  const key = (cs) => loads(cs).sort((a, b) => b - a);
  const better = (a, b) => { for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return a[i] < b[i]; return false; };
  for (let improved = true; improved;) {
    improved = false;
    for (let c = 1; c < nc; c++) for (const dl of [-1, 1]) {
      const nx = cuts[c] + dl;
      if (nx <= cuts[c - 1] || nx >= cuts[c + 1]) continue;
      const k0 = key(cuts), old = cuts[c];
      cuts[c] = nx;
      if (better(key(cuts), k0)) improved = true; else cuts[c] = old;
    }
  }
  const load = loads(cuts), slow = load.indexOf(Math.max(...load));
  // the slowest rank's FLOPs per token by rate class (integers: the sheet rebuilds λ from them exactly)
  let f8max = 0, bfmax = 0;
  for (let c = 0; c < nc; c++) if (rank(c) === slow) for (let l = cuts[c]; l < cuts[c + 1]; l++) { f8max += layers[l].f8; bfmax += layers[l].bf; }
  if (rank(nc - 1) === slow) { f8max += head.f8; bfmax += head.bf; }
  return { cuts, ranks: Array.from({ length: nc }, (_, c) => rank(c)), load, slow, f8max, bfmax, lam: Math.max(...load) * P / total };
}
// the sheet's pipeline inputs at one layout: λ per schedule family (1F1B,
// interleaved with V chunks, the V shapes) and r_res, the widest of the three
export function k3PipeInputs(P, V = 2, S = 8192, ratio = 2) {
  const fam = { chain: k3Split(P, 'chain', V, S, ratio), ilv: k3Split(P, 'ilv', V, S, ratio), v: k3Split(P, 'v', V, S, ratio) };
  return {
    fam, lam1: fam.chain.lam, lamI: fam.ilv.lam, lamV: fam.v.lam,
    rres: Math.max(...Object.entries(fam).map(([s, sp]) => attnresPipe(P, s, V, sp.cuts).rres)),
  };
}

// ---- AttnRes across a pipeline ---------------------------------------------
// Block AttnRes (arXiv 2603.15031 §4.1, K3 §5): every mix reads the block
// store b₀ … bₙ₋₁ (b₀ = the embedding, then one sum per finished 12-layer
// block) plus the running partial sum. A stage transition carries the blocks
// the receiving rank doesn't hold yet, plus the partial sum; a rank keeps
// every block it has seen for the microbatch (cross-stage caching), so only a
// SECOND visit — interleaved or V-shaped schedules — sends less. Backward
// returns the same widths as gradients. Chunks are k3Split's (balanced
// roofline time) unless cuts are given; sched = 'chain' (1F1B), 'ilv' (V
// chunks round-robin), 'v' (two chunks per rank, out and back).
export function attnresPipe(P, sched = 'v', V = 2, cuts = k3Split(P, sched, V).cuts) {
  const S = A.attnResBlock, L = A.layers;
  const nc = cuts.length - 1;
  const rank = (c) => sched === 'v' && c >= P ? nc - 1 - c : c % P;
  const reps = (l) => Math.ceil(l / S);        // blocks in the store after l layers (b₀ joins at layer 1)
  const held = new Array(P).fill(0), send = new Array(P).fill(0), tr = [];
  for (let c = 0; c < nc - 1; c++) {
    const l = cuts[c + 1], from = rank(c), to = rank(c + 1);
    held[from] = reps(l);
    const w = from === to ? 0 : reps(l) - held[to] + 1;   // new blocks + the partial sum
    tr.push({ l, from, to, w, naive: from === to ? 0 : reps(l) + 1 });
    send[from] += w; send[to] += w;                          // forward by the sender, gradients back by the receiver
  }
  held[rank(nc - 1)] = reps(L);
  return { cuts, ranks: Array.from({ length: nc }, (_, c) => rank(c)), tr, send, held, rres: Math.max(0, ...send) };
}

// ---- the roofline sheet's view of the stash ---------------------------------
// The sheet prices policy A's stash as Σ κ · rate per bucket, where κ is a
// 0/1 kept? cell per tensor (set live from the recompute figure: 1 = the
// figure's marks stash it) and rate is in the sheet's own symbols ({key} = a
// sheet cell). Formulas never change shape; flipping marks flips κ values.
// Buckets follow the sheet's rows: σ_res (every layer), σᴬ_kda / σᴬ_mla (the
// attention kinds), σᴬ_moe / σᴬ_dns (the FFN kinds); a κ named in two
// buckets (norm1, the gate, the out-proj, norm2) is one cell — its verdict is
// the same in every block kind (k3Linked checks). x0, the checkpoint anchor,
// is always stashed.
export const K3_SHEET = [
  { key: 'sres', graph: ['kda', 'moe'], fixed: '{d} · 2', terms: [
    ['attnres1', 'mix1', '{d} · 2'], ['res1', 'x1', '{d} · 2'], ['attnres2', 'mix2', '{d} · 2']] },
  { key: 'sAk', graph: ['kda', 'moe'], terms: [
    ['norm1', 'n1', '{d} · {b8}'], ['norm1:aux', 'rstd1', '4'],
    ['qkv', 'qkv', '3 · {P} · {b8}'], ['conv', 'conv', '3 · {P} · 2'],
    ['decay', 'α', '{P} · {b8}'], ['decay:aux', 'fa', '2 · {ra}'], ['beta', 'β', '{nh} · {b8}'],
    ['gate1', 'g', '{P} · {b8}'], ['kda_scan', 'scan', '{P} · 2'],
    ['gate_mul1', 'go', '{P} · {b8}'], ['gate_mul1:aux', 'rstdg', '4 · {nh}'], ['o_proj', 'out', '{d} · 2']] },
  { key: 'sAm', graph: ['mla', 'moe'], terms: [
    ['norm1', 'n1', '{d} · {b8}'], ['norm1:aux', 'rstd1', '4'],
    ['q_down', 'cq', '{rq} · {b8}'], ['kv_down', 'ckv', '({rkv} + {dR}) · {b8}'],
    ['q_norm', 'ncq', '{rq} · {b8}'], ['q_norm:aux', 'rstdq', '4'], ['kv_norm', 'nckv', '{rkv} · {b8}'], ['kv_norm:aux', 'rstdkv', '4'],
    ['q_up', 'q', '{nh} · ({dh} + {dR}) · 2'], ['kv_up', 'kv', '{nh} · (2 · {dh} + {dR}) · 2'],
    ['attn', 'attn', '{nh} · {dh} · 2'], ['attn:aux', 'lse', '4 · {nh}'],
    ['gate1', 'g', '{P} · {b8}'], ['gate_mul1', 'go', '{P} · {b8}'], ['o_proj', 'out', '{d} · 2']] },
  { key: 'sAe', graph: ['kda', 'moe'], terms: [
    ['norm2', 'n2', '{d} · (2 + {b8})'], ['norm2:aux', 'rstd2', '4'],
    ['router', 'rt', '({E} + {k}) · 4'], ['lat_down', 'z', '{l} · {b8}'], ['dispatch', 'disp', '{k} · {l} · {b8}'],
    ['gate_up', 'fc1', '{k} · 2 · {I} · {b8}'], ['situ', 'act', '{k} · {I} · {b8}'], ['ffn_down', 'fc2', '{k} · {l} · 2'],
    ['combine', 'comb', '{l} · {b8}'], ['lat_norm', 'nz', '{l} · {b8}'], ['lat_norm:aux', 'rstdz', '4'], ['lat_up', 'up', '{d} · 2'],
    ['shared_gu', 'fc1s', '2 · {I} · {nsh} · {b8}'], ['shared_situ', 'acts', '{I} · {nsh} · {b8}'], ['shared_down', 'fc2s', '{d} · 2'],
    ['moe_add', 'add', '{d} · 2']] },
  { key: 'sAd', graph: ['kda', 'dense'], terms: [
    ['norm2', 'n2', '{d} · {b8}'], ['norm2:aux', 'rstd2', '4'],
    ['ffn_gate_up_d', 'fc1d', '2 · {Id} · {b8}'], ['situ_d', 'actd', '{Id} · {b8}'], ['ffn_down_d', 'fc2d', '{d} · 2']] },
];
export const kappaName = (short) => `κ<sub>${short}</sub>`;
// the pipeline split's sheet inputs: r_res and the slowest rank's FLOPs per schedule family
export const PIPE_NAMES = { rres: 'r<sub>res</sub>',
  f81: 'c<sup>max</sup><sub>8,1F1B</sub>', bf1: 'c<sup>max</sup><sub>bf,1F1B</sub>',
  f8I: 'c<sup>max</sup><sub>8,ILV</sub>', bfI: 'c<sup>max</sup><sub>bf,ILV</sub>',
  f8V: 'c<sup>max</sup><sub>8,V</sub>', bfV: 'c<sup>max</sup><sub>bf,V</sub>' };
const pipeMemo = new Map();
export function k3PipeLinked(P, V = 2, S = 8192, ratio = 2) {
  const key = `${P},${V},${S},${ratio}`;
  if (pipeMemo.has(key)) return pipeMemo.get(key);
  const x = k3PipeInputs(P, V, S, ratio), N = PIPE_NAMES;
  const out = { [N.rres]: x.rres, [N.f81]: x.fam.chain.f8max, [N.bf1]: x.fam.chain.bfmax,
    [N.f8I]: x.fam.ilv.f8max, [N.bfI]: x.fam.ilv.bfmax, [N.f8V]: x.fam.v.f8max, [N.bfV]: x.fam.v.bfmax };
  pipeMemo.set(key, out);
  return out;
}
export const LINK_NAMES = { rhod: 'δ<sub>disp</sub>', rhoc: 'δ<sub>comb</sub>', r8A: 'c<sup>A</sup><sub>rp8</sub>', rbfA: 'c<sup>A</sup><sub>rpbf</sub>' };
// marks → the sheet's figure-driven inputs: every κ, whether the dispatch /
// combine replay (EP traffic), and the GEMM / kernel FLOPs the marks replay per
// token over the whole model (vector work stays unpriced, the site's rule);
// split by rate class: MXFP8 GEMMs vs the BF16 router and attention kernels.
export function k3Linked(marks) {
  const out = {}, clash = [];
  const anas = {};
  const ana = (a, f) => (anas[a + f] ??= analyze(kimiGraph(a, f), marks));
  for (const bk of K3_SHEET) {
    const r = ana(...bk.graph);
    for (const [id, short] of bk.terms) {
      const [nid, aux] = id.split(':');
      const kept = aux ? (r.byId[nid].aux && !r.replayed.has(nid) ? 1 : 0) : (r.neededSaved.has(nid) ? 1 : 0);
      const name = kappaName(short);
      if (name in out && out[name] !== kept) clash.push(short);
      out[name] = kept;
    }
  }
  const moe = ana('kda', 'moe');
  out[LINK_NAMES.rhod] = moe.replayed.has('dispatch') ? 1 : 0;
  out[LINK_NAMES.rhoc] = moe.replayed.has('combine') ? 1 : 0;
  let r8 = 0, rbf = 0;
  for (const [a, f, n] of K3_KINDS) {
    const r = ana(a, f);
    for (const id of r.replayed) {
      const nd = r.byId[id];
      const dt = k3ComputeDt(nd);
      if (dt === 'bf16') rbf += n * nd.flopsTok;
      else if (dt === 'mxfp8') r8 += n * nd.flopsTok;
    }
  }
  out[LINK_NAMES.r8A] = r8; out[LINK_NAMES.rbfA] = rbf;
  return { values: out, clash };
}

// ---- memory per pipeline rank ---------------------------------------------
// Parameters can't move between ranks; activations can (K3 §5.2.2: remote
// offload "achieving balanced activation memory across PP ranks"), so the fit
// is decided by the rank with the most parameter bytes plus its AttnRes block
// store (never offloaded). Per rank: its layers' routed experts (the sheet
// divides by EP) and non-expert params, the embedding with the first chunk,
// the head and final norms with the last (the V shape puts both on rank 0),
// and store = blocks held × microbatches in flight on that rank (1F1B: PP − r;
// interleaved: (2(PP − r − 1) + (V − 1)PP + 1)/V; V shapes: every rank alike,
// so the count is per microbatch and the sheet multiplies by n). w = the byte
// weights {e: per expert param, n: per non-expert param, s: per store
// block·microbatch}; the worst rank maximizes e·Xe + n·Xn + s·H.
export function k3RankMemory(P, sched, V = 2, w = { e: 1, n: 1, s: 0 }, S = 8192, ratio = 2) {
  const sp = k3Split(P, sched, V, S, ratio), pipe = attnresPipe(P, sched, V, sp.cuts);
  const exp = A.routedExperts * K.expert;
  const ne = (l) => (l + 1) % 4 === 0 || l === A.layers - 1 ? K.mlaAttn : K.kdaAttn;
  const nf = (l) => l < A.denseLayers ? K.denseFfn : K.routerWeight + K.routerBias + K.latents + K.shared;
  const rk = Array.from({ length: P }, () => ({ xe: 0, xn: 0 }));
  sp.ranks.forEach((r, c) => {
    for (let l = sp.cuts[c]; l < sp.cuts[c + 1]; l++) { rk[r].xn += ne(l) + nf(l) + K.misc; if (l >= A.denseLayers) rk[r].xe += exp; }
  });
  rk[sp.ranks[0]].xn += K.embed;
  rk[sp.ranks[sp.ranks.length - 1]].xn += K.embed + K.finalNorms;
  // store: [numerator, denominator] block·microbatches (exact; interleaved counts chunks of 1/V)
  const flight = (r) => sched === 'chain' ? [P - r, 1] : sched === 'ilv' ? [2 * (P - r - 1) + (V - 1) * P + 1, V] : [1, 1];
  const ranks = rk.map((x, r) => { const [fn, fd] = flight(r); return { ...x, h: [pipe.held[r] * fn, fd] }; });
  const cost = (x) => w.e * x.xe + w.n * x.xn + w.s * x.h[0] / x.h[1];
  const worst = ranks.reduce((bi, x, r) => cost(x) > cost(ranks[bi]) ? r : bi, 0);
  return { ranks, worst, ...ranks[worst] };
}
export const MEM_NAMES = {
  xe1: 'X<sup>e</sup><sub>1F1B</sub>', xn1: 'X<sup>n</sup><sub>1F1B</sub>', h1: 'H<sub>1F1B</sub>',
  xeI: 'X<sup>e</sup><sub>ILV</sub>', xnI: 'X<sup>n</sup><sub>ILV</sub>', hI: 'H<sub>ILV</sub>',
  xeV: 'X<sup>e</sup><sub>V</sub>', xnV: 'X<sup>n</sup><sub>V</sub>', hV: 'H<sub>V</sub>' };
const memMemo = new Map();
// nV: microbatches in flight under the V schedules (every rank alike), to weigh
// the store when picking the V family's worst rank (the sheet multiplies by
// each V schedule's own n)
export function k3MemLinked(P, V, w, S = 8192, ratio = 2, nV = 1) {
  const key = JSON.stringify([P, V, w, S, ratio, nV]);
  if (memMemo.has(key)) return memMemo.get(key);
  const out = {};
  for (const [t, s] of [['1', 'chain'], ['I', 'ilv'], ['V', 'v']]) {
    const m = k3RankMemory(P, s, V, s === 'v' ? { ...w, s: w.s * nV } : w, S, ratio);
    out[MEM_NAMES['xe' + t]] = m.xe; out[MEM_NAMES['xn' + t]] = m.xn; out[MEM_NAMES['h' + t]] = m.h;
  }
  memMemo.set(key, out);
  return out;
}
