// Kimi K3 study widgets: the 01-style architecture diagram for Moonshot's
// Kimi K3 (2.78T MoE, hybrid KDA + gated-MLA attention, Stable LatentMoE).
// <kimi-anatomy> = margin plan + expansion cone + tabbed block diagram,
// mirroring <dsv3-anatomy>, but self-contained: K3's block topology (KDA
// scan, latent expert projections, attention residuals) doesn't fit the
// DSv3 renderer, so this module draws its own SVG in the same visual
// grammar (docs/diagram-grammar.md) and imports only stable helpers.
//
// Sources: moonshotai/Kimi-K3 config.json + modeling_kimi_linear.py (tensor
// shapes) and the K3 tech report (topology). Parameter totals reconcile
// exactly with the advertised 2.78T / 104.2B — see the study's footnote.

import { fmtP, tokensCss, applyHighlight } from './viewer.js';
import { C } from './theme.js';
import { attachTip } from './tip.js';
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
const MARKABLE_K3 = [...new Set(['kda', 'mla'].flatMap((a) => ['moe', 'dense'].flatMap((f) =>
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

// ---- AttnRes across a pipeline ---------------------------------------------
// Block AttnRes (arXiv 2603.15031 §4.1, K3 §5): every mix reads the block
// store b₀ … bₙ₋₁ (b₀ = the embedding, then one sum per finished 12-layer
// block) plus the running partial sum. A stage transition carries the blocks
// the receiving rank doesn't hold yet, plus the partial sum; a rank keeps
// every block it has seen for the microbatch (cross-stage caching), so only a
// SECOND visit — interleaved or V-shaped schedules — sends less. Backward
// returns the same widths as gradients. Chunks split the 93 layers evenly
// (rounded); sched = 'chain' (1F1B), 'ilv' (V chunks round-robin), 'v' (two
// chunks per rank, out and back).
export function attnresPipe(P, sched = 'v', V = 2) {
  const S = A.attnResBlock, L = A.layers;
  const nc = P * (sched === 'chain' ? 1 : sched === 'v' ? 2 : V);
  const cuts = Array.from({ length: nc + 1 }, (_, c) => Math.round(c * L / nc));
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

// totals cross the fmtP=B ceiling; the tally speaks T there
const fmtPT = (n) => n >= 1e12 ? (n / 1e12).toFixed(2) + 'T' : fmtP(n);
// stash bytes per token (binary units)
const fmtB = (b) => b < 1024 ? `${Math.round(b)} B` : b < 1024 ** 2 ? `${(b / 1024).toFixed(1)} KiB` : `${(b / 1024 ** 2).toFixed(2)} MiB`;

// URL-hash state, same mechanics as viewer.js (#k3:<id>=<json>)
const readState = (key) => {
  try { const v = new URLSearchParams(location.hash.slice(1)).get(key); return v ? JSON.parse(v) : null; }
  catch { return null; }
};
const writeState = (key, obj) => {
  const p = new URLSearchParams(location.hash.slice(1));
  p.set(key, JSON.stringify(obj));
  history.replaceState(null, '', '#' + p.toString());
};

// ---- <kimi-layer> ----------------------------------------------------------
// One K3 transformer block, two tabbed columns: attention (KDA ×69 | gated
// MLA ×24) and FFN (LatentMoE ×92 | dense ×1). Static tier only: structure,
// dims, parameter parentheticals, params lens; no dtype/recompute machinery.

const KV_CSS = `
kimi-layer { display: block; margin: 14px 0 26px; }
.kv { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-0b0b0b);
  border: 1px solid var(--c-e1e0d9); border-radius: 6px; background: var(--c-fcfcfb); padding: 10px 12px; position: relative; }
.kv-head { display: flex; align-items: center; gap: 8px; padding-bottom: 6px; color: var(--c-52514e); flex-wrap: wrap; }
.kv-head button { font: 11px ui-monospace, monospace; padding: 2px 8px; border: 1px solid var(--c-c3c2b7);
  border-radius: 4px; background: var(--c-ffffff); color: var(--c-0b0b0b); cursor: pointer; margin-left: 8px; min-width: 8ch; box-sizing: content-box; }
.kv svg { display: block; margin: 0 auto; }
.kv-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
${tokensCss('.kv')}
.kv text.tensor { font: 10px system-ui; }
.kv .tidle { fill: var(--c-a8a69e); }
.kv .tdim { fill: var(--c-898781); font-weight: 400; }
.kv .micro { fill: var(--c-f7f6f1); stroke: var(--c-d8d6cb); }
.kv .microlabel { font: italic 10px system-ui; fill: var(--c-52514e); }
.kv svg.hlm > :not(.hl):not(defs) { opacity: 0.3; }
.kv g[data-op].hl .dims { fill: var(--c-52514e); font-weight: 600; }
/* recompute marks (the marks attribute): click an op to flip save ↔ ↻ */
.kv g[data-mk] { cursor: pointer; }
.kv g.rpl > rect { fill: var(--c-e0f3f3); stroke: var(--c-0a98a0); }
.kv .mkg { font: 600 11px system-ui; }
.kv .tsave { fill: var(--c-875600); }
.kv .tredo { fill: var(--c-52514e); font-style: italic; }
.kv-head .pols { display: inline-flex; gap: 4px; margin-left: 18px; align-items: center; }
.kv-head .pols button { margin-left: 0; min-width: 0; }
.kv-head .pols button.on { background: var(--c-eeede7); font-weight: 600; cursor: default; }
.kv-head .pols .cust { font-size: 11px; color: var(--c-898781); margin-left: 4px; }
.kv-led { font-size: 11.5px; color: var(--c-52514e); line-height: 1.55; padding: 6px 4px 0; max-width: 1060px; margin: 0 auto;
  min-height: calc(7 * 1.55em); }   /* worst case (every op replays, MLA + MoE): no reflow on a flip */
.kv-led b { color: var(--c-0b0b0b); }
.kv-led .st { white-space: nowrap; color: var(--c-875600); }
.kv-led .rp { white-space: nowrap; font-style: italic; }
`;

export class KimiLayer extends HTMLElement {
  connectedCallback() {
    const st = readState('k3:' + (this.id || 'layer'));
    // kind-a / kind-f / flat attributes seed the state (test matrix affordance)
    this.kindA = (st?.a ?? this.getAttribute('kind-a')) === 'mla' ? 'mla' : 'kda';
    this.kindF = (st?.f ?? this.getAttribute('kind-f')) === 'dense' ? 'dense' : 'moe';
    this.flatDims = st ? !!st.flat : this.hasAttribute('flat');
    this.detail = this.hasAttribute('detail');
    this.activeView = false;
    this.marksMode = this.hasAttribute('marks');
    const pol = this.getAttribute('recompute') ?? 'k3';
    this.marks = st?.m ? Object.fromEntries(st.m.map((i) => [i, true])) : { ...(K3_RECOMPUTE[st?.p ?? pol] ?? K3_RECOMPUTE.k3) };
    const style = document.createElement('style'); style.textContent = KV_CSS;
    this._card = document.createElement('div'); this._card.className = 'kv';
    this.append(style, this._card);
    this.render();
    addEventListener('dsv3-theme', () => this.render());
  }
  changed() {
    const st = { a: this.kindA, f: this.kindF, flat: this.flatDims ? 1 : 0 };
    if (this.marksMode) { const p = this.policy(); if (p) st.p = p; else st.m = Object.keys(this.marks).filter((i) => this.marks[i]); }
    writeState('k3:' + (this.id || 'layer'), st);
    this.dispatchEvent(new CustomEvent('recipe'));
  }
  // the preset these marks equal, or null (custom)
  policy() {
    const on = (m) => MARKABLE_K3.filter((i) => m[i] === true).join();
    return K3_POLICIES.map(([k]) => k).find((k) => on(K3_RECOMPUTE[k]) === on(this.marks)) ?? null;
  }
  highlightOps(ids) { this._hl = ids ? new Set(ids) : null; this.applyHl(); }
  applyHl() {
    const svg = this._card.querySelector('svg');
    if (svg) applyHighlight(this._card, this._hl);
  }
  render() {
    const P = this.getAttribute('lens') === 'params';   // params lens: counts, not dims
    const F = this.flatDims;
    const S = [];
    const W = 1086, H = 730;

    // ---- geometry ----------------------------------------------------------
    // spines run ~20px inside their boxes' left edges (wires thread box-to-box)
    const RAIL1 = 36, C1 = 80, SX1 = 100, W1 = 280;     // attention column
    const CS = 390, WS = 150;                           // its side boxes (decay/β/gate)
    const BRX = 378;                                    // side-branch vertical (the 360..390 gutter)
    const RR1 = 552, RR2 = 560;                         // right rails into the scan/attention box
    const UPX = 572, RAIL2 = 584;                       // x1 → FFN top; x1 → final add
    const C2 = 600, SX2 = 620, W2 = 286;                // FFN column
    const GATEX = 900;                                  // top-k weights rail
    const SHX = 912, CSH = 924, WSH = 146, SXSH = 944;  // shared-experts sub-column
    // shared row anchors (kind flips preserve them)
    const yTop = 16, yRes0 = 44, yN1 = 78, yChip1 = 112, yTab = 124, yEnc = 150, yBr = 158;
    const yA = 170, yB = 224, yMicro = 260, yC = 248, ySide3 = 278,
      yAtt = 312, yGate = 368, yO = 402, yEncB = 460, yAdd1 = 478, yX1 = 512;
    // FFN rows
    const yRout = 170, yRoutM = 208, yLatD = 236, yDisp = 290, yGrp = 336,
      yGU = 358, ySw = 410, yFD = 442, yGrpB = 486, yComb = 508, yLatN = 544,
      yLatU = 570, yMAdd = 624, yEnc2B = 656, yAdd2 = 672;

    // ---- drawing helpers (the grammar's tokens) ----------------------------
    const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
    // marks tier: the graph's verdicts; g() opens an op group with its mark
    // attributes, glyph() puts ↓ (saved) or ↻ (replays) at the op's right end
    const M = this.marksMode && !P;
    const graph = M ? kimiGraph(this.kindA, this.kindF) : [];
    const byId = Object.fromEntries(graph.map((n) => [n.id, n]));
    const ana = M ? analyze(graph, this.marks) : null;
    this._ana = ana; this._graph = graph;
    const MK = { router_bias: 'router' };   // boxes whose mark is another op's
    const mkOf = (id) => { const m = MK[id] ?? id; return byId[m] && !byId[m].always && !byId[m].nomark ? m : null; };
    const g = (id) => {
      const m = M && mkOf(id);
      return `<g data-op="${id}"${m ? ` data-mk="${m}"${this.marks[m] === true ? '' : ' class="rpl"'}` : ''}>`;
    };
    const glyph = (id, x, y) => {
      const m = M && mkOf(id);
      if (!m) return '';
      const sv = this.marks[m] === true;
      return `<text class="mkg" text-anchor="end" x="${x}" y="${y}" fill="${C(sv ? '#b05f00' : '#0a98a0')}">${sv ? '↓' : '↻'}</text>`;
    };
    const pk = (n, mult) => P || n ? ` <tspan class="dims">(${fmtP(n)}${mult ? ` ×${mult}` : ''})</tspan>` : '';
    const box = (id, x, y, w, name, dims, h = 32) =>
      S.push(`${g(id)}<rect class="box" x="${x}" y="${y}" width="${w}" height="${h}" rx="4"/>` +
        `<text class="name" x="${x + 8}" y="${y + 14}">${esc(name)}</text>` +
        (dims ? `<text class="dims" x="${x + 8}" y="${y + 26}">${dims}</text>` : '') + glyph(id, x + w - 7, y + 15) + `</g>`);
    const pill = (id, x, y, w, label, h = 20) =>
      S.push(`${g(id)}<rect class="op" x="${x}" y="${y}" width="${w}" height="${h}" rx="6"/>` +
        `<text class="oplabel" x="${x + 9}" y="${y + 14}">${label}</text>${glyph(id, x + w - 7, y + h / 2 + 4)}</g>`);
    const micro = (x, y, w, label, h = 16) =>
      S.push(`<rect class="micro" x="${x}" y="${y}" width="${w}" height="${h}" rx="6"/>` +
        `<text class="microlabel" x="${x + 9}" y="${y + 12}">${label}</text>`);
    const comm = (id, x, y, w, label, h = 22) =>
      S.push(`${g(id)}<rect class="comm" x="${x}" y="${y}" width="${w}" height="${h}" rx="6"/>` +
        `<text class="oplabel" x="${x + 9}" y="${y + 15}" fill="${C('#4636a3')}">${label}</text>${glyph(id, x + w - 7, y + 15)}</g>`);
    const res = (id, sx, y, w, label) => {   // ⊕ on the spine + dashed label box to its right
      S.push(`${g(id)}<circle cx="${sx}" cy="${y + 10}" r="7" fill="${C('#fcfcfb')}" stroke="${C('#c3c2b7')}"/>` +
        `<text class="plus" x="${sx - 3.5}" y="${y + 14}">+</text>` +
        `<rect class="res" x="${sx + 14}" y="${y}" width="${w}" height="20" rx="6"/>` +
        `<text class="oplabel" x="${sx + 23}" y="${y + 14}">${label}</text>${glyph(id, sx + 14 + w - 7, y + 14)}</g>`);
    };
    const wv = (x, y1, y2, arrow = true) =>
      S.push(`<line class="wire" x1="${x}" y1="${y1}" x2="${x}" y2="${y2}"${arrow ? ' marker-end="url(#karr)"' : ''}/>`);
    const poly = (pts, arrow = true) =>
      S.push(`<polyline class="wire" points="${pts.map(p => p.join(',')).join(' ')}"${arrow ? ' marker-end="url(#karr)"' : ''} fill="none"/>`);
    const dot = (x, y) => S.push(`<circle cx="${x}" cy="${y}" r="2.5" fill="${C('#898781')}"/>`);
    // marks tier: a chip names its producer (nid) and says what backward
    // gets — ↓ stashed (bytes per token) · ↻ recomputed · plain = not needed
    const chip = (x, y, name, dims, nid) => {
      if (P) return;   // the params lens drops tensor chips (01's convention)
      const n = M && byId[nid];
      if (n && ana.neededSaved.has(nid)) {
        S.push(`<text class="tensor tsave" x="${x}" y="${y}">↓ ${esc(name)} <tspan class="tdim">· ${fmtB(ana.savedById[nid])} ${n.dt}</tspan></text>`);
        return;
      }
      if (n && ana.replayed.has(nid) && ana.neededBy.has(nid)) {
        S.push(`<text class="tensor tredo" x="${x}" y="${y}">↻ ${esc(name)} — recomputed</text>`);
        return;
      }
      S.push(`<text class="tensor tidle" x="${x}" y="${y}">· ${esc(name)}${dims ? ` <tspan class="tdim">· ${dims}</tspan>` : ''}</text>`);
    };
    const grp = (x, y, w, h, label, lx = 10) =>
      S.push(`<rect class="grp" x="${x}" y="${y}" width="${w}" height="${h}" rx="6"/>` +
        (label ? `<text class="grplabel" x="${x + lx}" y="${y + 13}">${label}</text>` : ''));
    // tab flaps: the active tab fuses into the enclosure below it
    const tab = (x, w, key, cur, label, sub, onto) => {
      const on = cur === key, r = 6, y0 = yTab, y1 = yTab + 26;
      return `<g data-tab="${onto}:${key}" style="cursor:${on ? 'default' : 'pointer'}">` +
        (on
          ? `<path d="M ${x} ${y1} L ${x} ${y0 + r} Q ${x} ${y0} ${x + r} ${y0} L ${x + w - r} ${y0} ` +
            `Q ${x + w} ${y0} ${x + w} ${y0 + r} L ${x + w} ${y1} Z" fill="${C('#fcfcfb')}" stroke="none"/>` +
            `<path d="M ${x} ${y1} L ${x} ${y0 + r} Q ${x} ${y0} ${x + r} ${y0} L ${x + w - r} ${y0} ` +
            `Q ${x + w} ${y0} ${x + w} ${y0 + r} L ${x + w} ${y1}" fill="none" stroke="${C('#c3c2b7')}"/>`
          : `<path d="M ${x} ${y1} L ${x} ${y0 + r} Q ${x} ${y0} ${x + r} ${y0} L ${x + w - r} ${y0} ` +
            `Q ${x + w} ${y0} ${x + w} ${y0 + r} L ${x + w} ${y1} Z" fill="${C('#eeede7')}" stroke="${C('#d8d6cb')}"/>`) +
        `<text x="${x + 10}" y="${yTab + 17}" style="font:600 11px system-ui" fill="${C(on ? '#0b0b0b' : '#898781')}">${label}` +
        `<tspan style="font:10px system-ui" fill="${C(on ? '#898781' : '#a8a69e')}"> ${sub}</tspan></text></g>`;
    };
    // kind enclosure with a gap under the active tab (the tab's own edge shows)
    const enc = (x, y, w, h, tabX, tabW) =>
      S.push(`<path class="grp" d="M ${tabX} ${y} L ${x + 6} ${y} Q ${x} ${y} ${x} ${y + 6} ` +
        `L ${x} ${y + h - 6} Q ${x} ${y + h} ${x + 6} ${y + h} L ${x + w - 6} ${y + h} ` +
        `Q ${x + w} ${y + h} ${x + w} ${y + h - 6} L ${x + w} ${y + 6} Q ${x + w} ${y} ${x + w - 6} ${y} ` +
        `L ${tabX + tabW} ${y}" fill="none"/>`);

    S.push(`<defs><marker id="karr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" ` +
      `orient="auto-start-reverse"><path d="M 0 0 L 8 4 L 0 8 z" fill="${C('#898781')}"/></marker></defs>`);

    // ---- head: residual stream in, fork to the rail ------------------------
    S.push(`<text class="oplabel" x="${SX1 + 14}" y="${yTop - 4}">x — this block’s partial sum (${A.hidden})</text>`);
    wv(SX1, yTop, yRes0);
    dot(SX1, yTop + 6);
    // the rail carries x to the residual add; the module INPUT is the attn-res
    // mix (K3's twist: module input ≠ residual stream). Under Block AttnRes
    // the rail is the partial sum of the current 12-layer block, and the mix's
    // other input is the block store b₀ … bₙ₋₁ (symbolic: n grows with depth)
    poly([[SX1, yTop + 6], [RAIL1, yTop + 6], [RAIL1, yAdd1 + 10], [SX1 - 9, yAdd1 + 10]], true);
    pill('attnres1', C1, yRes0, W1, `attn-res mix — α over block reps${pk(K.attnRes / 2)}`);
    // the block store: b₀ = the embedding, one sum per finished block
    const store = (x0, wid, into) => {
      S.push(`<rect class="res" x="${x0}" y="${yRes0}" width="${wid}" height="20" rx="6"/>` +
        `<text class="oplabel" x="${x0 + 9}" y="${yRes0 + 14}">block store b₀ … bₙ₋₁ <tspan class="dims">· n×${A.hidden}</tspan></text>`);
      S.push(`<line class="wire" x1="${x0}" y1="${yRes0 + 10}" x2="${into + 3}" y2="${yRes0 + 10}" marker-end="url(#karr)"/>`);
    };
    store(C1 + W1 + 22, 186, C1 + W1);
    // the write: on a block's first layer the incoming partial sum (the
    // finished block, or the embedding at layer 1) joins the store as bₙ and
    // the sum restarts from this layer's own output
    const wx = C1 + W1 + 22 + 93;
    dot(SX1, yTop + 16);
    poly([[SX1, yTop + 16], [wx, yTop + 16], [wx, yRes0]], true);
    S.push(`<text class="tensor tidle" text-anchor="end" x="${wx - 6}" y="${yTop + 12}">a block’s first layer: x → bₙ, the sum restarts</text>`);
    // n is an annotation on the store, not an op: plain text, no box
    if (this.detail) S.push(`<text class="tensor tidle" x="${C1 + W1 + 31}" y="${yRes0 + 34}">n = 1 + ⌊(layer − 1) / ${A.attnResBlock}⌋ = 1 … ${Math.ceil(A.layers / A.attnResBlock)}</text>`);
    wv(SX1, yRes0 + 20, yN1);
    pill('norm1', C1, yN1, W1, `RMSNorm${pk(A.hidden)}`);
    wv(SX1, yN1 + 20, yA);
    chip(SX1 + 14, yChip1, 'norm1 out', A.hidden, 'norm1');
    // the side branch: one horizontal under the tab row, drops feed the
    // side boxes (decay/β/gate — MLA: kv + gate) via the column gutter
    dot(SX1, yBr);
    poly([[SX1, yBr], [BRX, yBr]], false);

    // ---- attention column tabs + enclosure ---------------------------------
    const t1x = C1 + 30, t1w = 128, t2x = t1x + t1w + 8, t2w = 168;
    S.push(tab(t1x, t1w, 'kda', this.kindA, 'KDA', `×${A.kdaLayers} · ${fmtP(K.kdaAttn)}`, 'a'));
    S.push(tab(t2x, t2w, 'mla', this.kindA, 'gated MLA', `×${A.mlaLayers} · ${fmtP(K.mlaAttn)}`, 'a'));
    const at = this.kindA === 'kda' ? { x: t1x, w: t1w } : { x: t2x, w: t2w };
    enc(C1 - 10, yEnc, CS + WS + 26 - (C1 - 10), yEncB - yEnc, at.x, at.w);

    // the out-gate proj is common to both kinds (same params, same anchor)
    box('gate1', CS, ySide3, WS, 'out-gate proj',
      P ? fmtP(K.gate + (this.kindA === 'kda' ? K.gateNorm : 0))
        : F ? `${A.hidden} → ${PJ}${pk(K.gate)}` : `${A.hidden} → ${A.heads}×${A.headDim}${pk(K.gate)}`);
    poly([[BRX, yBr], [BRX, ySide3 + 16], [CS - 3, ySide3 + 16]], true);
    poly([[CS + 75, ySide3 + 32], [CS + 75, yGate + 10], [C1 + W1 + 3, yGate + 10]], true);

    if (this.kindA === 'kda') {
      // main spine: fused qkv → short convs → the delta-rule scan
      box('qkv', C1, yA, W1, 'qkv proj (fused)',
        P ? fmtP(K.qkv) : F ? `${A.hidden} → ${3 * PJ}${pk(K.qkv)}` : `${A.hidden} → 3×${A.heads}×${A.headDim}${pk(K.qkv)}`);
      wv(SX1, yA + 32, yB);
      if (M) chip(SX1 + 14, yB - 6, 'q, k, v (pre-conv)', '', 'qkv');
      pill('conv', C1, yB, W1, `short conv + SiLU — q, k, v · k=${A.convK}${pk(K.conv)}`);
      if (this.detail) {   // the L2Norm micro sits ON the spine: segment around it
        wv(SX1, yB + 20, yMicro);
        micro(C1, yMicro, 150, 'L2Norm — q, k');
        wv(SX1, yMicro + 16, yAtt);
      } else wv(SX1, yB + 20, yAtt);
      // side boxes: decay + β off the branch verticals, outputs rail right
      dot(BRX, yA + 16); dot(BRX, yB + 16);
      poly([[BRX, yA + 16], [CS - 3, yA + 16]], true);
      poly([[BRX, yB + 16], [CS - 3, yB + 16]], true);
      box('decay', CS, yA, WS, 'decay proj α',
        P ? fmtP(K.decay) : F ? `${A.hidden} → ${A.decayRank} → ${PJ}${pk(K.decay)}` : `${A.hidden} → ${A.decayRank} → ${A.heads}×${A.headDim}${pk(K.decay)}`);
      box('beta', CS, yB - 4, WS, 'β proj · sigmoid',
        P ? fmtP(K.beta) : `${A.hidden} → ${A.heads}${pk(K.beta)}`);
      poly([[CS + WS, yA + 16], [RR1, yA + 16], [RR1, yAtt + 8], [C1 + W1 + 3, yAtt + 8]], true);
      poly([[CS + WS, yB + 12], [RR2, yB + 12], [RR2, yAtt + 20], [C1 + W1 + 3, yAtt + 20]], true);
      chip(CS + 2, yA + 44, 'α', `${A.heads}×${A.headDim}`, 'decay');
      chip(CS + 2, yB + 42, 'β', String(A.heads), 'beta');
      if (M) chip(SX1 + 14, yAtt - 8, 'q, k, v', '', 'conv');
      box('kda_scan', C1, yAtt, W1, 'KDA scan — chunked delta rule',
        `state ${A.heads}×(${A.headDim}×${A.headDim}) · causal`, 36);
      wv(SX1, yAtt + 36, yGate);
      chip(SX1 + 14, yGate - 8, 'scan out', `${A.heads}×${A.headDim}`, 'kda_scan');
      pill('gate_mul1', C1, yGate, W1, `σ(gate) ⊙ RMSNorm${pk(K.gateNorm)}`);
    } else {
      // gated MLA: DSv3's MLA wiring minus RoPE (NoPE), plus the output gate
      const w1 = 132, x2 = C1 + w1 + 14, w2 = 134, sxk = x2 + 20;
      dot(sxk, yBr);
      poly([[sxk, yBr], [sxk, yA]], true);
      box('q_down', C1, yA, w1, 'q down-proj',
        P ? fmtP(K.qDown) : `${A.hidden} → ${A.qRank}${pk(K.qDown)}`);
      box('kv_down', x2, yA, w2, 'kv down-proj',
        P ? fmtP(K.kvDown) : `${A.hidden} → ${A.kvRank} + ${A.qkRope}${pk(K.kvDown)}`);
      wv(SX1, yA + 32, yB);
      wv(sxk, yA + 32, yB);
      chip(SX1 + 14, yB - 6, 'q latent', A.qRank, 'q_down');
      chip(sxk + 14, yB - 6, 'kv latent', A.kvRank, 'kv_down');
      pill('q_norm', C1, yB, w1, `RMSNorm${pk(A.qRank)}`, 18);
      pill('kv_norm', x2, yB, w2, `RMSNorm${pk(A.kvRank)}`, 18);
      // the 64-dim k slice: split from kv down, shared across heads, NEVER
      // rotated (NoPE) — the rail is the same shape as DSv3's k_rope
      poly([[x2 + w2, yA + 10], [RR1, yA + 10], [RR1, yAtt + 8], [C1 + W1 + 3, yAtt + 8]], true);
      chip(x2 + w2 + 10, yA + 6, `k shared · ${A.qkRope} (unrotated)`);
      if (this.detail) micro(CS, yB - 4, WS, 'NoPE — no rotary anywhere');
      wv(SX1, yB + 18, yC);
      wv(sxk, yB + 18, yC);
      box('q_up', C1, yC, w1, 'q up-proj',
        P ? fmtP(K.qUp) : F ? `${A.qRank} → ${A.heads * (A.qkNope + A.qkRope)}${pk(K.qUp)}` : `${A.qRank} → ${A.heads}×${A.qkNope + A.qkRope}${pk(K.qUp)}`);
      box('kv_up', x2, yC, w2, 'kv up-proj',
        P ? fmtP(K.kvUp) : F ? `${A.kvRank} → ${A.heads * (A.qkNope + A.vHead)}${pk(K.kvUp)}` : `${A.kvRank} → ${A.heads}×(${A.qkNope}+${A.vHead})${pk(K.kvUp)}`);
      wv(SX1, yC + 32, yAtt);
      wv(sxk, yC + 32, yAtt);
      chip(SX1 + 14, yAtt - 8, 'q', `${A.heads}×${A.qkNope + A.qkRope}`, 'q_up');
      chip(sxk + 14, yAtt - 8, 'k,v', `${A.heads}×(${A.qkNope}+${A.vHead})`, 'kv_up');
      box('attn', C1, yAtt, W1, 'attention', 'softmax(QKᵀ)V · causal', 36);
      wv(SX1, yAtt + 36, yGate);
      chip(SX1 + 14, yGate - 8, 'attn out', `${A.heads}×${A.vHead}`, 'attn');
      pill('gate_mul1', C1, yGate, W1, 'σ(gate) ⊙');
    }
    // shared tail: out-proj, residual add
    wv(SX1, yGate + 20, yO);
    box('o_proj', C1, yO, W1, 'attn out-proj',
      P ? fmtP(K.oProj) : F ? `${PJ} → ${A.hidden}${pk(K.oProj)}` : `${A.heads}×${A.headDim} → ${A.hidden}${pk(K.oProj)}`);
    wv(SX1, yO + 32, yAdd1 - 2);
    chip(SX1 + 14, yO + 46, 'attn proj out', '', 'o_proj');
    res('res1', SX1, yAdd1, 150, 'residual add');
    wv(SX1, yAdd1 + 17, yX1, false);
    chip(SX1 + 14, yX1 + 14, 'x1 (partial sum)', A.hidden, 'res1');
    // x1 → FFN column top (up and over), plus its rail to the final add
    dot(SX1, yX1);
    poly([[SX1, yX1], [RAIL2, yX1], [RAIL2, yAdd2 + 10], [SX2 - 9, yAdd2 + 10]], true);
    dot(UPX, yX1);
    poly([[UPX, yX1], [UPX, 32], [SX2, 32], [SX2, yRes0]], true);
    pill('attnres2', C2, yRes0, W2, `attn-res mix — α over block reps${pk(K.attnRes / 2)}`);
    store(C2 + W2 + 16, 182, C2 + W2);
    wv(SX2, yRes0 + 20, yN1);
    pill('norm2', C2, yN1, W2, `RMSNorm${pk(A.hidden)}`);
    wv(SX2, yN1 + 20, this.kindF === 'moe' ? yRout : yGU);   // dense: one run into the FFN
    chip(SX2 + 14, yChip1, 'norm2 out', A.hidden, 'norm2');

    // ---- FFN column tabs + enclosure ---------------------------------------
    const f1x = C2 + 24, f1w = 186, f2x = f1x + f1w + 8, f2w = 138;
    S.push(tab(f1x, f1w, 'moe', this.kindF, 'LatentMoE FFN',
      `×${A.layers - A.denseLayers} · ${fmtP(this.activeView ? K.activeMoeFfn : K.moeFfn)}`, 'f'));
    S.push(tab(f2x, f2w, 'dense', this.kindF, 'dense FFN', `×${A.denseLayers} · ${fmtP(K.denseFfn)}`, 'f'));
    const ft = this.kindF === 'moe' ? { x: f1x, w: f1w } : { x: f2x, w: f2w };
    enc(C2 - 10, yEnc, CSH + WSH + 10 - (C2 - 10), yEnc2B - yEnc, ft.x, ft.w);

    if (this.kindF === 'moe') {
      const EX = this.activeView ? A.topk : A.routedExperts;
      // shared experts read norm2 out: branch under the tab row, rail outside
      // the routed group, into the sub-column
      dot(SX2, yBr);
      poly([[SX2, yBr], [SHX, yBr], [SHX, yGU + 10], [CSH - 3, yGU + 10]], true);
      box('router', C2, yRout, W2, 'router',
        P ? fmtP(K.routerWeight) : `${A.hidden} → ${A.routedExperts}${pk(K.routerWeight)}`);
      wv(SX2, yRout + 32, yRoutM);
      S.push(g('router_bias'));
      micro(C2, yRoutM, W2, `sigmoid · biased top-${A.topk} · renorm (${A.routedExperts})`);
      S.push(`</g>`);
      wv(SX2, yRoutM + 16, yLatD);
      // top-k weights rail: router → the gating multiply inside the expert group
      poly([[C2 + W2, yRoutM + 8], [GATEX, yRoutM + 8], [GATEX, ySw + 10], [C2 + W2 + 3, ySw + 10]], true);
      if (!P) S.push(`<text class="tensor tidle" text-anchor="end" x="${GATEX - 6}" y="282">· top-k weights <tspan class="tdim">· ${A.topk}</tspan></text>`);
      box('lat_down', C2, yLatD, W2, 'latent down-proj',
        P ? fmtP(K.latDown) : `${A.hidden} → ${A.moeLatent}${pk(K.latDown)}`);
      wv(SX2, yLatD + 32, yDisp);
      chip(SX2 + 14, yLatD + 46, 'z (latent)', A.moeLatent, 'lat_down');
      comm('dispatch', C2, yDisp, W2, 'a2a dispatch (permute + comm) → EP group');
      wv(SX2, yDisp + 22, yGU);
      chip(SX2 + 14, yDisp + 38, 'dispatched tokens', `${A.topk}×${A.moeLatent}`, 'dispatch');
      grp(C2 - 6, yGrp, W2 + 12, yGrpB - yGrp, `routed experts ×${A.routedExperts} · ${fmtP(K.expert)} each`, SX2 + 20 - C2);
      box('gate_up', C2, yGU, W2, `ffn gate/up (grouped ×${A.topk})`,
        P ? `${fmtP(2 * K.expert / 3)} ×${EX}` : F ? `${A.moeLatent} → ${2 * A.moeInter} (${fmtP(2 * K.expert / 3 * EX)})` : `${A.moeLatent} → 2×${A.moeInter}${pk(2 * K.expert / 3, EX)}`);
      wv(SX2, yGU + 32, ySw);
      chip(SX2 + 14, yGU + 46, 'gate, up (routed)', `${A.topk}×2×${A.moeInter}`, 'gate_up');
      pill('situ', C2, ySw, W2, 'SiTU-GLU · × top-k weight');
      wv(SX2, ySw + 20, yFD);
      box('ffn_down', C2, yFD, W2, `ffn down (grouped ×${A.topk})`,
        P ? `${fmtP(K.expert / 3)} ×${EX}` : F ? `${A.moeInter} → ${A.moeLatent} (${fmtP(K.expert / 3 * EX)})` : `${A.moeInter} → ${A.moeLatent}${pk(K.expert / 3, EX)}`);
      wv(SX2, yFD + 32, yComb);
      chip(SX2 + 14, yGrpB + 12, 'expert outputs', '', 'ffn_down');
      comm('combine', C2, yComb, W2, 'a2a combine (comm + unpermute · sum)');
      wv(SX2, yComb + 22, yLatN);
      pill('lat_norm', C2, yLatN, W2, `RMSNorm${pk(A.moeLatent)}`, 18);
      wv(SX2, yLatN + 18, yLatU);
      box('lat_up', C2, yLatU, W2, 'latent up-proj',
        P ? fmtP(K.latUp) : `${A.moeLatent} → ${A.hidden}${pk(K.latUp)}`);
      wv(SX2, yLatU + 32, yMAdd - 2);
      chip(SX2 + 14, yLatU + 46, 'moe out', '', 'lat_up');
      // shared experts: every token, full width, alongside the routed group
      grp(CSH - 6, yGrp, WSH + 12, yGrpB - yGrp);
      S.push(`<text class="grplabel" x="${CSH - 6}" y="${yGrp - 8}">shared experts ×${A.sharedExperts} (every token)</text>`);
      box('shared_gu', CSH, yGU, WSH, 'shared gate/up',
        P ? fmtP(2 * K.shared / 3) : F ? `${A.hidden} → ${2 * A.sharedExperts * A.moeInter}${pk(2 * K.shared / 3)}` : `${A.hidden} → 2×${A.sharedExperts * A.moeInter}${pk(2 * K.shared / 3)}`);
      wv(SXSH, yGU + 32, ySw);
      pill('shared_situ', CSH, ySw, WSH, 'SiTU-GLU');
      wv(SXSH, ySw + 20, yFD);
      box('shared_down', CSH, yFD, WSH, 'shared down',
        P ? fmtP(K.shared / 3) : `${A.sharedExperts * A.moeInter} → ${A.hidden}${pk(K.shared / 3)}`);
      chip(SXSH + 8, yGrpB + 12, 'shared out', '', 'shared_down');
      poly([[SXSH, yFD + 32], [SXSH, yMAdd + 10], [SX2 + 14 + 210 + 3, yMAdd + 10]], true);
      res('moe_add', SX2, yMAdd, 210, 'add — routed + shared');
      wv(SX2, yMAdd + 17, yAdd2 - 2);
    } else {
      // dense: one wide FFN on the same spine; whitespace where routing sits
      box('ffn_gate_up_d', C2, yGU, W2, 'ffn gate/up',
        P ? fmtP(2 * K.denseFfn / 3) : F ? `${A.hidden} → ${2 * A.denseInter}${pk(2 * K.denseFfn / 3)}` : `${A.hidden} → 2×${A.denseInter}${pk(2 * K.denseFfn / 3)}`);
      wv(SX2, yGU + 32, ySw);
      chip(SX2 + 14, yGU + 46, 'gate, up', `2×${A.denseInter}`, 'ffn_gate_up_d');
      pill('situ_d', C2, ySw, W2, 'SiTU-GLU');
      wv(SX2, ySw + 20, yFD);
      box('ffn_down_d', C2, yFD, W2, 'ffn down',
        P ? fmtP(K.denseFfn / 3) : `${A.denseInter} → ${A.hidden}${pk(K.denseFfn / 3)}`);
      wv(SX2, yFD + 32, yAdd2 - 2);
      chip(SX2 + 14, yFD + 46, 'ffn out', '', 'ffn_down_d');
    }
    res('res2', SX2, yAdd2, 150, 'residual add');
    wv(SX2, yAdd2 + 17, yAdd2 + 36, false);
    chip(SX2 + 14, yAdd2 + 50, 'x2 (block output)', '');

    // ---- assemble ----------------------------------------------------------
    const head = document.createElement('div'); head.className = 'kv-head';
    const lab = document.createElement('span'); lab.textContent = 'sizes:';
    const b = document.createElement('button');
    b.textContent = this.flatDims ? String(PJ) : `${A.heads}×${A.headDim}`;
    b.title = `toggle sizes: factored (${A.heads}×${A.headDim}) vs multiplied out (${PJ})`;
    b.onclick = () => { this.flatDims = !this.flatDims; this.render(); this.changed(); };
    head.append(lab, b);
    const scroll = document.createElement('div'); scroll.className = 'kv-scroll';
    scroll.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${S.join('')}</svg>`;
    const parts = [head, scroll];
    if (M) {
      const pol = this.policy();
      const pols = document.createElement('span'); pols.className = 'pols';
      pols.innerHTML = 'recompute: ' + K3_POLICIES.map(([k2, lb]) =>
        `<button type="button" data-pol="${k2}"${pol === k2 ? ' class="on"' : ''}>${lb}</button>`).join('') +
        `<span class="cust">${pol ? '' : 'custom — click ops to flip save ↔ ↻'}</span>`;
      head.append(pols);
      for (const pb of pols.querySelectorAll('[data-pol]')) pb.onclick = () => {
        if (this.policy() === pb.dataset.pol) return;
        this.marks = { ...K3_RECOMPUTE[pb.dataset.pol] }; this.render(); this.changed();
      };
      const led = document.createElement('div'); led.className = 'kv-led';
      const st = graph.filter((n) => ana.neededSaved.has(n.id) || (n.aux && ana.savedById[n.id]))
        .map((n) => `<span class="st" data-op="${n.id}">↓ ${n.tensor} ${fmtB(ana.savedById[n.id])}</span>`);
      const rp = graph.filter((n) => ana.replayed.has(n.id)).map((n) => `<span class="rp" data-op="${n.id}">↻ ${n.tensor}</span>`);
      const kindN = K3_KINDS.find(([a2, f2]) => a2 === this.kindA && f2 === this.kindF)?.[2] ?? 0;
      led.innerHTML = `<b>stashed for backward: ${fmtB(ana.savedBytes)} per token</b> for this block ` +
        `(${this.kindA === 'kda' ? 'KDA' : 'gated MLA'} + ${this.kindF === 'moe' ? 'LatentMoE' : 'dense FFN'}, ×${kindN}): ${st.join(' · ')}` +
        (rp.length ? `<br>replayed in backward (+${(100 * ana.replayFrac).toFixed(ana.replayFrac < 0.01 ? 2 : 0)} % of the block’s forward FLOPs${ana.replayComm.length ? `, plus the ${ana.replayComm.join(' and ')} all-to-all` : ''}): ${rp.join(' · ')}` : '') +
        `<br><b>whole model: ${fmtB(k3Stash(this.marks))} per token</b> over all ${A.layers} layers under these marks, ` +
        `plus the head’s logits and the AttnRes block store (both outside the blocks)`;
      parts.push(led);
    }
    this._card.replaceChildren(...parts);
    for (const mg of this._card.querySelectorAll('svg [data-mk]')) mg.addEventListener('click', () => {
      const id = mg.dataset.mk;
      if (this.marks[id] === true) delete this.marks[id]; else this.marks[id] = true;
      this.render(); this.changed();
    });
    for (const g of this._card.querySelectorAll('[data-tab]')) {
      g.addEventListener('click', () => {
        const [onto, key] = g.dataset.tab.split(':');
        if (onto === 'a') { if (this.kindA === key) return; this.kindA = key; }
        else { if (this.kindF === key) return; this.kindF = key; }
        this.render(); this.changed();
      });
    }
    this.applyHl();
  }
}
customElements.define('kimi-layer', KimiLayer);

// ---- <kimi-anatomy-plan> ---------------------------------------------------
// The margin plan: embedding → [KDA ×3 → gated MLA] ×23 → final gated MLA →
// attn-res α → final RMSNorm → lm head → softmax/loss. Clicking a block
// flips the diagram's attention kind; the dashed cone ties plan to diagram.

const PLAN_CSS = `
kimi-anatomy-plan { display: block; }
.kanp { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-0b0b0b); }
.kanp svg { display: block; max-width: 100%; height: auto; }
${tokensCss('.kanp')}
.kanp .box.on { fill: var(--c-fff8ea); stroke: var(--c-eda100); }
.kanp [data-kind] { cursor: pointer; }
.kanp [data-kind].on { cursor: default; }
.kanp svg.hlm > :not(.hl):not(defs) { opacity: 0.3; }
.kanp g[data-op].hl rect { fill: var(--c-fff8ea); stroke: var(--c-eda100); }
.kanp g[data-op].hl .dims { fill: var(--c-52514e); font-weight: 600; }
`;

export class KimiAnatomyPlan extends HTMLElement {
  connectedCallback() {
    const style = document.createElement('style'); style.textContent = PLAN_CSS;
    this._root = document.createElement('div'); this._root.className = 'kanp';
    this.append(style, this._root);
    const host = this.parentElement;
    if (host && getComputedStyle(host).position === 'static') host.style.position = 'relative';
    this._ov = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    this._ov.setAttribute('class', 'anat-cone');
    this._ov.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;overflow:visible;';
    host?.append(this._ov);
    this.draw();
    queueMicrotask(() => {
      const le = this.layerEl();
      le?.addEventListener('recipe', () => this.draw());
      if (le && 'ResizeObserver' in window) new ResizeObserver(() => this.expansion()).observe(le);
      this.draw();
    });
    const up = () => this.expansion();
    window.addEventListener('resize', up);
    window.addEventListener('scroll', up, { passive: true });
    addEventListener('dsv3-theme', () => this.draw());
  }
  layerEl() { return document.getElementById(this.getAttribute('layer') ?? ''); }
  expansion() {
    if (!this._ov) return;
    const src = this._root.querySelector('g.on rect');
    const dst = this.layerEl()?.querySelector('.kv');
    if (!src || !dst) { this._ov.innerHTML = ''; return; }
    const h = this._ov.getBoundingClientRect(), a = src.getBoundingClientRect(), b = dst.getBoundingClientRect();
    const L = (x1, y1, x2, y2) => {
      const g = Math.max(24, (x2 - x1) * 0.6);
      return `<path d="M ${x1 - h.left} ${y1 - h.top} C ${x1 - h.left + g} ${y1 - h.top}, ` +
        `${x2 - h.left - g} ${y2 - h.top}, ${x2 - h.left} ${y2 - h.top}" ` +
        `fill="none" stroke="${C('#c3c2b7')}" stroke-width="1.2" stroke-dasharray="5 4"/>`;
    };
    this._ov.innerHTML = L(a.right, a.top, b.left, b.top) + L(a.right, a.bottom, b.left, b.bottom);
  }
  draw() {
    const l = this.layerEl();
    const kind = l?.kindA ?? 'kda';
    const PL = l?.getAttribute('lens') === 'params';
    const S = [];
    S.push(`<defs><marker id="kplanarr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" ` +
      `orient="auto-start-reverse"><path d="M 0 0 L 8 4 L 0 8 z" fill="${C('#898781')}"/></marker></defs>`);
    const BX = 8, W = 158, CX = BX + W / 2;
    let y = 14;
    const wire = (gap, name) => {
      S.push(`<line class="wire" x1="${CX}" y1="${y}" x2="${CX}" y2="${y + gap}" marker-end="url(#kplanarr)"/>`);
      if (name) S.push(`<text class="dims" x="${CX + 7}" y="${y + gap / 2 + 3}" font-style="italic">${name}</text>`);
      y += gap;
    };
    const op = (label, dims, opId) => {
      S.push(`${opId ? `<g data-op="${opId}">` : ''}<rect class="op" x="${BX}" y="${y}" width="${W}" height="22" rx="6"/>` +
        `<text class="oplabel" x="${BX + 9}" y="${y + 15}">${label}${dims ? ` <tspan class="dims">${dims}</tspan>` : ''}</text>` +
        `${opId ? '</g>' : ''}`);
      y += 22;
    };
    const blockBox = (k, label, dims, x = BX, w = W) => {
      const on = kind === k;
      S.push(`<g data-kind="${k}" data-op="block-${k}" class="${on ? 'on' : ''}">` +
        `<rect class="box${on ? ' on' : ''}" x="${x}" y="${y}" width="${w}" height="34" rx="4"/>` +
        `<text class="name" x="${x + 8}" y="${y + 14}">${label}</text>` +
        `<text class="dims" x="${x + 8}" y="${y + 27}">${dims}</text></g>`);
      y += 34;
    };
    S.push(`<text class="oplabel" x="${CX - 22}" y="${y - 2}">tokens</text>`);
    wire(14);
    op('embedding', `(${fmtP(K.embed)})`, 'embed');
    wire(24, `x · ${A.hidden}`);
    // the repeating 4-layer group: 3× KDA then one gated MLA
    const gTop = y;
    y += 20;
    blockBox('kda', 'KDA block ×3', `${fmtP(K.kdaMoeBlock)} each`, BX + 6, W - 12);
    wire(22, `x · ${A.hidden}`);
    blockBox('mla', 'gated MLA block', `${fmtP(K.mlaMoeBlock)}`, BX + 6, W - 12);
    y += 8;
    S.push(`<rect class="grp" x="${BX - 4}" y="${gTop}" width="${W + 8}" height="${y - gTop}" rx="6"/>` +
      `<text class="grplabel" x="${BX + 4}" y="${gTop + 14}">×23 (layers 1–92)</text>`);
    S.push(`<text class="dims" x="${BX}" y="${y + 12}" font-style="italic">layer 1's FFN is dense (${fmtP(K.denseFfn)})</text>`);
    y += 18;
    wire(18, `x · ${A.hidden}`);
    blockBox('mla', 'gated MLA block', 'layer 93 — always global');
    wire(22);
    op('attn-res α (output)', `(${fmtP(2 * A.hidden)})`, 'final_res');
    op('final RMSNorm', `(${fmtP(A.hidden)})`, 'final_norm');
    wire(24, `norm out · ${A.hidden}`);
    S.push(`<g data-op="lm_head"><rect class="box" x="${BX}" y="${y}" width="${W}" height="34" rx="4"/>` +
      `<text class="name" x="${BX + 8}" y="${y + 14}">lm head</text>` +
      `<text class="dims" x="${BX + 8}" y="${y + 27}">${PL ? `(${fmtP(K.embed)})` : `${A.hidden} → ${A.vocab} (${fmtP(K.embed)})`}</text></g>`);
    y += 34;
    wire(24, `logits · ${A.vocab}`);
    op('softmax / loss', null);
    y += 8;
    this._root.innerHTML = `<svg viewBox="0 0 ${BX + W + 10} ${y + 30}" width="${BX + W + 10}" height="${y + 30}">${S.join('')}</svg>`;
    for (const g of this._root.querySelectorAll('[data-kind]')) {
      g.onclick = () => {
        const l = this.layerEl();
        if (!l || l.kindA === g.dataset.kind) return;
        l.kindA = g.dataset.kind; l.render(); l.changed();
        this.draw();
      };
    }
    this.applyHl();
    requestAnimationFrame(() => this.expansion());
  }
  highlightOps(ids) { this._hl = ids ? new Set(ids) : null; this.applyHl(); }
  applyHl() { applyHighlight(this._root, this._hl); }
}
customElements.define('kimi-anatomy-plan', KimiAnatomyPlan);

// ---- <kimi-param-tally> ----------------------------------------------------
// Same interaction model as <dsv3-param-tally> (hover previews, click pins,
// term-level highlights, exact-value tooltips), rows keyed to K3's three
// block types. Counting conventions in the study's footnote.

const ATTN_KDA_IN = ['qkv', 'conv', 'decay', 'beta'];
const ATTN_KDA_OUT = ['gate1', 'gate_mul1', 'o_proj'];
const ATTN_MLA_Q = ['q_down', 'q_norm', 'q_up'];
const ATTN_MLA_KV = ['kv_down', 'kv_norm', 'kv_up'];
const MOE_EXPERTS = ['gate_up', 'ffn_down', 'shared_gu', 'shared_down'];
const MOE_LATENTS = ['lat_down', 'lat_norm', 'lat_up'];
const NORM_OPS = ['norm1', 'norm2', 'attnres1', 'attnres2'];
const kdaAttnIn = K.qkv + K.conv + K.decay + K.beta;
const kdaAttnOut = K.gate + K.gateNorm + K.oProj;
const mlaGateOut = K.gate + K.oProj;
const mkMoeTerms = (active) => [
  { name: `experts × (${active ? `${A.topk} active` : `${A.routedExperts} routed`} + ${A.sharedExperts} shared)`,
    nv: (active ? K.activeExperts : K.experts), ops: MOE_EXPERTS },
  { name: 'latent down·norm·up', nv: K.latents, ops: MOE_LATENTS },
  { name: 'router', nv: K.routerWeight, ops: ['router'] },
  { name: 'correction bias', nv: K.routerBias, ops: ['router_bias'] },
  { name: 'norms + attn-res', nv: K.misc, ops: NORM_OPS },
];
const TALLY_ROWS = [
  { label: 'embedding', kindA: null, plan: ['embed'],
    terms: [{ name: 'lookup table', val: `${A.hidden} × ${A.vocab}`, nv: K.embed, ops: [] }],
    per: K.embed, count: 1, mult: '× 1',
    active: { per: 0, terms: [{ name: 'lookup', val: 'not counted', ops: [] }] } },
  { label: 'KDA block, dense FFN', kindA: 'kda', kindF: 'dense', plan: ['block-kda'],
    terms: [
      { name: 'attn in (qkv·conv·α·β)', nv: kdaAttnIn, ops: ATTN_KDA_IN },
      { name: 'attn out (gate·proj)', nv: kdaAttnOut, ops: ATTN_KDA_OUT },
      { name: 'dense ffn', nv: K.denseFfn, ops: ['ffn_gate_up_d', 'situ_d', 'ffn_down_d'] },
      { name: 'norms + attn-res', nv: K.misc, ops: NORM_OPS },
    ],
    per: K.kdaDenseBlock, count: 1, mult: '× 1 (layer 1)' },
  { label: 'KDA block, MoE FFN', kindA: 'kda', kindF: 'moe', plan: ['block-kda'],
    terms: [
      { name: 'attn in (qkv·conv·α·β)', nv: kdaAttnIn, ops: ATTN_KDA_IN },
      { name: 'attn out (gate·proj)', nv: kdaAttnOut, ops: ATTN_KDA_OUT },
      ...mkMoeTerms(false),
    ],
    per: K.kdaMoeBlock, count: K.nKdaMoe, mult: `× ${K.nKdaMoe} blocks`,
    active: { per: K.kdaAttn + K.activeMoeFfn + K.misc,
      terms: [
        { name: 'attn in (qkv·conv·α·β)', nv: kdaAttnIn, ops: ATTN_KDA_IN },
        { name: 'attn out (gate·proj)', nv: kdaAttnOut, ops: ATTN_KDA_OUT },
        ...mkMoeTerms(true),
      ] } },
  { label: 'gated-MLA block, MoE FFN', kindA: 'mla', kindF: 'moe', plan: ['block-mla'],
    terms: [
      { name: 'attn q (down·norm·up)', nv: K.qDown + K.qNorm + K.qUp, ops: ATTN_MLA_Q },
      { name: 'attn kv (down·norm·up)', nv: K.kvDown + K.kvNorm + K.kvUp, ops: ATTN_MLA_KV },
      { name: 'attn out (gate·proj)', nv: mlaGateOut, ops: ['gate1', 'gate_mul1', 'o_proj'] },
      ...mkMoeTerms(false),
    ],
    per: K.mlaMoeBlock, count: K.nMla, mult: `× ${K.nMla} blocks`,
    active: { per: K.mlaAttn + K.activeMoeFfn + K.misc,
      terms: [
        { name: 'attn q (down·norm·up)', nv: K.qDown + K.qNorm + K.qUp, ops: ATTN_MLA_Q },
        { name: 'attn kv (down·norm·up)', nv: K.kvDown + K.kvNorm + K.kvUp, ops: ATTN_MLA_KV },
        { name: 'attn out (gate·proj)', nv: mlaGateOut, ops: ['gate1', 'gate_mul1', 'o_proj'] },
        ...mkMoeTerms(true),
      ] } },
  { label: 'final norm + attn-res', kindA: null, plan: ['final_norm', 'final_res'],
    terms: [{ name: 'RMSNorm + α (norm·proj)', nv: K.finalNorms, ops: [] }],
    per: K.finalNorms, count: 1, mult: '× 1' },
  { label: 'lm head', kindA: null, plan: ['lm_head'],
    terms: [{ name: 'output matrix', val: `${A.hidden} × ${A.vocab}`, nv: K.embed, ops: [] }],
    per: K.embed, count: 1, mult: '× 1' },
];
for (const r of TALLY_ROWS) {
  for (const t of r.terms.concat(r.active?.terms ?? [])) if (!t.val) t.val = fmtP(t.nv);
  r.ops = [...new Set(r.terms.flatMap(t => t.ops))];
  if (r.active) r.active.ops = [...new Set(r.active.terms.flatMap(t => t.ops))];
}
const rowIn = (r, mode) => mode === 'active' && r.active ? { ...r, ...r.active } : r;

const TALLY_CSS = `
kimi-param-tally { display: block; margin: 14px 0; }
.ptal { font: 13.5px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-0b0b0b); position: relative; }
.ptal .pnum { cursor: pointer; }
.ptal table { border-collapse: collapse; width: 100%; max-width: 760px; }
.ptal th, .ptal td { text-align: left; padding: 5px 12px 5px 7px; border-bottom: 1px solid var(--c-e1e0d9);
  font-variant-numeric: tabular-nums; vertical-align: top; }
.ptal th { color: var(--c-52514e); font-weight: 600; font-size: 12.5px; }
.ptal td.num { text-align: right; padding-right: 0; white-space: nowrap; }
.ptal .title { font: 600 12px system-ui; color: var(--c-52514e); margin: 0 0 4px; }
.ptal .formula { color: var(--c-898781); font-size: 12.5px; }
.ptal td .fterm { border-bottom: 1px dotted var(--c-c3c2b7); }
.ptal .fterm:hover { color: var(--c-0b0b0b); }
.ptal td .fterm:hover { border-bottom-color: var(--c-52514e); }
.ptal .fterm.pin { color: var(--c-0b0b0b); font-weight: 600; }
.ptal td .fterm.pin { border-bottom: 1px solid var(--c-52514e); }
.ptal .fxline { display: grid; grid-template-columns: 10px 1fr auto; gap: 0 6px; align-items: baseline; }
.ptal .fxline .fxop { color: var(--c-a8a69e); }
.ptal .fxline .fxval { text-align: right; font-variant-numeric: tabular-nums; }
.ptal tbody tr { cursor: pointer; }
.ptal tbody tr:hover { background: var(--c-f7f6f1); }
.ptal tbody tr.sel { background: var(--c-ffffff); box-shadow: inset 3px 0 0 var(--c-52514e); }
.ptal tbody tr.sel td:first-child { font-weight: 600; }
.ptal tfoot td { font-weight: 600; border-bottom: none; }
.ptal .mbtn { font-weight: 400; color: var(--c-898781); cursor: pointer; border-bottom: 1px dotted var(--c-c3c2b7); }
.ptal .mbtn.on { font-weight: 600; color: var(--c-0b0b0b); border-bottom: 1px solid var(--c-52514e); cursor: default; }
.ptal.compact { font-size: 11.5px; }
.ptal.compact .title { font: 600 11px system-ui; color: var(--c-52514e); margin: 0 0 2px; }
.ptal.compact td { padding: 3px 6px 3px 7px; }
.ptal.compact .formula { font-size: 10px; display: block; }
.ptal.compact .fxout { min-height: 120px; padding: 4px 0 0 7px; font-size: 10px;
  color: var(--c-52514e); line-height: 1.5; }
`;

export class KimiParamTally extends HTMLElement {
  connectedCallback() {
    this._mode = this.getAttribute('mode') === 'active' ? 'active' : 'total';
    const style = document.createElement('style'); style.textContent = TALLY_CSS;
    this._root = document.createElement('div');
    this._root.className = 'ptal' + (this.hasAttribute('compact') ? ' compact' : '');
    this.append(style, this._root);
    // the shared tooltip: a count's exact value (outside _root, which build() rewrites)
    attachTip(this._root, (ev) => {
      const sp = ev.target.closest?.('.pnum');
      return sp ? Number(sp.dataset.v).toLocaleString('en-US') : null;
    }, { parent: this });
    this.build();
  }
  build() {
    const compact = this.hasAttribute('compact');
    const mode = this._mode;
    const root = this._root;
    const rows = TALLY_ROWS.map(r => rowIn(r, mode));
    const total = rows.reduce((t, r) => t + r.per * r.count, 0);
    const num = (v) => `<span class="pnum" data-v="${v}">${fmtPT(v)}</span>`;
    const modeBtn = (m, label) =>
      `<span class="mbtn${mode === m ? ' on' : ''}" data-mode="${m}">${label}</span>`;
    const head = `parameters: ${modeBtn('total', 'total')} · ${modeBtn('active', 'active / token')}`;
    root.innerHTML =
      `<div class="title">${head}</div><table>` +
      (compact ? '' : `<thead><tr><th>component</th><th>parameters, per copy</th>` +
        `<th>copies</th><th style="text-align:right">total</th></tr></thead>`) +
      `<tbody>` +
      rows.map((r, i) => compact
        ? `<tr data-row="${i}"><td>${r.label}<span class="formula">${fmtP(r.per)} ${r.mult}</span></td>` +
          `<td class="num">${num(r.per * r.count)}</td></tr>`
        : `<tr data-row="${i}"><td>${r.label}</td>` +
          `<td><span class="formula">${r.terms.map((t, j) => `<span class="fterm" data-t="${j}">${t.name} ${t.val}</span>`).join(' + ')} =</span> ${num(r.per)}</td>` +
          `<td>${r.mult}</td><td class="num">${num(r.per * r.count)}</td></tr>`).join('') +
      `</tbody><tfoot><tr><td${compact ? '' : ' colspan="3"'}>total</td><td class="num">${num(total)}</td></tr></tfoot></table>` +
      (compact ? `<div class="fxout"></div>` : '');
    const lid = this.getAttribute('layer') ?? '';
    const layer = () => document.getElementById(lid);
    const plan = () => document.querySelector(`kimi-anatomy-plan[layer="${lid}"]`);
    const state = { pin: null, hover: null };
    const rowOf = (st) => rows[st.ri];
    const opsOf = (st) => st.ti == null ? rowOf(st).ops : rowOf(st).terms[st.ti].ops;
    const termsHtml = (r) => r.terms.map((t, i) =>
      `<div class="fterm fxline" data-t="${i}"><span class="fxop">${i ? '+' : '='}</span>` +
      `<span class="fxname">${t.name}</span><span class="fxval">${t.val}</span></div>`).join('');
    const wireTerms = (container, ri, inRow = true) => {
      for (const sp of container.querySelectorAll('.fterm')) {
        const ti = +sp.dataset.t;
        sp.onmouseenter = () => { state.hover = { ri, ti }; apply(); };
        sp.onmouseleave = () => { state.hover = inRow ? { ri, ti: null } : null; apply(); };
        sp.onclick = (ev) => {
          ev.stopPropagation();
          if (state.pin?.ri === ri && state.pin?.ti === ti) state.pin = { ri, ti: null };
          else pinTo(ri, ti);
          apply();
        };
      }
    };
    const pinTo = (ri, ti) => {
      state.pin = { ri, ti };
      const r = rows[ri], l = layer();
      if (!l) return;
      let dirty = false;
      if (r.kindA && l.kindA !== r.kindA) { l.kindA = r.kindA; dirty = true; }
      if (r.kindF && l.kindF !== r.kindF) { l.kindF = r.kindF; dirty = true; }
      if (dirty) { l.render(); l.changed(); }
    };
    const apply = () => {
      const cur = state.hover ?? state.pin;
      layer()?.highlightOps?.(cur ? opsOf(cur) : null);
      plan()?.highlightOps?.(cur ? rowOf(cur).plan : null);
      for (const tr of root.querySelectorAll('tbody tr'))
        tr.classList.toggle('sel', state.pin?.ri === +tr.dataset.row);
      const fx = root.querySelector('.fxout');
      if (fx) {
        const show = state.hover ?? state.pin;
        const want = show ? String(show.ri) : '';
        if (fx.dataset.ri !== want) {
          fx.dataset.ri = want;
          fx.innerHTML = want === '' ? '' : termsHtml(rowOf(show));
          if (want !== '') wireTerms(fx, show.ri, false);
        }
      }
      for (const sp of root.querySelectorAll('.fterm')) {
        const ri = sp.closest('tr') ? +sp.closest('tr').dataset.row : state.pin?.ri;
        sp.classList.toggle('pin', state.pin != null && state.pin.ti != null &&
          state.pin.ri === ri && state.pin.ti === +sp.dataset.t);
      }
    };
    for (const tr of root.querySelectorAll('tbody tr')) {
      const ri = +tr.dataset.row;
      tr.onmouseenter = () => { state.hover = { ri, ti: null }; apply(); };
      tr.onmouseleave = () => { state.hover = null; apply(); };
      tr.onclick = () => {
        if (state.pin?.ri === ri && state.pin?.ti == null) state.pin = null;
        else pinTo(ri, null);
        apply();
      };
      wireTerms(tr, ri);
    }
    // clicking a count also pins its row (the tooltip pins via src/tip.js)
    for (const sp of root.querySelectorAll('.pnum')) {
      sp.onclick = (ev) => {
        ev.stopPropagation();
        const tr = sp.closest('tbody tr');
        if (tr) pinTo(+tr.dataset.row, null);
        apply();
      };
    }
    for (const b of root.querySelectorAll('.mbtn')) {
      b.onclick = () => {
        if (this._mode === b.dataset.mode) return;
        this._mode = b.dataset.mode;
        const l = layer();
        if (l) {
          l.activeView = this._mode === 'active';
          l.highlightOps?.(null); l.render(); l.changed();
        }
        plan()?.highlightOps?.(null);
        this.build();
      };
    }
  }
}
customElements.define('kimi-param-tally', KimiParamTally);

// ---- <kimi-anatomy> --------------------------------------------------------
const ANAT_CSS = `
kimi-anatomy { display: block; margin: 14px 0 26px; }
kimi-anatomy .anat-grid { display: grid; grid-template-columns: 186px minmax(0, 1fr);
  gap: 0 28px; align-items: start; position: relative; left: 50%;
  transform: translateX(-50%); width: min(1330px, calc(100vw - 32px)); }
kimi-anatomy .anat-grid > * { min-width: 0; }
kimi-anatomy kimi-anatomy-plan { margin-top: 46px; }
@media (max-width: 860px) {
  kimi-anatomy .anat-grid { grid-template-columns: 1fr; gap: 18px 0; }
  kimi-anatomy kimi-anatomy-plan { margin-top: 0; }
  .anat-cone { display: none; }
}
`;
const FWD = ['detail', 'lens', 'nocaption', 'marks', 'recompute'];
export class KimiAnatomy extends HTMLElement {
  connectedCallback() {
    const lid = this.getAttribute('layer') ?? ((this.id || 'kimi-anatomy') + '-layer');
    const style = document.createElement('style'); style.textContent = ANAT_CSS;
    const grid = document.createElement('div'); grid.className = 'anat-grid';
    const col1 = document.createElement('div');
    const plan = document.createElement('kimi-anatomy-plan');
    plan.setAttribute('layer', lid);
    col1.append(plan);
    if (this.hasAttribute('tally')) {
      const tal = document.createElement('kimi-param-tally');
      tal.setAttribute('layer', lid);
      tal.setAttribute('compact', '');
      col1.append(tal);
    }
    const layer = document.createElement('kimi-layer');
    layer.id = lid;
    for (const a of FWD) if (this.hasAttribute(a)) layer.setAttribute(a, this.getAttribute(a));
    grid.append(col1, layer);
    this.append(style, grid);
  }
}
customElements.define('kimi-anatomy', KimiAnatomy);

// ---- <kimi-attnres-pipe> -------------------------------------------------
// The whole stack across a pipeline, concretely: the 93 layers grouped into
// AttnRes blocks, the chunks each rank runs, every transition's width (block
// representations + the partial sum, in units of d = one BF16 hidden vector,
// 14 KiB per token), then per rank what it sends (forward + backward) and
// how many blocks it holds per microbatch. PP and schedule are knobs; the
// figure's height never depends on them (per-rank rows are fixed-height
// column charts).
const PIPE_CSS = `
kimi-attnres-pipe { display: block; margin: 14px 0 26px; }
.kpp { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-0b0b0b); max-width: 940px;
  border: 1px solid var(--c-e1e0d9); border-radius: 6px; background: var(--c-fcfcfb); padding: 10px 12px; box-sizing: border-box; }
.kpp-head { display: flex; align-items: center; gap: 4px; padding-bottom: 6px; color: var(--c-52514e); flex-wrap: wrap; }
.kpp-head .lab { margin: 0 4px 0 14px; } .kpp-head .lab:first-child { margin-left: 0; }
.kpp-head button { font: 11px ui-monospace, monospace; padding: 2px 7px; border: 1px solid var(--c-c3c2b7);
  border-radius: 4px; background: var(--c-ffffff); color: var(--c-0b0b0b); cursor: pointer; }
.kpp-head button.on { background: var(--c-eeede7); font-weight: 600; cursor: default; }
.kpp svg { display: block; }
.kpp svg text { font: 10px system-ui; }
.kpp-led { font-size: 11.5px; color: var(--c-52514e); line-height: 1.55; padding-top: 4px; min-height: calc(2 * 1.55em); }
.kpp-led b { color: var(--c-0b0b0b); }
`;
const PIPE_PPS = [3, 4, 6, 8, 12, 24];
const PIPE_SCHEDS = [['chain', '1F1B'], ['ilv', 'interleaved, V = 2'], ['v', 'V shape']];
export class KimiAttnresPipe extends HTMLElement {
  connectedCallback() {
    const st = readState('k3p:' + (this.id || 'pipe'));
    const pp = +(st?.pp ?? this.getAttribute('pp') ?? 6);
    this.pp = PIPE_PPS.includes(pp) ? pp : 6;
    const s = st?.s ?? this.getAttribute('sched') ?? 'v';
    this.sched = PIPE_SCHEDS.some(([k]) => k === s) ? s : 'v';
    const style = document.createElement('style'); style.textContent = PIPE_CSS;
    this._card = document.createElement('div'); this._card.className = 'kpp';
    this.append(style, this._card);
    this.render();
    addEventListener('dsv3-theme', () => this.render());
  }
  render() {
    const P = this.pp, R = attnresPipe(P, this.sched), S = A.attnResBlock, L = A.layers;
    const W = 914, X0 = 58, CW = (W - X0 - 8) / L, x = (l) => X0 + l * CW;   // x(l) = the left edge of 1-indexed layer l + 1
    const out = [];
    const t = (tx, ty, s, o = '') => out.push(`<text x="${tx}" y="${ty}"${o}>${s}</text>`);
    const ink = C('#52514e'), grey = C('#898781');
    // row 1: AttnRes blocks (b₀ = the embedding, then one per 12 layers)
    const yB = 14, yL = 22, yC = 46, yT = 72, yS = 104, yH = 166;
    t(4, yL + 12, 'layers', ` fill="${grey}"`);
    out.push(`<rect x="${X0 - 26}" y="${yL}" width="20" height="16" rx="3" fill="${C('#fff8ea')}" stroke="${C('#eda100')}"/>`);
    t(X0 - 16, yB + 2, 'b₀', ` text-anchor="middle" fill="${ink}"`);
    for (let j = 1; j * S - S < L; j++) {
      const a = x((j - 1) * S), b = x(Math.min(j * S, L));
      out.push(`<path d="M ${a + 1} ${yB + 6} V ${yB + 3} H ${b - 1} V ${yB + 6}" fill="none" stroke="${C('#c3c2b7')}"/>`);
      t((a + b) / 2, yB, `b${'₀₁₂₃₄₅₆₇₈₉'[j]}`, ` text-anchor="middle" fill="${ink}"`);
    }
    for (let l = 0; l < L; l++) {   // MLA every 4th layer and the last; the rest KDA
      const mla = (l + 1) % 4 === 0 || l === L - 1;
      out.push(`<rect x="${x(l) + 0.5}" y="${yL}" width="${CW - 1}" height="16" fill="${C(mla ? '#d9e4f0' : '#f7f6f1')}" stroke="${C(mla ? '#4a6a8e' : '#d8d6cb')}" stroke-width="0.6"/>`);
    }
    // row 2: chunks, labeled with the rank that runs them
    t(4, yC + 12, 'ranks', ` fill="${grey}"`);
    R.ranks.forEach((r, c) => {
      const a = x(R.cuts[c]), b = x(R.cuts[c + 1]);
      out.push(`<rect x="${a + 0.5}" y="${yC}" width="${b - a - 1}" height="16" rx="2" fill="${C(r % 2 ? '#e1e0d9' : '#f0efe9')}"/>`);
      if (b - a > 12) t((a + b) / 2, yC + 11.5, r, ` text-anchor="middle" fill="${ink}"`);
    });
    // row 3: each transition's width (in d); a turn on the same rank sends nothing
    t(4, yT + 4, 'widths (d)', ` fill="${grey}"`);
    R.tr.forEach((tr) => {
      const xx = x(tr.l);
      out.push(`<line x1="${xx}" y1="${yC + 17}" x2="${xx}" y2="${yT - 8}" stroke="${C(tr.w ? '#52514e' : '#c3c2b7')}"/>`);
      t(xx, yT + 4, tr.w || '·', ` text-anchor="middle" fill="${tr.w ? C('#0b0b0b') : grey}"`);
    });
    // rows 4–5: per rank, fixed-height columns — sends (fwd + bwd, d) and blocks held
    const cols = (y0, h, vals, max, label, unit, hot) => {
      t(4, y0 + h - 2, label, ` fill="${grey}"`);
      const cw = (W - X0 - 8) / P;
      vals.forEach((v, r) => {
        const bh = max ? Math.max(1, h * v / max) : 0, bx = X0 + r * cw + cw * 0.15;
        out.push(`<rect x="${bx}" y="${y0 + h - bh}" width="${cw * 0.7}" height="${bh}" fill="${C(r === hot ? '#eda100' : '#c3c2b7')}"/>`);
        if (cw > 22 || r === hot) t(bx + cw * 0.35, y0 + h - bh - 3, `${v}${r === hot ? unit : ''}`, ` text-anchor="middle" fill="${C(r === hot ? '#8a5f00' : '#52514e')}"`);
      });
    };
    const hot = R.send.indexOf(R.rres);
    cols(yS, 44, R.send, Math.max(...R.send), 'sent / tok', ' d', hot);
    cols(yH, 30, R.held, Math.ceil(L / S), 'blocks held', '', R.held.indexOf(Math.max(...R.held)));
    const cw = (W - X0 - 8) / P;   // rank axis under the columns
    for (let r = 0; r < P; r++) if (cw > 14 || r % 2 === 0) t(X0 + (r + 0.5) * cw, yH + 44, r, ` text-anchor="middle" fill="${grey}"`);
    t(4, yH + 44, 'rank', ` fill="${grey}"`);
    const H = yH + 48;
    const naiveSend = new Array(P).fill(0);
    for (const tr of R.tr) { naiveSend[tr.from] += tr.naive; naiveSend[tr.to] += tr.naive; }
    const d2 = A.hidden * 2;
    const head = `<span class="lab">PP</span>` + PIPE_PPS.map((p) => `<button type="button" data-pp="${p}"${p === P ? ' class="on"' : ''}>${p}</button>`).join('') +
      `<span class="lab">schedule</span>` + PIPE_SCHEDS.map(([k, lb]) => `<button type="button" data-s="${k}"${k === this.sched ? ' class="on"' : ''}>${lb}</button>`).join('');
    const held = Math.max(...R.held);
    this._card.innerHTML = `<div class="kpp-head">${head}</div>` +
      `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${out.join('')}</svg>` +
      `<div class="kpp-led"><b>busiest rank (${hot}): ${R.rres} d per token</b>, forward and backward = ${fmtB(R.rres * d2)}` +
      (this.sched === 'chain' ? '' : ` (without the cross-stage cache: ${Math.max(...naiveSend)} d)`) +
      ` · widths in d = one BF16 hidden vector (${fmtB(d2)} per token): the blocks the receiver lacks + the partial sum. ` +
      `<b>Block store: up to ${held} blocks = ${fmtB(held * d2)} per token</b> per microbatch in flight.</div>`;
    for (const bb of this._card.querySelectorAll('[data-pp]')) bb.onclick = () => { if (+bb.dataset.pp !== this.pp) { this.pp = +bb.dataset.pp; this.render(); this.changed(); } };
    for (const bb of this._card.querySelectorAll('[data-s]')) bb.onclick = () => { if (bb.dataset.s !== this.sched) { this.sched = bb.dataset.s; this.render(); this.changed(); } };
  }
  changed() { writeState('k3p:' + (this.id || 'pipe'), { pp: this.pp, s: this.sched }); }
}
customElements.define('kimi-attnres-pipe', KimiAttnresPipe);
