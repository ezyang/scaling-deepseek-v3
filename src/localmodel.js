// The local-lens model: what ONE GPU holds under the essay's fiat
// parallelism — PP stage layout, virtual-stage placement, 1F1B/DualPipeV
// in-flight admission, per-component byte rates, and the activation buckets
// the fit charts price from. Pure module (no DOM): Node consumers (sanity,
// goldens) import it directly; the renderers in viewer.js build on it.
import { DSV3 } from './model.js';
import { PARAMS } from './params.js';
import { resolveMatmuls } from './memory.js';
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
  { label: 'gate, up (routed+sh)', ids: ['gate_up'] },
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
export const CFG_DEFAULTS = { world: 2048, pp: 1, ep: 1, zero: 0, sched: '1f1b' };   // vpp/fold are derived from pp
// virtual-stage placement: with VPP = vpp chunks per rank the chain is
// vpp·pp virtual stages deep; 'wrap' places stage v on rank v mod pp
// (Megatron interleaving), 'reflect' bounces each pass (ZB-V / DualPipeV:
// rank 0 → pp−1 → 0 → …), so with even vpp the chain both starts AND ends
// on rank 0. DualPipeV ≡ vpp 2 + reflect.
export const vstagesOf = (r, pp, vpp = 1, fold = 'reflect') =>
  Array.from({ length: vpp }, (_, c) =>
    c * pp + (fold === 'reflect' && c % 2 ? pp - 1 - r : r));
export const ppStage = (s, pp = LOCAL_PAR.pp, vpp = 1, fold = 'reflect') => {
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
export const inflightOf = (sched, s, pp, vpp = 1, fold = 'reflect') => {
  if (sched === 'one') return 1;
  const D = vpp * pp, s2 = Math.min(s, pp - 1);
  return vstagesOf(s2, pp, vpp, fold).reduce((t, v) => t + (D - v), 0) / vpp;
};

// the PP stage holding the most resident bytes under the local model (all
// components on, vocab counted on the end stages, activations under the
// schedule) — the default stage to show: the fully loaded rank.
export const peakStage = (pp, ep, zero, world = LOCAL_PAR.world, sched = '1f1b', vpp = 1, fold = 'reflect') => {
  const dp = world / pp;
  const bpp = (cls) => BYTE_COMPS.reduce((t, c) =>
    t + (zero >= c.zthresh ? c.bpp / (cls === 'e' ? dp / ep : dp) : c.bpp), 0);
  const moeExp = PARAMS.expert * DSV3.routedExperts;
  const dB = PARAMS.denseBlock * bpp('d');
  const mB = (PARAMS.moeBlock - moeExp) * bpp('d') + (moeExp / ep) * bpp('e');
  let best = 0, bestV = -1;
  for (let s2 = 0; s2 < pp; s2++) {
    const g = ppStage(s2, pp, vpp, fold);
    const v = g.dense * dB + g.moe * mB
      + (((g.emb ? 1 : 0) + (g.head ? 1 : 0)) * PARAMS.embed * bpp('d'))
      + (g.dense * actLayerBytes('dense') + g.moe * actLayerBytes('moe')) * inflightOf(sched, s2, pp, vpp, fold);
    if (v > bestV) { bestV = v; best = s2; }
  }
  return best;
};
