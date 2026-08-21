// Memory model: high-watermark per-GPU memory, per pipeline stage.
//
// The watermark moment is 1F1B steady state: min(pp−s, m) forward activation
// sets are live on stage s, gradient accumulators are allocated (backwards
// have started), optimizer state is resident, and comm/FSDP working buffers
// are in flight. Forward-pass temporaries (op workspace) are deliberately
// ignored — they are not set in stone. Pure module (no DOM) so Node can use it.

import { DSV3, HARDWARE, stageLayerKinds, stageParams, attnLayerParams } from './model.js';
import { blockGraph, analyze, resolveMarks, DTYPE_BYTES } from './blockgraph.js';

const GB = 2 ** 30;
export { DTYPE_BYTES };

// The matmuls of one DSv3 layer (plus the head). Precision is chosen per
// matmul; a matmul's dtype governs both its weight copy and the stashed
// input activations its backward needs.
export const MATMULS = [
  { id: 'qkv_down', label: 'q/kv down-proj', dims: '7168 → 1536 + 576',
    dimsNote: 'hidden 7168 → q latent 1536, plus kv latent 512 + shared rope key 64' },
  { id: 'q_up', label: 'q up-proj', dims: '1536 → 128×192',
    dimsNote: 'q latent 1536 → 128 heads × (128 nope + 64 rope) = 24576' },
  { id: 'kv_up', label: 'kv up-proj', dims: '512 → 128×(128+128)',
    dimsNote: 'kv latent 512 → 128 heads × (128 k-nope + 128 v) = 32768' },
  { id: 'attn', label: 'attention', dims: 'softmax(QKᵀ)V · causal',
    dimsNote: 'softmax(QKᵀ)V over causal context ≤ 4096 (average ≈ seq/2)' },
  { id: 'o_proj', label: 'attn out-proj', dims: '128×128 → 7168',
    dimsNote: '128 heads × 128 v-dim = 16384 → hidden 7168' },
  { id: 'router', label: 'router', dims: '7168 → 256',
    dimsNote: 'hidden 7168 → 256 expert logits (top-8 kept, fp32 gating)' },
  { id: 'ffn_gate_up', label: 'ffn gate/up', dims: '7168 → 2×2048',
    dimsNote: 'hidden 7168 → gate 2048 + up 2048, per expert (8 routed + 1 shared)' },
  { id: 'ffn_down', label: 'ffn down', dims: '2048 → 7168',
    dimsNote: 'inter 2048 → hidden 7168, per expert (8 routed + 1 shared)' },
  { id: 'lm_head', label: 'lm head', dims: '7168 → 129280',
    dimsNote: 'hidden 7168 → 129280 vocabulary logits' },
];

// Presets. Unlisted matmuls stay bf16 (attention core, router, head — the
// things every recipe keeps in high precision).
export const RECIPES = {
  'bf16': {},
  // DeepSeek-V3 paper recipe: linears in fine-grained FP8 (1x128 tile scales —
  // same bytes as MX); attention ops and head high-precision; the router runs
  // fp32 in production; the attn-out linear's stash kept wide (E5M6 in
  // the paper — we use bf16).
  'dsv3-fp8': { qkv_down: 'mxfp8', q_up: 'mxfp8', kv_up: 'mxfp8', router: 'fp32', ffn_gate_up: 'mxfp8', ffn_down: 'mxfp8' },
  // NVIDIA NeMo/Megatron-Bridge (MLPerf 6.0 submission) recipe: MXFP8 for every
  // GEMM (32-element MX blocks, UE8M0 scales, via TE) INCLUDING the attention
  // core (Blackwell FP8 attention: q/k/v saved MXFP8). The attention OUTPUT is
  // saved bf16 (o_proj stash wide); router/head bf16.
  'nv-mxfp8': { qkv_down: 'mxfp8', q_up: 'mxfp8', kv_up: 'mxfp8', attn: 'mxfp8', o_proj: 'bf16', ffn_gate_up: 'mxfp8', ffn_down: 'mxfp8' },
};

export function resolveMatmuls(cfg) {
  const recipe = cfg.recipe ?? (cfg.dtype === 'mxfp8' ? 'nv-mxfp8' : 'bf16');
  return {
    ...Object.fromEntries(MATMULS.map(m => [m.id, 'bf16'])),
    ...RECIPES[recipe],
    ...(cfg.matmuls ?? {}),
  };
}

// Stashed activation bytes per token for one layer, broken down by component.
// Derived from the block op-graph (src/blockgraph.js): memory = the outputs of
// save-marked ops that backward actually needs; `recompute` names a marking
// preset (none / dsv3 / selective / full) and `saved` holds per-op overrides.
export function actBreakdownPerToken(kind, a, recompute, mm, saved, seqLen = 4096, transposedStash = false) {
  if (kind === 'embed') return { residual: 2 * a.hidden };
  if (kind === 'head') return { logits: 6 * a.vocab }; // bf16 logits + fp32 softmax
  const marks = resolveMarks({ recompute, saved });
  return analyze(blockGraph(kind, a, mm, seqLen), marks, transposedStash).buckets;
}

// Full graph analysis for one layer kind (memory + replay overhead), for the
// schematic and the trace sim's recompute charge.
export function layerAnalysis(kind, cfg) {
  const a = archOf(cfg);
  return analyze(blockGraph(kind, a, resolveMatmuls(cfg), cfg.seqLen), resolveMarks(cfg), cfg.transposedStash ?? false);
}

// Total params of the (possibly overridden) architecture.
export function totalParams(a, pp, ep) {
  let t = 0;
  for (let s = 0; s < pp; s++) {
    const p = stageParams(a, pp, ep, s);
    t += p.dense + p.expert * ep;
  }
  return t;
}

export function archOf(cfg) { return cfg.arch ? { ...DSV3, ...cfg.arch } : DSV3; }

export function memoryUsage(cfg) {
  const a = archOf(cfg), hw = HARDWARE[cfg.hardware];
  const mm = resolveMatmuls(cfg);
  const zero = cfg.zero ?? 3;
  const recompute = cfg.recompute ?? 'selective';
  const dpe = Math.max(1, cfg.dp / cfg.ep);                 // FSDP replicas of the expert shards
  const tokens = cfg.mbs * cfg.seqLen;
  const gradB = cfg.gradBytes ?? 2;                         // bf16 accumulators (DSv3 paper: fp32 -> 4)
  const optB = cfg.optBytes ?? 8;                           // fp32 master + bf16 moments (AMSGrad fp32 x3 -> 12)
  const expertLayerParams = 3 * a.hidden * a.moeInter;
  // low-precision weight copies (kept resident alongside bf16 working weights)
  const ffnCopy = mm.ffn_gate_up !== 'bf16' ? DTYPE_BYTES[mm.ffn_gate_up] : 0;
  const attnCopy = mm.q_up !== 'bf16' ? DTYPE_BYTES[mm.q_up] : 0;

  const perStage = [];
  for (let s = 0; s < cfg.pp; s++) {
    const p = stageParams(a, cfg.pp, cfg.ep, s);
    // vocab matrices (embedding on stage 0, lm head on the last) reported as
    // their own component — same byte convention as the dense weights
    const vocabP = ((s === 0 ? 1 : 0) + (s === cfg.pp - 1 ? 1 : 0)) * a.hidden * a.vocab;
    const shDense = (zero >= 3 ? (p.dense - vocabP) / cfg.dp : p.dense - vocabP);
    const shVocab = zero >= 3 ? vocabP / cfg.dp : vocabP;
    const shExpert = zero >= 3 ? p.expert / dpe : p.expert;
    // bytes/param: explicit override, else bf16 working weights + resident low-precision copy
    const denseBytes = cfg.weightBytes ?? (2 + attnCopy);
    const expertBytes = cfg.weightBytes ?? (2 + ffnCopy);
    const weights = shDense * denseBytes + shExpert * expertBytes;
    const vocab = shVocab * denseBytes;
    const grads = (shDense + shExpert) * gradB;             // accumulators live at the watermark
    const vocabGrads = shVocab * gradB;
    const denseNoV = p.dense - vocabP;
    // world-sharded optimizer slices aren't stage-local, so no vocab attribution there
    const optimizer = cfg.optShard === 'world'
      ? totalParams(a, cfg.pp, cfg.ep) / cfg.gpus * optB    // ZeRO-1 over the whole world
      : (zero >= 1 ? denseNoV / cfg.dp + p.expert / dpe : denseNoV + p.expert) * optB;
    const vocabOpt = cfg.optShard === 'world' ? 0 : (zero >= 1 ? vocabP / cfg.dp : vocabP) * optB;

    const layers = stageLayerKinds(a, cfg.pp, s).map(l => l.kind);
    if (s === 0) layers.unshift('embed');
    if (s === cfg.pp - 1) layers.push('head');
    const inFlight = cfg.inflight ?? Math.min(cfg.pp - s, cfg.microbatches);
    const act = { mla: 0, moe: 0, residual: 0, logits: 0 };
    for (const kind of layers) {
      const b = actBreakdownPerToken(kind, a, recompute, mm, cfg.saved, cfg.seqLen, cfg.transposedStash);
      // the head's logits live for ~1 microbatch (bwd is immediate on the last stage)
      const n = kind === 'head' ? 1 : inFlight;
      for (const k of Object.keys(b)) act[k] += b[k] * tokens * n;
    }
    act.total = act.mla + act.moe + act.residual + act.logits;

    // transient working set coincident with the watermark:
    // a2a send+recv buffers, FSDP's gathered layer (current + prefetch),
    // and one layer of unsharded gradients awaiting reduce-scatter.
    const maxLayer = Math.max(...layers.map(kind =>
      kind === 'embed' || kind === 'head' ? a.hidden * a.vocab :
        attnLayerParams(a) + (kind === 'dense' ? 3 * a.hidden * a.denseInter
          : a.hidden * a.routedExperts + a.sharedExperts * expertLayerParams + (a.routedExperts / cfg.ep) * expertLayerParams)));
    const buffers = (layers.includes('moe') ? tokens * a.topk * a.hidden * (DTYPE_BYTES[mm.ffn_gate_up] + 2) : 0)
      + (zero >= 3 ? 3 * maxLayer * 2 : 0);                 // 2 gathered weight buffers + 1 grad bucket

    const overhead = (cfg.overheadGB ?? 6) * GB;            // CUDA/NCCL context, allocator slack, misc
    const entry = { stage: s, inFlight, weights, vocab, grads, vocabGrads, optimizer, vocabOpt, act, buffers, overhead };
    entry.total = weights + vocab + grads + vocabGrads + optimizer + vocabOpt + act.total + buffers + overhead;
    perStage.push(toGB(entry));
  }
  const worst = perStage.reduce((x, y) => (y.total > x.total ? y : x));
  return {
    perStage, worst, capacityGB: hw.memGB, fits: worst.total <= hw.memGB,
    matmuls: mm,
    moment: '1F1B steady state (fwd stashes at max, grad accumulators live)',
  };
}

function toGB(e) {
  const out = { stage: e.stage, inFlight: e.inFlight, act: {} };
  for (const k of ['weights', 'vocab', 'grads', 'vocabGrads', 'optimizer', 'vocabOpt', 'buffers', 'overhead', 'total']) out[k] = e[k] / GB;
  for (const k of Object.keys(e.act)) out.act[k] = e.act[k] / GB;
  return out;
}
