// Memory model: high-watermark per-GPU memory, per pipeline stage.
//
// The watermark moment is 1F1B steady state: min(pp−s, m) forward activation
// sets are live on stage s, gradient accumulators are allocated (backwards
// have started), optimizer state is resident, and comm/FSDP working buffers
// are in flight. Forward-pass temporaries (op workspace) and non-tensor
// overhead (CUDA/NCCL context, allocator fragmentation) are deliberately
// ignored — this is a roofline: it is assumed you didn't get it completely
// correct. Pure module (no DOM) so Node can use it.

import { DSV3, HARDWARE, archOf, stageLayerKinds, stageParams, attnLayerParams, layerNormParams } from './model.js';
import { blockGraph, analyze, resolveMarks, DTYPE_BYTES } from './blockgraph.js';
import { resolveMatmuls } from './recipes.js';

const GB = 2 ** 30;

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
      kind === 'embed' ? a.hidden * a.vocab : kind === 'head' ? a.hidden * a.vocab + a.hidden :
        attnLayerParams(a) + layerNormParams(a) + (kind === 'dense' ? 3 * a.hidden * a.denseInter
          : (a.hidden + 1) * a.routedExperts + a.sharedExperts * expertLayerParams + (a.routedExperts / cfg.ep) * expertLayerParams)));
    const buffers = (layers.includes('moe') ? tokens * a.topk * a.hidden * (DTYPE_BYTES[mm.ffn_gate_up] + 2) : 0)
      + (zero >= 3 ? 3 * maxLayer * 2 : 0);                 // 2 gathered weight buffers + 1 grad bucket

    const entry = { stage: s, inFlight, weights, vocab, grads, vocabGrads, optimizer, vocabOpt, act, buffers };
    entry.total = weights + vocab + grads + vocabGrads + optimizer + vocabOpt + act.total + buffers;
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
  for (const k of ['weights', 'vocab', 'grads', 'vocabGrads', 'optimizer', 'vocabOpt', 'buffers', 'total']) out[k] = e[k] / GB;
  for (const k of Object.keys(e.act)) out.act[k] = e.act[k] / GB;
  return out;
}
