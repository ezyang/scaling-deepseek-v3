// DeepSeek-V4.1-Flash: the architecture constants and the exact parameter
// tally, derived once from tensor shapes. Pure data (no DOM) so
// scripts/sanity.mjs can import it; src/dsv41.js draws from it.
//
// Sources (mirrored in ~/Dev/scaling-dsv3-refs/dsv41-flash/, see SOURCES.md):
// the checkpoint's config.json + inference/model.py (shapes, wiring) and the
// safetensors headers of all 48 shards (every tensor's dtype + shape). The
// totals reconcile EXACTLY with the headers: 551,566,180,464 backbone
// ("552B") and 196,928,504,320 Engram ("196B"); the per-token active counts
// land on the advertised 8B (prefill) / 16B (decode).

export const DSV41 = {
  layers: 40, encLayers: 20, decLayers: 20,
  hidden: 5120, vocab: 129280,
  hcMult: 4,                                   // mHC: the residual stream is 4 parallel copies
  heads: 64, headDim: 512, ropeDim: 64,        // one 512-d KV latent shared by all heads (k = v)
  qRank: 1280, oGroups: 8, oRank: 1024,        // q low-rank; grouped low-rank out-proj
  window: 128,                                 // SWA on every layer (FP8 cache)
  idxHeads: 32, idxDim: 128, topk: 512,        // the indexer
  candBlocks: 2048, candBlock: 8,              // hierarchical candidate pool (decoder): 16,384 positions
  routedExperts: 384, sharedExperts: 1, ffnTopk: 6, moeInter: 2304,
  // per-layer attention kinds (config compress_ratios / kv_source / index_source):
  //   0,1 = SWA only; 2..19 encoder CSA2 m=2 (Full at 2, 8, 14; rest Reuse);
  //   20..39 decoder CSA2 m=1 (Full at 20; Reindex at 24, 28, 32, 36; rest Reuse)
  kinds: { swa: 2, full2: 3, full1: 1, reindex: 4, reuse: 30 },
  kvSource: [2, 8, 14, 20], indexSource: [2, 8, 14, 20, 24, 28, 32, 36],
  engramLayers: [1, 14],
  engramRows: [384006168, 384016682], engramDim: 256, engramHeads: 8, engramOrders: 3,   // n-gram orders {2,3,4}
  // DSpark drafter (mtp.0–2) and the vision tower are counted separately
  dspark: { blocks: 3, experts: 128, topk: 3, blockSize: 5, markovRank: 256, targets: [37, 38, 39] },
  vision: { layers: 32, dim: 1024, heads: 16, inter: 2816, patch: 14, unshuffle: 3 },
  kv: { mainBytes: 512 / 2 + 512 / 16, idxBytes: 128 / 2 + 128 / 32 },   // FP4 + e4m3/16 · FP4 + e8m0/32
};

const A = DSV41;
const d = A.hidden;
const PJ = A.heads * A.headDim;   // 32768

export const VPARAMS = (() => {
  // attention core, every layer (tensor names from the checkpoint)
  const qDown = d * A.qRank, qNorm = A.qRank;                       // wq_a, q_norm
  const qUp = A.qRank * PJ;                                         // wq_b
  const kv = d * A.headDim, kvNorm = A.headDim;                     // wkv, kv_norm (the SWA latent)
  const oA = PJ * A.oRank;                                          // wo_a: 8 groups × (4096 → 1024)
  const oB = A.oGroups * A.oRank * d;                               // wo_b: 8192 → 5120
  const sink = A.heads;                                             // attn_sink
  const attnCore = qDown + qNorm + qUp + kv + kvNorm + oA + oB + sink;
  // CSA2 extras by mode
  const compKv = d * A.headDim, compGate = d * A.headDim, compNorm = A.headDim;   // compressor.wkv / .wgate / .norm
  const idxK = A.headDim * A.idxDim, idxKNorm = A.idxDim;           // indexer.wk / .k_norm (Full only)
  const idxQ = A.qRank * A.idxHeads * A.idxDim;                     // indexer.wq_b (Full + Reindex)
  const idxW = d * A.idxHeads;                                      // indexer.weights_proj
  const indexer = idxQ + idxW;
  const full2 = compKv + compGate + compNorm + idxK + idxKNorm + indexer;   // encoder Full (m=2: gated pooling)
  const full1 = compKv + compNorm + idxK + idxKNorm + indexer;               // decoder Full (m=1: plain projection)
  const reindex = indexer;
  // MoE
  const expert = 3 * d * A.moeInter;                                // w1, w2, w3 (routed: FP4; shared: FP8)
  const routed = A.routedExperts * expert, shared = A.sharedExperts * expert;
  const routerWeight = A.routedExperts * d, routerBias = 2 * A.routedExperts;   // gate.weight + bias + bias_vl
  const activeRouted = A.ffnTopk * expert;
  const moeFfn = routerWeight + routerBias + routed + shared;
  const activeMoeFfn = routerWeight + routerBias + activeRouted + shared;
  // mHC coefficient predictors (fp32): per sub-layer hc_fn [24, 4·5120] + base [24] + scale [3]
  const mix = (2 + A.hcMult) * A.hcMult;
  const hcSub = mix * A.hcMult * d + mix + 3;
  const hc = 2 * hcSub;
  const norms = 2 * d;                                              // attn_norm, ffn_norm
  const misc = hc + norms;
  const blockBase = attnCore + moeFfn + misc;
  const block = { swa: blockBase, full2: blockBase + full2, full1: blockBase + full1, reindex: blockBase + reindex, reuse: blockBase };
  const activeBase = attnCore + activeMoeFfn + misc;
  const activeBlock = { swa: activeBase, full2: activeBase + full2, full1: activeBase + full1, reindex: activeBase + reindex, reuse: activeBase };
  // Engram (per module): FP8 table rows × 256, wkv (3 orders × 8 heads × 256 → (hc+1)·d), q/k gate weights
  const engramWkv = A.engramOrders * A.engramHeads * A.engramDim * (A.hcMult + 1) * d;
  const engramGate = 2 * A.hcMult * d;
  const engramTables = A.engramRows.map(r => r * A.engramDim);
  const engramProj = engramWkv + engramGate;
  const engram = engramTables.map(t => t + engramProj);
  const engramTotal = engram.reduce((s, x) => s + x, 0);
  const embed = A.vocab * d, finalNorm = d;
  const layersTotal = Object.entries(A.kinds).reduce((s, [k, n]) => s + n * block[k], 0);
  const backbone = embed + layersTotal + finalNorm + embed;
  // "active / token" conventions (DeepSeek's: the embedding lookup is not
  // counted, the head is; Engram TABLE lookups are lookups, its projections
  // count). Decode runs all 40 layers + head. Prefill runs the 20 encoder
  // layers only (CED): the decoder's global KV is projected from the encoder
  // output by layer 20's compressor + index-K projection, so those count too;
  // the decoder's indexer Qs and the head do not run.
  const engramActive = A.engramLayers.length * engramProj;
  const activeDecode = Object.entries(A.kinds).reduce((s, [k, n]) => s + n * activeBlock[k], 0)
    + finalNorm + embed + engramActive;
  const decKvProj = compKv + compNorm + idxK + idxKNorm;
  const activePrefill = A.kinds.swa * activeBlock.swa + A.kinds.full2 * activeBlock.full2
    + (A.encLayers - A.kinds.swa - A.kinds.full2) * activeBlock.reuse + decKvProj + engramActive;
  // DSpark + vision (excluded from the tally; footnoted)
  const dsExpert = 3 * d * A.moeInter;
  const dsMoe = routerWeight / A.routedExperts * A.dspark.experts + 2 * A.dspark.experts + A.dspark.experts * dsExpert + dsExpert;
  const dsBlock = attnCore + dsMoe + misc;
  const dspark = A.dspark.blocks * dsBlock + d * A.dspark.targets.length * d + d   // main_proj + main_norm
    + d + 2 * A.vocab * A.dspark.markovRank + (d + A.dspark.markovRank);       // norm, markov embed+head, confidence
  const V = A.vision;
  const vBlock = 3 * V.dim * V.dim + 3 * V.dim + V.dim * V.dim + V.dim + 2 * V.dim * V.inter + V.inter * V.dim + 2 * V.dim;
  const vision = 3 * V.patch * V.patch * V.dim + V.dim + V.layers * vBlock + V.dim
    + (V.dim * V.unshuffle ** 2 * d + d) + (d * d + d) + 3 * d;             // aligner + image_start/end/newline
  return {
    qDown, qNorm, qUp, kv, kvNorm, oA, oB, sink, attnCore,
    compKv, compGate, compNorm, idxK, idxKNorm, idxQ, idxW, indexer, full2, full1, reindex,
    expert, routed, shared, routerWeight, routerBias, activeRouted, moeFfn, activeMoeFfn,
    hcSub, hc, norms, misc, block, activeBlock,
    engramTables, engramWkv, engramGate, engramProj, engram, engramTotal,
    embed, finalNorm, layersTotal, backbone, activeDecode, activePrefill, dspark, vision,
  };
})();

// KV cache bytes per token (the headline 890): main KV + indexer K, both
// FP4, at 3 encoder sources (m=2 → ½ entry/token) + 1 decoder source (m=1)
export const KV_PER_TOKEN = (() => {
  const entries = A.kinds.full2 / 2 + A.kinds.full1;
  return { entries, main: A.kv.mainBytes * entries, idx: A.kv.idxBytes * entries, total: (A.kv.mainBytes + A.kv.idxBytes) * entries,
    swaPerLayer: A.headDim + A.headDim / 32 };   // FP8 + e8m0/32, ×128 window × 40 layers, bounded
})();
