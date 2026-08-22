// Study 01 — a DSv3-family MoE on 2048 Hopper GPUs: the memory story.
// Post-facing: our model's own inventory (no external reference tables here).
// Runs in the browser (study page) and in Node: node studies/02-hopper-memory.js

import { defaultConfig } from '../src/sim.js';
import { memoryUsage } from '../src/memory.js';

// A 288-routed-expert DSv3-family variant: 62 layers, vocab 128000, no MTP.
export const VARIANT_ARCH = {
  layers: 62, denseLayers: 0, denseInter: 18432, hidden: 7168, vocab: 128000,
  heads: 128, qkNope: 128, qkRope: 64, vHead: 128, qRank: 1536, kvRank: 512,
  moeInter: 2048, routedExperts: 288, topk: 8, sharedExperts: 1,
};

// 2048 H100-class GPUs = FSDP-replicas 4 × PP 8 × EP 64 (our decomposition:
// gpus=2048, pp=8, ep=64 → dp=256, dp/ep=4), seq 4096, mbs 1, 64 microbatches.
// FP8 GEMMs (weights ×2 for the transpose, +4/128 fp32 scales), fp32 gradient
// accumulators, 3-state AMSGrad optimizer sharded over all 2048 GPUs,
// attn-replay recompute (replay all of attention from x0 in backward; save the
// post-norm output + fp8 expert inputs and gate/up; recompute swiglu-out),
// DualPipe-style in-flight = (2·PP+1)/2 = 8.5 microbatches/rank.
export const CFG_OVERRIDES = {
  hardware: 'h100', arch: VARIANT_ARCH,
  gpus: 2048, pp: 8, ep: 64, mbs: 1, seqLen: 4096, microbatches: 64,
  nodeLimit: 4,
  zero: 1, recipe: 'dsv3-fp8', recompute: 'attn-replay',
  gradBytes: 4,                       // fp32 gradient accumulators
  optBytes: 12,                       // AMSGrad: ema/emasq/max-emasq, fp32 ×3
  optShard: 'world',                  // optimizer sharded over all 2048 GPUs
  weightBytes: 2 * (1 + 4 / 128),     // fp8 ×2 (transpose) × tile-scale overhead
  inflight: 8.5,                      // DualPipe two-chunk layout: 17 mb × 4 layers/chunk ≡ 8.5 × 8 layers/rank
};

// Our per-rank inventory at this configuration (mid rank + the two vocab ranks).
export function inventory() {
  const cfg = defaultConfig(CFG_OVERRIDES);
  const mem = memoryUsage(cfg);
  const row = (s) => {
    const w = mem.perStage[s];
    return {
      stage: s, weights: w.weights, vocab: w.vocab, grads: w.grads + w.vocabGrads,
      optimizer: w.optimizer + w.vocabOpt, actMla: w.act.mla, actMoe: w.act.moe,
      actResidual: w.act.residual, buffers: w.buffers, total: w.total,
    };
  };
  return { cfg, mem, mid: row(1), first: row(0), last: row(cfg.pp - 1), capacityGB: mem.capacityGB };
}

// Node runner: print the inventory
if (typeof process !== 'undefined' && process.argv?.[1]?.includes('02-hopper-memory')) {
  const { mid, first, last, capacityGB } = inventory();
  for (const [name, r] of [['mid rank (pp1)', mid], ['first rank (pp0)', first], ['last rank (pp7)', last]]) {
    console.log(`${name}: total ${r.total.toFixed(1)} / ${capacityGB} GiB ` +
      `(w ${r.weights.toFixed(1)} + vocab ${r.vocab.toFixed(1)} + g ${r.grads.toFixed(1)} + o ${r.optimizer.toFixed(1)} ` +
      `+ act ${(r.actMla + r.actMoe + r.actResidual).toFixed(1)} + buf ${r.buffers.toFixed(1)})`);
  }
  process.exitCode = 0;
}
