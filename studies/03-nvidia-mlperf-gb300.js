// Study 02 — the NVIDIA MLPerf 6.0 256-GPU submitted configuration (GB300).
// Matches the published NeMo/Megatron-Bridge 26.06 recipe we use as our
// calibration anchor: TP1 · PP2 (×8 virtual) · EP32 · CP1, mbs 1, seq 4096,
// GBS 4096 sequences on 256× GB300 → 6,338 tok/s/GPU = 1,648 model
// TFLOP/s/GPU = 10.30 s/step. Precision: MXFP8 on every GEMM including the
// attention core (q/k/v saved MXFP8; attention OUTPUT saved bf16), rstd/lse
// fp32, fp8 parameter gather (fp8-only working weights), fp32 master + bf16
// grads/moments. NO activation checkpointing — the point of this study.
// Runs in the browser (study page) and in Node:
//   node studies/03-nvidia-mlperf-gb300.js

import { defaultConfig, simulate } from '../src/sim.js';
import { memoryUsage, layerAnalysis, resolveMatmuls } from '../src/memory.js';
import { DSV3 } from '../src/model.js';

export const CFG_OVERRIDES = {
  hardware: 'gb300', gpus: 256, pp: 2, ep: 32, globalBatch: 4096, seqLen: 4096,
  zero: 1,                       // Megatron distributed optimizer: params replicated, optimizer sharded
  recipe: 'nv-mxfp8', recompute: 'none',
  weightBytes: 1 + 1 / 32,       // fp8 parameter gather: fp8-only working weights (+MX scales); master lives in the optimizer
  gradBytes: 2, optBytes: 8,     // bf16 main grads; fp32 master + bf16 Adam moments
  dtype: 'mxfp8',
};

export function computeStudy() {
  const cfg = defaultConfig(CFG_OVERRIDES);
  const mem = memoryUsage(cfg);
  const ana = layerAnalysis('moe', cfg);
  const mm = resolveMatmuls(cfg);
  const bp = (id) => ana.byId[id].outBytes / ana.byId[id].elems; // ascribed bytes/elem
  const w = mem.worst;

  const counts = [
    { name: 'GPUs (pp × dp)', expected: 256, actual: cfg.pp * cfg.dp, note: 'TP1 · PP2(×8 virtual) · EP32 · CP1' },
    { name: 'dp', expected: 128, actual: cfg.dp, note: '256 ÷ pp 2' },
    { name: 'microbatches/step', expected: 32, actual: cfg.microbatches, note: 'GBS 4096 sequences ÷ dp 128, mbs 1' },
    { name: 'worst-stage 1F1B in-flight', expected: 2, actual: Math.min(cfg.pp, cfg.microbatches), info: true, note: 'min(pp, m) — the ×8 virtual-stage interleave raises this somewhat (unmodeled); pp2 is why no-AC is affordable' },
  ];

  // the precision accounting, asserted against the model's ascriptions (bytes/element)
  const precision = [
    { name: 'q/k/v saves (FP8 attention)', expected: 1 + 1 / 32, actual: bp('q_up'), tolPct: 0.1, note: 'MXFP8 E4M3 + E8M0 block scales' },
    { name: 'attention output save', expected: 2, actual: bp('attn'), tolPct: 0.1, note: 'bf16 (o_proj stash kept wide)' },
    { name: 'wgrad stash: dispatched tokens', expected: 1 + 1 / 32, actual: bp('dispatch'), tolPct: 0.1, note: 'MXFP8' },
    { name: 'wgrad stash: gate/up', expected: 1 + 1 / 32, actual: bp('gate_up'), tolPct: 0.1, note: 'MXFP8' },
    { name: 'norm-out saves (no AC)', expected: 1 + 1 / 32, actual: bp('norm1'), tolPct: 0.1, note: 'quantized copy for the following GEMM wgrad; RMSNorm in/out themselves bf16 (fwd temporaries)' },
    { name: 'rstd', expected: 4, actual: ana.byId.norm1.aux.bytes, tolPct: 0.1, note: 'fp32/token' },
    { name: 'lse (softmax statistic)', expected: 4 * DSV3.heads, actual: ana.byId.attn.aux.bytes, tolPct: 0.1, note: 'fp32/head' },
    { name: 'working weights B/param', expected: 1 + 1 / 32, actual: cfg.weightBytes, tolPct: 0.1, note: 'fp8 parameter gather' },
    { name: 'grads B/param', expected: 2, actual: cfg.gradBytes ?? 2, tolPct: 0.1, note: 'bf16 main grads' },
    { name: 'optimizer B/param', expected: 8, actual: cfg.optBytes ?? 8, tolPct: 0.1, note: 'fp32 master + bf16 moments' },
  ];

  const memory = [
    { name: 'weights (fp8-gathered)', expected: w.weights + w.vocab, actual: w.weights + w.vocab, unit: 'GiB', info: true, note: `${w.weights.toFixed(1)} + ${w.vocab.toFixed(1)} embed/head — per rank: pp2 halves the stage; EP32 shards the experts` },
    { name: 'grads / optimizer', expected: w.grads + w.vocabGrads + w.optimizer + w.vocabOpt, actual: w.grads + w.vocabGrads + w.optimizer + w.vocabOpt, unit: 'GiB', info: true, note: `${(w.grads + w.vocabGrads).toFixed(0)} + ${(w.optimizer + w.vocabOpt).toFixed(0)} GiB (incl. embed/head)` },
    { name: 'activations, NO recompute', expected: w.act.total, actual: w.act.total, unit: 'GiB', info: true, note: `mla ${w.act.mla.toFixed(0)} · moe ${w.act.moe.toFixed(0)} · residual ${w.act.residual.toFixed(1)} — only ${Math.min(cfg.pp, cfg.microbatches)} microbatches in flight` },
    { name: 'watermark fits under 288 GB', expected: 288, actual: w.total, unit: 'GiB', lte: true, note: 'no activation checkpointing needed — this is what GB300 headroom buys' },
  ];

  // public throughput anchors (scaling-puzzles SOURCES.md); sim is knowingly optimistic
  const sim = simulate({ ...CFG_OVERRIDES, level: 5 }).stats;
  const timing = [
    { name: 'tok/s/GPU', expected: 6338, actual: Math.round(sim.tokPerSecPerGpu), info: true, note: 'sim optimistic: gemmEff/comm assumptions unvalidated; VP8 interleave unmodeled — the timing studies close this' },
    { name: 'step time (s)', expected: 10.30, actual: +(sim.stepUs / 1e6).toFixed(2), info: true, note: 'published: GBS 4096 → 10.30 s/step' },
    { name: 'model TFLOP/s/GPU', expected: 1648, actual: Math.round(sim.tokPerSecPerGpu * 0.260), info: true, note: 'Megatron convention: 260 GFLOP/token (3× fwd, causal-halved)' },
  ];

  return { cfg, mem, counts, precision, memory, timing };
}

// Node runner
if (typeof process !== 'undefined' && process.argv?.[1]?.includes('03-nvidia-mlperf')) {
  const { counts, precision, memory, timing, mem } = computeStudy();
  let fails = 0;
  for (const [title, rows] of [['counts', counts], ['precision (B/elem)', precision], ['memory', memory], ['timing anchors', timing]]) {
    console.log('\n== ' + title);
    for (const r of rows) {
      const d = r.expected ? (r.actual - r.expected) / r.expected * 100 : 0;
      const ok = r.info || (r.lte ? r.actual <= r.expected : Math.abs(d) <= (r.tolPct ?? 2));
      if (!ok) fails++;
      console.log(`${r.info ? 'INFO' : ok ? 'PASS' : 'FAIL'}  ${r.name}: ref ${r.expected} vs ours ${typeof r.actual === 'number' ? +r.actual.toFixed(3) : r.actual}  ${r.note ?? ''}`);
    }
  }
  console.log(`\nwatermark: ${mem.worst.total.toFixed(1)} GiB / 288 (no AC)`);
  process.exitCode = fails ? 1 : 0;
}
