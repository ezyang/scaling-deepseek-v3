// The CODA paper's own benchmarks (arXiv:2605.19269v2, §5 and Fig. 2–4: a single
// H100, BF16, torch.compile'd PyTorch + cuBLAS as the baseline), priced the way
// studies/coda.html prices DeepSeek's launches — every kernel at max(FLOP ÷ the
// measured BF16 GEMM peak, bytes ÷ HBM), each operand counted once — so the
// predicted speedups can be held against the measured bars (src/codapaper.js,
// digitized from the paper's figures by refs/coda/digitize-bars.py). This is the
// calibration of the method: the paper measured what the DeepSeek page can only
// model.
//
// What each benchmark runs is read off the paper's code at the tag it was
// measured with (refs/coda/v0.1/: kernels/benchmarks/block.py + gemm.py,
// models/ops.py for CODA, models/ops2.py + kernels/refs/gpt2.py for the
// baselines). Baseline = a torch.compile(fullgraph) reference: cuBLAS for the
// mm, one fused Triton kernel for everything after it, so the GEMM output is
// written once and read back once. CODA = the paper's GEMM-plus-epilogue
// kernels (Kernel 4–10 of its Appendix) plus the auxiliary reductions over tile
// partials and, in the layer backward, a standalone RoPE-backward kernel (the
// paper fuses RoPE's backward into nothing: attn_bwd_rope_patch). Kernel
// benchmarks: M = N = K = m ∈ {4096, 8192}, RoPE N = 3m, cross-entropy N =
// 32,768, reduction tiles of 128. Blocks: 16,384 tokens (2 × 8,192), hidden d ∈
// {2048, 4096, 8192}, FFN 8/3·d rounded up to 256, QKV 3d, vocabulary 32,768.
// Node-importable; tests/coda.js and the goldens pin it.
import { RATES, EPI } from './kerntrace.js';
import { PAPER } from './codapaper.js';

export const PI = RATES.bf16, BW = RATES.hbm;     // 03's measured BF16 GEMM peak; the H100 and H800 share the die and the HBM
const bf = (n) => 2 * n, f32 = (n) => 4 * n;
const BLK = 128;                                   // the paper's reduction tile (block_size) for partials
export const TOKENS = 16384, VOCAB = 32768;
export const ffn = (d) => Math.ceil(d * 8 / 3 / 256) * 256;    // 5,632 · 11,008 · 22,016

// kernels: a GEMM reads A (M×K) and B (K×N), writes its output (M×outN) and whatever its epilogue adds (`extra`)
const G = (name, M, N, K, extra = 0, outN = N) => ({ name, kind: 'gemm', f: 2 * M * N * K, b: bf(M * K) + bf(K * N) + bf(M * outN) + extra, M, N, K });
const OP = (name, b) => ({ name, kind: 'op', f: 0, b });           // a memory-bound kernel: its operands, once
const AUX = (name, b) => ({ name, kind: 'aux', f: 0, b });         // a reduction over tile partials
export const time = (k) => Math.max(k.f / PI, k.b / BW);

// ---- the benchmarks: baseline launches, CODA launches, the raw GEMM ----------
// kernel level (Fig. 2 and 3 of the paper), size m
const KERNELS = {
  'GEMM-Residual-PartialRMS-GEMM': (m) => ({ fig: 'fig-2', K: m, base: [G('mm', m, m, m), OP('+ residual, partial RMS, ⊙ γ (fused)', 4 * bf(m * m) + f32(m * m / BLK))],
    coda: [G('Kernel 4', m, m, m, 2 * bf(m * m) + f32(m * m / BLK))], gemm: [G('mm', m, m, m)] }),
  'GEMM-RMS': (m) => ({ fig: 'fig-2', K: m, base: [G('mm', m, m, m), OP('× r', 2 * bf(m * m))], coda: [G('Kernel 5', m, m, m, f32(m))], gemm: [G('mm', m, m, m)] }),
  'GEMM-RMS-SwiGLU': (m) => ({ fig: 'fig-2', K: m, base: [G('mm', m, m, m), OP('× r, SwiGLU', 2 * bf(m * m) + bf(m * m / 2))],
    coda: [G('Kernel 6', m, m, m, f32(m) + bf(m * m / 2))], gemm: [G('mm', m, m, m)], sfu: true }),
  'GEMM-RMS-RoPE': (m) => ({ fig: 'fig-2', K: m, base: [G('mm', m, 3 * m, m), OP('× r, RoPE (cos/sin table M × 3m)', 4 * bf(3 * m * m))],
    coda: [G('Kernel 7', m, 3 * m, m, f32(m) + 2 * bf(3 * m * m))], gemm: [G('mm', m, 3 * m, m)] }),
  'GEMM-RMS-CrossEntropy': (m) => ({ fig: 'fig-2', K: m, base: [G('mm', m, VOCAB, m), OP('× r, target logit, partial LSE', 2 * bf(m * VOCAB) + f32(m * VOCAB / BLK))],
    coda: [G('Kernel 8', m, VOCAB, m, f32(m) + f32(m * VOCAB / BLK) + bf(m))], gemm: [G('mm', m, VOCAB, m)], sfu: true }),
  'GEMM-Residual-PartialRMS-GEMM (Backward)': (m) => ({ fig: 'fig-2', K: m, base: [G('mm', m, m, m), OP('RMSNorm backward + residual, partial dγ (fused)', 5 * bf(m * m) + f32(m * m / BLK))],
    coda: [G('Kernel 9', m, m, m, 3 * bf(m * m) + f32(m * m / BLK) + 2 * f32(m))], gemm: [G('mm', m, m, m)] }),
  'GEMM-RMS-SwiGLU (Backward)': (m) => ({ fig: 'fig-2', K: m, base: [G('mm', m, m, m), OP('SwiGLU backward, partial z·dz (fused)', 6 * bf(m * m) + f32(m * m / BLK))],
    coda: [G('Kernel 10', m, m, m, bf(2 * m * m) + bf(m * m) + f32(m * m / BLK), 2 * m)], gemm: [G('mm', m, m, m)], sfu: true }),
  'GEMM + SwiGLU': (m) => ({ fig: 'fig-1', K: m, base: [G('mm', m, m, m), OP('SwiGLU', bf(m * m) + bf(m * m / 2))], coda: [G('Kernel 2', m, m, m, bf(m * m / 2))], gemm: [G('mm', m, m, m)], sfu: true }),
  'GEMM + RoPE': (m) => ({ fig: 'fig-1', K: m, base: [G('mm', m, 3 * m, m), OP('RoPE (cos/sin table M × 3m)', 3 * bf(3 * m * m))], coda: [G('Kernel 1', m, 3 * m, m, 2 * bf(3 * m * m))], gemm: [G('mm', m, 3 * m, m)] }),
  'GEMM + Cross Entropy': (m) => ({ fig: 'fig-1', K: m, base: [G('mm', m, VOCAB, m), OP('target logit, partial LSE', bf(m * VOCAB) + f32(m * VOCAB / BLK))],
    coda: [G('Kernel 3', m, VOCAB, m, f32(m * VOCAB / BLK) + bf(m))], gemm: [G('mm', m, VOCAB, m)], sfu: true }),
};

// block level (Fig. 1 and 4 of the paper), hidden size d, M = 16,384 tokens
const norm = (M, d) => OP('residual add + RMSNorm (fused)', 4 * bf(M * d) + f32(M));
const k4 = (name, M, d, Kd) => G(`Kernel 4 (${name})`, M, d, Kd, 2 * bf(M * d) + f32(M * d / BLK));
const rstd = (M, d) => AUX('r from partials', f32(M * d / BLK) + f32(M));
const k9 = (name, M, d, Kd) => G(`Kernel 9 (${name})`, M, d, Kd, 3 * bf(M * d) + f32(M * d / BLK) + 2 * f32(M));
const dwn = (M, d) => AUX('dγ from partials', f32(d * M / BLK) + f32(d));
const BLOCKS = {
  'GEMM-Residual-RMS-GEMM': (d) => { const M = TOKENS, I = ffn(d); return { fig: 'fig-0',
    base: [G('out-proj', M, d, d), norm(M, d), G('gate/up', M, 2 * I, d)],
    coda: [k4('out-proj', M, d, d), rstd(M, d), G('Kernel 5 (gate/up)', M, 2 * I, d, f32(M))] }; },
  'GEMM-Residual-RMS-GEMM-SwiGLU': (d) => { const M = TOKENS, I = ffn(d); return { fig: 'blocks',
    base: [G('out-proj', M, d, d), norm(M, d), G('gate/up', M, 2 * I, d), OP('SwiGLU', bf(2 * M * I) + bf(M * I))],
    coda: [k4('out-proj', M, d, d), rstd(M, d), G('Kernel 6 (gate/up)', M, 2 * I, d, f32(M) + bf(M * I))] }; },
  'GEMM-Residual-RMS-GEMM-RoPE': (d) => { const M = TOKENS, I = ffn(d); return { fig: 'blocks',
    base: [G('down', M, d, I), norm(M, d), G('qkv', M, 3 * d, d), OP('RoPE', 2 * bf(3 * M * d))],
    coda: [k4('down', M, d, I), rstd(M, d), G('Kernel 7 (qkv)', M, 3 * d, d, f32(M) + 2 * bf(3 * M * d))] }; },
  'GEMM-Residual-RMS-GEMM-CrossEntropy': (d) => { const M = TOKENS, I = ffn(d); return { fig: 'blocks',
    base: [G('down', M, d, I), norm(M, d), G('lm head', M, VOCAB, d), OP('cross-entropy', bf(M * VOCAB) + f32(M))],
    coda: [k4('down', M, d, I), rstd(M, d), G('Kernel 8 (lm head)', M, VOCAB, d, f32(M) + f32(M * VOCAB / BLK) + bf(M)), AUX('LSE + loss from partials', f32(M * VOCAB / BLK) + f32(M))] }; },
  'Layer Forward': (d) => ({ fig: 'blocks', base: [...BLOCKS['GEMM-Residual-RMS-GEMM-SwiGLU'](d).base, ...BLOCKS['GEMM-Residual-RMS-GEMM-RoPE'](d).base],
    coda: [...BLOCKS['GEMM-Residual-RMS-GEMM-SwiGLU'](d).coda, ...BLOCKS['GEMM-Residual-RMS-GEMM-RoPE'](d).coda] }),
  'Layer Backward': (d) => { const M = TOKENS, I = ffn(d); return { fig: 'blocks',
    base: [OP('RoPE backward', 2 * bf(3 * M * d)), G('qkv: input grad', M, d, 3 * d), G('qkv: weight grad', d, 3 * d, M), OP('RMSNorm backward + residual (fused)', 4 * bf(M * d) + f32(M) + f32(d)), G('down: input grad', M, I, d), G('down: weight grad', I, d, M),
      OP('SwiGLU backward', bf(2 * M * I) + bf(M * I) + bf(2 * M * I)), G('gate/up: input grad', M, d, 2 * I), G('gate/up: weight grad', d, 2 * I, M), OP('RMSNorm backward + residual (fused)', 4 * bf(M * d) + f32(M) + f32(d)), G('out-proj: input grad', M, d, d), G('out-proj: weight grad', d, d, M)],
    coda: [OP('RoPE backward + z·dz (standalone, unfused)', 4 * bf(3 * M * d) + f32(M)), k9('qkv', M, d, 3 * d), dwn(M, d), G('qkv: weight grad', d, 3 * d, M), G('Kernel 10 (down)', M, I, d, bf(2 * M * I) + bf(M * I) + f32(M * I / BLK), 2 * I), AUX('z·dz from partials', f32(M * I / BLK) + f32(M)), G('down: weight grad', I, d, M),
      k9('gate/up', M, d, 2 * I), dwn(M, d), G('gate/up: weight grad', d, 2 * I, M), G('out-proj: input grad', M, d, d), G('out-proj: weight grad', d, d, M)] }; },
  'Layer Forward + Backward': (d) => ({ fig: 'blocks', base: [...BLOCKS['Layer Forward'](d).base, ...BLOCKS['Layer Backward'](d).base], coda: [...BLOCKS['Layer Forward'](d).coda, ...BLOCKS['Layer Backward'](d).coda] }),
};
export const SIZES = { kernel: [4096, 8192], block: [2048, 4096, 8192] };

// ---- prediction vs measurement ------------------------------------------------
const sum = (ks) => ks.reduce((t, k) => t + time(k), 0);
export function predict(name, size) {
  const def = (KERNELS[name] ?? BLOCKS[name])(size), level = KERNELS[name] ? 'kernel' : 'block';
  const tBase = sum(def.base), tCoda = sum(def.coda);
  const gemmBase = def.base.filter((k) => k.kind === 'gemm'), tGemm = def.gemm ? sum(def.gemm) : gemmBase.reduce((t, k) => t + k.f / PI, 0);
  const nonGemmBase = tBase - sum(gemmBase), nonGemmCoda = tCoda - def.coda.filter((k) => k.kind === 'gemm').reduce((t, k) => t + k.f / PI, 0);
  // epilogue bytes CODA's host GEMMs take on, and whether the HBM meter hides them under the tensor cores
  const hosts = def.coda.filter((k) => k.kind === 'gemm' && /Kernel/.test(k.name));
  const exposed = hosts.reduce((t, k) => t + Math.max(0, k.b / BW - k.f / PI), 0);
  const meas = PAPER[name]?.[size] ?? {};
  const ceilMeas = meas.cublas?.[0] ?? null;
  const out = { name, size, level, fig: def.fig, base: def.base, coda: def.coda, tBase, tCoda, tGemm, nonGemmBase, nonGemmCoda, exposed, hosts,
    pred: tBase / tCoda, ceilPred: tBase / tGemm, meas: { llm: meas.llm?.[0] ?? null, human: meas.human?.[0] ?? null, ceil: ceilMeas, quack: meas.quack?.[0] ?? null, liger: meas.liger?.[0] ?? null, flashinfer: meas.flashinfer?.[0] ?? null },
    sd: { llm: meas.llm?.[1] ?? null, human: meas.human?.[1] ?? null } };
  if (ceilMeas != null) {
    // the baseline's non-GEMM time implied by the measured ceiling, against its floor: the paper's analogue of 03's "× floor"
    out.xFloor = (ceilMeas - 1) / (out.ceilPred - 1);
    // what the epilogue cost the fused kernel, as a fraction of the raw GEMM: ceiling ÷ CODA − 1 (measured) vs the serialized-epilogue estimates
    const best = Math.max(out.meas.llm ?? 0, out.meas.human ?? 0);
    out.epiMeas = ceilMeas / best - 1;
    const k = hosts[0];
    out.epiBytes = (k.b - (def.gemm[0].b)) / BW / (k.f / PI);                                         // the epilogue's extra bytes, serialized
    const mma = 2 * EPI.tile * k.K / (PI / EPI.sms);
    out.epiExec = { plain: EPI.tile * EPI.plain / mma, fast: EPI.tile * (def.sfu ? EPI.fast : EPI.plain) / mma, precise: EPI.tile * (def.sfu ? EPI.precise : EPI.plain) / mma };   // the field guide's rates per tile, serialized
  }
  return out;
}
export function all() {
  const rows = [];
  for (const n of Object.keys(KERNELS)) for (const m of SIZES.kernel) rows.push(predict(n, m));
  for (const n of Object.keys(BLOCKS)) for (const d of SIZES.block) rows.push(predict(n, d));
  return rows;
}
export { KERNELS, BLOCKS };
