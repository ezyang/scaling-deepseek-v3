// Every kernel launch of one DeepSeek-V3 MoE layer, priced on both meters.
// src/kerndata.js carries the 240 launches of one steady-state layer cycle of
// DeepSeek's EP64 profile (traced µs, grids, names); this module says what
// each launch IS — which tensor it moves, in which dtype, through which GEMM —
// and prices it at speed of light: FLOPs ÷ the measured GEMM rate against
// bytes ÷ HBM bandwidth, the longer meter binding (03's conventions: the
// FP8/BF16 rates are 03's measured π^sol, HBM is the data-sheet 3.35 TB/s).
// Then it says where each launch goes under CODA (arXiv:2605.19269): into
// which GEMM's epilogue, into an auxiliary reduction over tile partials,
// into the all-to-all kernels, or nowhere (it stays). Operands are read off
// kernel names, template arguments and grid sizes; where a shape is inferred
// from the grid alone the spec says so (`inf: true`) and the page marks it.
// Node-importable (goldens pin the per-pass totals).
import { LAUNCHES, ROUTED_ROWS } from './kerndata.js';
import { DSV3 } from './model.js';

// ---- the shapes --------------------------------------------------------------
const a = DSV3;
export const S = 4096;                       // tokens per microbatch (the profile's)
export const D = a.hidden, QR = a.qRank, KVR = a.kvRank, RO = a.qkRope, H = a.heads, NO = a.qkNope, DV = a.vHead;
export const I = a.moeInter, NE = a.routedExperts, K = a.topk, EL = NE / 64;   // EP64: 4 local experts
export const R = ROUTED_ROWS.norm;            // routed rows per GPU, normalized (k · S)
export const QH = H * (NO + RO), KVH = H * (NO + DV), O = H * DV, KVP = KVR + RO;   // 24576 · 32768 · 16384 · 576
export const SCALE_BLOCK = 128;              // DeepSeek's fine-grained FP8: one fp32 scale per 1×128 (activations) / 128×128 (weights)

// rates (03's sheet): measured GEMM peaks, data-sheet HBM
export const RATES = { fp8: 1456.6e12, bf16: 757.7e12, hbm: 3.35e12 };

// bytes of one tensor of n elements in a format
const bf = (n) => 2 * n;
const f32 = (n) => 4 * n;
const f8 = (n) => n + 4 * n / SCALE_BLOCK;      // e4m3 + its per-128 fp32 scales
const e5m6 = (n) => 1.5 * n;                    // the 12-bit attention-output stash
// kernel families
const cast = (n) => bf(n) + f8(n);              // bf16 in → fp8 (+scales) out
const castBack = (n) => f8(n) + bf(n);          // fp8 → bf16
const requantT = (n) => 2 * f8(n);              // fp8 (row scales) → fp8 (column scales), transposed
const scaleT = (n) => 2 * f32(n / SCALE_BLOCK); // the scales tensor transposed
const wT = (n) => 2 * n;                        // an fp8 weight transposed (read + write)
const copy = (n) => 2 * bf(n);
const rope = (n) => 2 * bf(n);
const normRes = (r, c) => 4 * bf(r * c) + f32(r);        // read x, read the add; write x', write the norm; rstd
const norm = (r, c) => 2 * bf(r * c) + f32(r);
const normBwd = (r, c, g) => 3 * bf(r * c) + f32(r) + f32(c) * g;   // dy, x, dx; rstd; per-block dγ partials
const ew = (grid, block, perElem, vec = 4) => grid * block * vec * perElem;   // an elementwise kernel's traffic, from its launch shape
// a GEMM: M rows × K → N; `w` weight bytes/elem, `o` output bytes/elem, `groups` local experts, `acc` fp32 accumulate (weight grads)
const gemm = (M, N, Kd, { ab = 1, w = 1, o = 2, groups = 1, acc = false } = {}) => ({
  M, N, Kd, groups, f: 2 * M * N * Kd,
  b: (ab === 1 ? f8(M * Kd) : M * Kd * ab) + N * Kd * w * groups + (acc ? 2 * f32(M * N) : M * N * o),   // weight grads accumulate into fp32 (read + write)
});

// ---- the spec: one entry per launch, in kerndata order -----------------------
// kind: gemm · attn · op (memory-bound kernel) · glue (latency-bound, tiny) · comm
// how (CODA): epilogue (into `host`'s epilogue) · aux (a reduction over tile
// partials; `after` = its traffic) · fa (into FlashAttention's own kernels) ·
// a2a (into DeepEP's kernels) · cache (a weight transpose that need not recur
// per microbatch) · stash (replaced by a stash the forward epilogue writes) ·
// keep (unchanged). `add` on a host GEMM = HBM traffic its epilogue takes on.
const SPEC = [];
const put = (i, re, e) => {
  const o = LAUNCHES[i];
  if (!o || !re.test(o.name)) throw new Error(`launch ${i} is ${o?.k}, spec expected ${re}`);
  SPEC[i] = { i, ...e };
};
const GEMM = /GemmKernel|sm90_xmma_gemm/, CAST = /per_token_cast_to_fp8_with_channels/, ST = /utils::transpose/, LN = /ln_fwd_kernel/, LNB = /ln_bwd_kernel/, LNF = /ln_bwd_finalize/, CT = /per_channel_cast_and_transpose/, BT = /batched_transpose/, CB = /cast_back_impl/, EW = /elementwise|reduce_kernel|FillFunctor|scatter_gather|_copy_tensor|arange|Distribution|topk|unique_group|aux_fi/, ANY = /./;

// ======================= forward: attention sub-layer (attn(F)) =======================
put(0, LN, { lab: 'residual add + input RMSNorm', col: 'attn', op: 'input norm', kind: 'op', b: normRes(S, D),
  how: 'a2a', host: 'combine', after: bf(S * D) + f8(S * D), note: 'the previous layer\'s combine kernel assembles each token\'s full row, so it can add the residual, take the exact row RMS, and emit x ⊙ γ in FP8 with r folded into the scales' });
put(1, CAST, { lab: 'quantize the normed input → FP8 (for Q down)', col: 'attn', op: 'Q down', kind: 'op', b: cast(S * D), how: 'a2a', host: 'combine', after: 0, note: 'emitted by the combine epilogue' });
put(2, ST, { lab: 'transpose its scales', col: 'attn', op: 'Q down', kind: 'glue', b: scaleT(S * D), how: 'a2a', host: 'combine', after: 0 });
put(3, GEMM, { lab: 'Q down projection', col: 'attn', op: 'Q down', kind: 'gemm', dt: 'fp8', ...gemm(S, QR, D), epi: ['⊙ γ_q', 'partial RMS → r_q', 'FP8 out'], add: f32(S * QR / 128) });
put(4, CAST, { lab: 'quantize the normed input → FP8 (again, for KV down)', col: 'attn', op: 'KV down', kind: 'op', b: cast(S * D), how: 'a2a', host: 'combine', after: 0, note: 'the same tensor quantized twice' });
put(5, ST, { lab: 'transpose its scales', col: 'attn', op: 'KV down', kind: 'glue', b: scaleT(S * D), how: 'a2a', host: 'combine', after: 0 });
put(6, GEMM, { lab: 'KV down projection', col: 'attn', op: 'KV down', kind: 'gemm', dt: 'fp8', ...gemm(S, KVP, D), epi: ['⊙ γ_kv', 'partial RMS → r_kv', 'RoPE(k_rope)', 'FP8 out'], add: f32(S * KVR / 128) });
put(7, EW, { lab: 'position indices (arange)', col: 'attn', op: 'RoPE', kind: 'glue', b: 0, how: 'keep', note: 'a constant per step' });
put(8, EW, { lab: 'position indices (arange)', col: 'attn', op: 'RoPE', kind: 'glue', b: 0, how: 'keep' });
put(9, LN, { lab: 'q-latent RMSNorm', col: 'attn', op: 'q norm', kind: 'op', b: norm(S, QR), how: 'aux', after: 2 * f32(S * QR / 128) + f32(S), note: 'r_q from the Q-down epilogue\'s 12 partials per row; applied in the Q-up epilogue' });
put(10, EW, { lab: 'layout copy (q latent)', col: 'attn', op: 'q norm', kind: 'op', b: copy(S * QR), how: 'epilogue', host: 3, after: 0 });
put(11, LN, { lab: 'kv-latent RMSNorm', col: 'attn', op: 'kv norm', kind: 'op', b: norm(S, KVR), how: 'aux', after: 2 * f32(S * KVR / 128) + f32(S) });
put(12, EW, { lab: 'layout copy (k_rope slice)', col: 'attn', op: 'kv norm', kind: 'op', b: copy(S * RO), how: 'epilogue', host: 6, after: 0, inf: true });
put(13, CAST, { lab: 'quantize q latent → FP8', col: 'attn', op: 'Q up', kind: 'op', b: cast(S * QR), how: 'epilogue', host: 3, after: 0 });
put(14, ST, { lab: 'transpose its scales', col: 'attn', op: 'Q up', kind: 'glue', b: scaleT(S * QR), how: 'epilogue', host: 3, after: 0 });
put(15, GEMM, { lab: 'Q up projection', col: 'attn', op: 'Q up', kind: 'gemm', dt: 'fp8', ...gemm(S, QH, QR), epi: ['× r_q', 'RoPE(q_rope)', 'Q in FA layout'] });
put(16, CAST, { lab: 'quantize kv latent → FP8', col: 'attn', op: 'KV up', kind: 'op', b: cast(S * KVR), how: 'epilogue', host: 6, after: 0 });
put(17, ST, { lab: 'transpose its scales', col: 'attn', op: 'KV up', kind: 'glue', b: scaleT(S * KVR), how: 'epilogue', host: 6, after: 0 });
put(18, GEMM, { lab: 'KV up projection', col: 'attn', op: 'KV up', kind: 'gemm', dt: 'fp8', ...gemm(S, KVH, KVR), epi: ['× r_kv', 'K, V in FA layout'] });
put(19, /rotary/, { lab: 'RoPE on k_rope', col: 'attn', op: 'RoPE', kind: 'op', b: rope(S * RO), how: 'epilogue', host: 6, after: 0 });
put(20, /_copy_tensor/, { lab: 'copy q_nope into the Q buffer', col: 'attn', op: 'RoPE', kind: 'op', b: copy(S * H * NO), how: 'epilogue', host: 15, after: 0, inf: true });
put(21, EW, { lab: 'copy q_rope out', col: 'attn', op: 'RoPE', kind: 'op', b: copy(S * H * RO), how: 'epilogue', host: 15, after: 0, inf: true });
put(22, /rotary/, { lab: 'RoPE on q_rope (128 heads)', col: 'attn', op: 'RoPE', kind: 'op', b: rope(S * H * RO), how: 'epilogue', host: 15, after: 0 });
put(23, /compute_attn/, { lab: 'FlashAttention forward', col: 'attn', op: 'attention', kind: 'attn', dt: 'bf16', f: S * S * H * (NO + RO + DV),
  b: bf(S * QH) * 2 + bf(S * O) * 2 + f32(S * H), how: 'keep', note: 'its own fused kernel; the FP8 and E5M6 casts of its output could be its epilogue (tile-local: one head = one 128-block)' });
put(24, /special_fmt/, { lab: 'stash the attention output as E5M6', col: 'attn', op: 'O proj', kind: 'op', b: bf(S * O) + e5m6(S * O), how: 'fa', after: e5m6(S * O), note: 'written from FlashAttention\'s output tile instead of re-read' });
put(25, CAST, { lab: 'quantize the attention output → FP8', col: 'attn', op: 'O proj', kind: 'op', b: cast(S * O), how: 'fa', after: f8(S * O) });
put(26, ST, { lab: 'transpose its scales', col: 'attn', op: 'O proj', kind: 'glue', b: scaleT(S * O), how: 'fa', after: 0 });
put(27, GEMM, { lab: 'attention output projection', col: 'attn', op: 'O proj', kind: 'gemm', dt: 'fp8', ...gemm(S, D, O),
  epi: ['× c', '+ residual', '⊙ γ', 'partial RMS → r', 'FP8 out (+ bf16 for the router)'], add: 2 * bf(S * D) + f8(S * D) });
put(28, EW, { lab: 'scale the projection output (× constant)', col: 'attn', op: 'O proj', kind: 'op', b: 2 * bf(S * D), how: 'epilogue', host: 27, after: 0, inf: true });
put(29, LN, { lab: 'residual add + MoE-input RMSNorm', col: 'attn', op: 'MoE norm', kind: 'op', b: normRes(S, D), how: 'epilogue', host: 27, after: 2 * f32(S * D / 128) + f32(S), note: 'residual and γ in the epilogue; r from a reduction over 56 partials per row' });
put(30, GEMM, { lab: 'router logits', col: 'attn', op: 'router', kind: 'gemm', dt: 'bf16', ...gemm(S, NE, D, { ab: 2, w: 2, o: 4 }), epi: ['× r', 'sigmoid', '+ bias'] });
const RG = (i, lab, kind = 'glue', how = 'keep', host) => put(i, ANY, { lab, col: 'attn', op: 'router', kind, b: ew(LAUNCHES[i].grid[0] * LAUNCHES[i].grid[1], LAUNCHES[i].block[0], 8), how, host, after: how === 'keep' ? undefined : 0, inf: true });
RG(31, 'random normal (router noise)'); RG(32, 'sigmoid', 'glue', 'epilogue', 30); RG(33, 'row sum'); RG(34, 'divide'); RG(35, 'add the correction bias', 'glue', 'epilogue', 30); RG(36, 'group scores (top-2 sum per group)');
RG(37, 'fill mask'); RG(38, 'scatter group mask'); RG(39, 'copy'); RG(40, 'masked fill'); put(41, /gatherTopK/, { lab: 'top-8 experts per token (sort)', col: 'attn', op: 'router', kind: 'op', b: ew(4096, 256, 8), how: 'keep', inf: true }); RG(42, 'floor-divide'); RG(43, 'unique group indices'); RG(44, 'gather');
for (const [i, l] of [[45, 'fill'], [46, 'fill'], [47, 'add'], [48, 'scatter-add'], [49, 'compare'], [50, 'copy'], [51, 'reduce'], [52, 'multiply'], [53, 'add'], [54, 'add'], [55, 'reduce'], [56, 'add'], [57, 'divide']]) RG(i, `load-balance statistics: ${l}`);
put(58, /gatherTopK/, { lab: 'load-balance statistics: top-k (sort)', col: 'attn', op: 'router', kind: 'op', b: ew(4096, 256, 8), how: 'keep', inf: true });
RG(59, 'load-balance statistics: reduce'); RG(60, 'load-balance statistics: aux_fi'); RG(61, 'load-balance statistics: multiply'); RG(62, 'load-balance statistics: reduce'); RG(63, 'load-balance statistics: multiply');
put(64, CAST, { lab: 'quantize the MoE input → FP8 (dispatch)', col: 'attn', op: 'dispatch', kind: 'op', b: cast(S * D), how: 'epilogue', host: 27, after: 0, note: 'the O-proj epilogue emits x ⊙ γ in FP8; r rides the per-token scales through the all-to-all' });

// ======================= forward: MoE sub-layer (mlp(F)) =======================
put(102, ST, { lab: 'transpose scales', col: 'mlp', op: 'shared FC13', kind: 'glue', b: scaleT(S * D), how: 'epilogue', host: 27, after: 0 });
put(103, GEMM, { lab: 'shared expert gate/up', col: 'mlp', op: 'shared FC13', kind: 'gemm', dt: 'fp8', ...gemm(S, 2 * I, D), epi: ['× r', 'SwiGLU (pairwise)', 'FP8 out'], out: f8(S * I) - bf(S * 2 * I) });
put(104, /clean_and_count_expert/, { lab: 'count tokens per expert', col: 'mlp', op: 'pack', kind: 'glue', b: ew(4, 512, 16), how: 'a2a', host: 'dispatch', after: 0, note: 'DeepEP bookkeeping, 4 blocks: latency, not bytes', inf: true });
put(105, /get_fused_mapping/, { lab: 'build the token → expert-row mapping', col: 'mlp', op: 'pack', kind: 'glue', b: ew(4, 512, 16), how: 'a2a', host: 'dispatch', after: 0, note: '4 blocks: latency, not bytes', inf: true });
put(106, /expand_to_fused/, { lab: 'expand received tokens into expert rows (FP8 + scales)', col: 'mlp', op: 'pack', kind: 'op', b: 2 * f8(R * D), how: 'keep', note: 'not tile-local; the dispatch kernel could land tokens in expert order directly, which is a DeepEP change, not an epilogue — counted as staying' });
put(107, BT, { lab: 'transpose the 4 local W13 (FP8)', col: 'mlp', op: 'pack', kind: 'op', b: wT(EL * 2 * I * D), how: 'cache', after: 0, note: 'weights are constant for the step: transpose once, not per microbatch' });
put(108, EW, { lab: 'copy the scales', col: 'mlp', op: 'pack', kind: 'glue', b: 2 * f32(R * D / 128), how: 'a2a', host: 'dispatch', after: 0, inf: true });
put(109, GEMM, { lab: 'routed experts gate/up (grouped)', col: 'mlp', op: 'routed FC13', kind: 'gemm', dt: 'fp8', ...gemm(R, 2 * I, D, { groups: EL }), epi: ['× r (in the scales)', 'SwiGLU (pairwise)', '× top-k weight', 'FP8 out'], out: f8(R * I) - bf(R * 2 * I), add: f32(R) });
put(110, EW, { lab: 'scale fixup', col: 'mlp', op: 'routed FC13', kind: 'glue', b: ew(453, 128, 8), how: 'epilogue', host: 109, after: 0, inf: true });
put(111, /swiglu_forward_with_weight/, { lab: 'SwiGLU × routing weight → FP8', col: 'mlp', op: 'routed SwiGLU', kind: 'op', b: bf(R * 2 * I) + f8(R * I) + f32(R), how: 'epilogue', host: 109, after: 0, note: 'already one fused kernel; under CODA the gate/up intermediate is never written' });
put(112, BT, { lab: 'transpose the 4 local W2 (FP8)', col: 'mlp', op: 'routed FC2', kind: 'op', b: wT(EL * I * D), how: 'cache', after: 0 });
put(113, EW, { lab: 'copy the scales', col: 'mlp', op: 'routed FC2', kind: 'glue', b: 2 * f32(R * I / 128), how: 'epilogue', host: 109, after: 0, inf: true });
put(114, GEMM, { lab: 'routed experts down (grouped)', col: 'mlp', op: 'routed FC2', kind: 'gemm', dt: 'fp8', ...gemm(R, D, I, { groups: EL }), epi: [] });
put(115, /reduce_fused/, { lab: 'sum each token\'s local expert copies', col: 'mlp', op: 'combine', kind: 'op', b: 2 * bf(R * D), how: 'keep', note: 'a cross-row reduction, not tile-local; the combine kernel reads these rows anyway and could sum them — a DeepEP change, counted as staying' });
put(116, /swiglu_forward_with_weight/, { lab: 'shared SwiGLU → FP8', col: 'mlp', op: 'shared SwiGLU', kind: 'op', b: bf(S * 2 * I) + f8(S * I), how: 'epilogue', host: 103, after: 0 });
put(117, ST, { lab: 'transpose scales', col: 'mlp', op: 'shared FC2', kind: 'glue', b: scaleT(S * I), how: 'epilogue', host: 103, after: 0 });
put(118, GEMM, { lab: 'shared expert down', col: 'mlp', op: 'shared FC2', kind: 'gemm', dt: 'fp8', ...gemm(S, D, I), epi: ['+ residual (x after attention)'], add: bf(S * D) });
put(119, CAST, { lab: 're-quantize the routed SwiGLU output (R × 2048) for the weight gradient', col: 'mlp', op: 'stash', kind: 'op', b: cast(R * I), how: 'epilogue', host: 109, after: 0, inf: true, note: 'grid = routed rows; the time fits a 2,048-wide tensor, not 7,168' });
put(120, CAST, { lab: 're-quantize the shared SwiGLU output (S × 2048) for the weight gradient', col: 'mlp', op: 'stash', kind: 'op', b: cast(S * I), how: 'epilogue', host: 103, after: 0, inf: true });

// ======================= backward: MoE sub-layer (mlp(B)) =======================
put(65, CB, { lab: 'dequantize the shared gate/up stash (FP8 → bf16)', col: 'mlp', op: 'shared FC2 bwd', kind: 'op', b: castBack(S * 2 * I), how: 'epilogue', host: 68, after: 0, note: 'the SwiGLU-backward epilogue reads the FP8 stash tile directly' });
put(66, BT, { lab: 'transpose the shared W2 (FP8)', col: 'mlp', op: 'shared FC2 bwd', kind: 'op', b: wT(I * D), how: 'cache', after: 0 });
put(67, ST, { lab: 'transpose scales', col: 'mlp', op: 'shared FC2 bwd', kind: 'glue', b: scaleT(S * D), how: 'epilogue', host: 'grad', after: 0 });
put(68, GEMM, { lab: 'shared FC2: input gradient', col: 'mlp', op: 'shared FC2 bwd', kind: 'gemm', dt: 'fp8', ...gemm(S, I, D), epi: ['SwiGLU backward (pairwise)', 'd(gate/up) → FP8, both layouts'], add: f8(S * 2 * I) + 2 * f8(S * 2 * I) - bf(S * I) });
put(69, CT, { lab: 'd(shared out) → FP8ᵀ (weight-gradient operand)', col: 'mlp', op: 'shared FC2 bwd', kind: 'op', b: cast(S * D), how: 'epilogue', host: 'grad', after: 0, note: 'the epilogue that produces this gradient can write both scale layouts' });
put(70, /swiglu_and_per_channel/, { lab: 'recompute SwiGLU → FP8ᵀ (weight-gradient operand)', col: 'mlp', op: 'shared SwiGLU bwd', kind: 'op', b: bf(S * 2 * I) + f8(S * I), how: 'epilogue', host: 68, after: 0 });
put(71, /swiglu_backward/, { lab: 'SwiGLU backward', col: 'mlp', op: 'shared SwiGLU bwd', kind: 'op', b: bf(S * 2 * I) * 2 + bf(S * I), how: 'epilogue', host: 68, after: 0 });
put(72, CB, { lab: 'dequantize the routed gate/up stash (FP8 → bf16)', col: 'mlp', op: 'routed SwiGLU bwd', kind: 'op', b: castBack(R * 2 * I), how: 'epilogue', host: 75, after: 0, inf: true, note: '112 blocks for a 400 MB pass: 1.4× its floor' });
put(73, /expand_to_fused/, { lab: 'expand the received output gradients into expert rows', col: 'mlp', op: 'unpack', kind: 'op', b: 2 * f8(R * D), how: 'keep', note: 'not tile-local (see the forward expand)' });
put(74, EW, { lab: 'copy the scales', col: 'mlp', op: 'unpack', kind: 'glue', b: 2 * f32(R * D / 128), how: 'a2a', host: 'dispatch', after: 0, inf: true });
put(75, GEMM, { lab: 'routed FC2: input gradient (grouped)', col: 'mlp', op: 'routed FC2 bwd', kind: 'gemm', dt: 'fp8', ...gemm(R, I, D, { groups: EL }),
  epi: ['SwiGLU backward (pairwise)', '× top-k weight', 'partial Σ for d(weight)', 'd(gate/up) → FP8, both layouts'], add: f8(R * 2 * I) + 2 * f8(R * 2 * I) - bf(R * I) });
put(76, /swiglu_forward_backward_with_weight/, { lab: 'SwiGLU forward + backward × routing weight', col: 'mlp', op: 'routed SwiGLU bwd', kind: 'op', b: bf(R * 2 * I) * 2 + bf(R * I) + f32(R), how: 'epilogue', host: 75, after: 0 });
put(77, /swiglu_weight_backward/, { lab: 'routing-weight gradient (row dot products)', col: 'mlp', op: 'routed SwiGLU bwd', kind: 'op', b: bf(R * I) * 2 + f32(R), how: 'aux', host: 75, after: f32(R * 16) * 2, note: 'a row reduction: tile partials from the FC2-backward epilogue, summed by a small kernel' });
put(78, CT, { lab: 'SwiGLU output → FP8ᵀ (weight-gradient operand)', col: 'mlp', op: 'routed SwiGLU bwd', kind: 'op', b: bf(R * I) + f8(R * I), how: 'epilogue', host: 75, after: 0 });
put(79, EW, { lab: 'scale fixup', col: 'mlp', op: 'routed SwiGLU bwd', kind: 'glue', b: ew(502, 128, 8), how: 'epilogue', host: 75, after: 0, inf: true });
put(80, LN, { lab: 'recompute: residual add + MoE-input RMSNorm', col: 'mlp', op: 'recompute', kind: 'op', b: normRes(S, D), how: 'epilogue', host: 91, after: 0, note: 'the RMSNorm-backward epilogue reads the saved pre-norm row (h₁) and r instead of recomputing the norm' });
put(81, CAST, { lab: 'quantize the recomputed MoE input → FP8', col: 'mlp', op: 'routed FC13 bwd', kind: 'op', b: cast(S * D), how: 'stash', after: 0, inf: true, note: 'the forward O-proj epilogue already wrote this tensor in FP8' });
put(82, CAST, { lab: 'quantize d(gate/up) → FP8 (R × 4096)', col: 'mlp', op: 'routed FC13 bwd', kind: 'op', b: cast(R * 2 * I), how: 'epilogue', host: 75, after: 0 });
put(83, EW, { lab: 'copy the scales', col: 'mlp', op: 'routed FC13 bwd', kind: 'glue', b: 2 * f32(R * 2 * I / 128), how: 'epilogue', host: 75, after: 0, inf: true });
put(84, GEMM, { lab: 'routed FC13: input gradient (grouped)', col: 'mlp', op: 'routed FC13 bwd', kind: 'gemm', dt: 'fp8', ...gemm(R, D, 2 * I, { groups: EL }), epi: ['partial Σ x ⊙ dx → ŝ (rides the combine)'], add: f32(R) });
put(85, /reduce_fused/, { lab: 'sum each token\'s local input-gradient copies', col: 'mlp', op: 'unpack', kind: 'op', b: 2 * bf(R * D), how: 'keep', note: 'not tile-local (see the forward reduction)' });
put(86, CT, { lab: 'd(gate/up) → FP8ᵀ (weight-gradient operand)', col: 'mlp', op: 'routed FC13 bwd', kind: 'op', b: cast(R * 2 * I), how: 'epilogue', host: 75, after: 0 });
put(87, CB, { lab: 'dequantize the MoE-input stash (FP8 → bf16)', col: 'mlp', op: 'shared FC13 bwd', kind: 'op', b: castBack(S * D), how: 'stash', after: 0, inf: true });
put(88, BT, { lab: 'transpose the shared W13 (FP8)', col: 'mlp', op: 'shared FC13 bwd', kind: 'op', b: wT(2 * I * D), how: 'cache', after: 0 });
put(89, CAST, { lab: 'quantize d(gate/up) → FP8 (S × 4096)', col: 'mlp', op: 'shared FC13 bwd', kind: 'op', b: cast(S * 2 * I), how: 'epilogue', host: 68, after: 0 });
put(90, ST, { lab: 'transpose scales', col: 'mlp', op: 'shared FC13 bwd', kind: 'glue', b: scaleT(S * 2 * I), how: 'epilogue', host: 68, after: 0 });
put(91, GEMM, { lab: 'shared FC13: input gradient', col: 'mlp', op: 'shared FC13 bwd', kind: 'gemm', dt: 'fp8', ...gemm(S, D, 2 * I),
  epi: ['+ router and routed d(input)', 'RMSNorm backward (needs s, r)', 'partial dγ'], add: 2 * bf(S * D) + bf(S * D) });
put(92, CT, { lab: 'd(gate/up) → FP8ᵀ (weight-gradient operand)', col: 'mlp', op: 'shared FC13 bwd', kind: 'op', b: cast(S * 2 * I), how: 'epilogue', host: 68, after: 0 });
put(93, CT, { lab: 'MoE input → FP8ᵀ (weight-gradient operand)', col: 'mlp', op: 'shared FC13 bwd', kind: 'op', b: cast(S * D), how: 'stash', after: 0, note: 'the forward epilogue can stash both scale layouts (+29 MB of stash per layer)' });

// ======================= weight gradients (mlp(W), attn(W)) =======================
put(94, GEMM, { lab: 'shared FC2: weight gradient', col: 'mlp', op: 'shared FC2 wgrad', kind: 'gemm', dt: 'fp8', ...gemm(I, D, S, { acc: true }) });
put(95, CT, { lab: 'd(routed out) → FP8ᵀ (R × 7168)', col: 'mlp', op: 'routed FC2 wgrad', kind: 'op', b: requantT(R * D), how: 'a2a', host: 'dispatch', after: 0, note: 'the backward dispatch could deliver the gradient in both scale layouts' });
put(96, GEMM, { lab: 'routed FC2: weight gradient, expert 1', col: 'mlp', op: 'routed FC2 wgrad', kind: 'gemm', dt: 'fp8', ...gemm(I, D, R / EL, { acc: true }) });
put(97, GEMM, { lab: 'routed FC2: weight gradient, expert 2', col: 'mlp', op: 'routed FC2 wgrad', kind: 'gemm', dt: 'fp8', ...gemm(I, D, R / EL, { acc: true }) });
put(98, CT, { lab: 'dispatched input → FP8ᵀ (R × 7168)', col: 'mlp', op: 'routed FC13 wgrad', kind: 'op', b: requantT(R * D), how: 'a2a', host: 'dispatch', after: 0, note: 'the forward dispatch could stash both scale layouts' });
put(99, GEMM, { lab: 'routed FC13: weight gradient, expert 1', col: 'mlp', op: 'routed FC13 wgrad', kind: 'gemm', dt: 'fp8', ...gemm(2 * I, D, R / EL, { acc: true }) });
put(100, GEMM, { lab: 'routed FC13: weight gradient, expert 2', col: 'mlp', op: 'routed FC13 wgrad', kind: 'gemm', dt: 'fp8', ...gemm(2 * I, D, R / EL, { acc: true }) });
put(101, GEMM, { lab: 'shared FC13: weight gradient', col: 'mlp', op: 'shared FC13 wgrad', kind: 'gemm', dt: 'fp8', ...gemm(2 * I, D, S, { acc: true }) });
put(233, GEMM, { lab: 'routed FC2: weight gradient, expert 3 (2nd stream)', col: 'mlp', op: 'routed FC2 wgrad', kind: 'gemm', dt: 'fp8', ...gemm(I, D, R / EL, { acc: true }) });
put(235, GEMM, { lab: 'routed FC2: weight gradient, expert 4 (2nd stream)', col: 'mlp', op: 'routed FC2 wgrad', kind: 'gemm', dt: 'fp8', ...gemm(I, D, R / EL, { acc: true }) });
put(236, GEMM, { lab: 'routed FC13: weight gradient, expert 3 (2nd stream)', col: 'mlp', op: 'routed FC13 wgrad', kind: 'gemm', dt: 'fp8', ...gemm(2 * I, D, R / EL, { acc: true }) });
put(237, GEMM, { lab: 'routed FC13: weight gradient, expert 4 (2nd stream)', col: 'mlp', op: 'routed FC13 wgrad', kind: 'gemm', dt: 'fp8', ...gemm(2 * I, D, R / EL, { acc: true }) });
put(221, GEMM, { lab: 'O proj: weight gradient', col: 'attn', op: 'O proj wgrad', kind: 'gemm', dt: 'fp8', ...gemm(O, D, S, { acc: true }) });
put(222, GEMM, { lab: 'KV up: weight gradient', col: 'attn', op: 'KV up wgrad', kind: 'gemm', dt: 'fp8', ...gemm(KVR, KVH, S, { acc: true }) });
put(223, GEMM, { lab: 'Q up: weight gradient', col: 'attn', op: 'Q up wgrad', kind: 'gemm', dt: 'fp8', ...gemm(QR, QH, S, { acc: true }) });
put(224, GEMM, { lab: 'KV down: weight gradient', col: 'attn', op: 'KV down wgrad', kind: 'gemm', dt: 'fp8', ...gemm(D, KVP, S, { acc: true }) });
put(225, GEMM, { lab: 'Q down: weight gradient', col: 'attn', op: 'Q down wgrad', kind: 'gemm', dt: 'fp8', ...gemm(D, QR, S, { acc: true }) });

// ======================= backward: attention sub-layer (attn(B)) =======================
const RB = (i, lab, how = 'keep', host) => put(i, ANY, { lab: `router backward: ${lab}`, col: 'attn', op: 'router bwd', kind: 'glue', b: ew(LAUNCHES[i].grid[0], LAUNCHES[i].block[0], 8), how, host, after: how === 'keep' ? undefined : 0, inf: true });
for (const [i, l] of [[121, 'fill'], [122, 'reduce'], [123, 'multiply'], [124, 'multiply'], [125, 'fill'], [126, 'copy'], [127, 'divide'], [128, 'divide'], [129, 'negate'], [130, 'multiply'], [131, 'divide'], [132, 'reduce'], [133, 'add'], [134, 'fill'], [135, 'scatter-add'],
  [136, 'divide'], [137, 'divide'], [138, 'negate'], [139, 'multiply'], [140, 'divide'], [141, 'reduce'], [142, 'add'], [143, 'add'], [144, 'sigmoid backward'], [145, 'copy']]) RB(i, l);
put(146, GEMM, { lab: 'router: input gradient', col: 'attn', op: 'router bwd', kind: 'gemm', dt: 'bf16', ...gemm(S, D, NE, { ab: 2, w: 2, o: 2 }), epi: ['× r'] });
put(147, GEMM, { lab: 'router: weight gradient', col: 'attn', op: 'router bwd', kind: 'gemm', dt: 'bf16', ...gemm(D, NE, S, { ab: 2, w: 2, o: 4 }) });
put(148, EW, { lab: 'accumulate d(MoE input) (+ router\'s)', col: 'attn', op: 'MoE norm bwd', kind: 'op', b: 3 * bf(S * D), how: 'epilogue', host: 91, after: 0 });
put(149, LNB, { lab: 'MoE-input RMSNorm backward', col: 'attn', op: 'MoE norm bwd', kind: 'op', b: normBwd(S, D, 132), how: 'epilogue', host: 91, after: 0, note: 's for this norm = Σ over the router, shared and routed outputs ⊙ their gradients: the routed share is accumulated on other ranks and must ride the gradient combine' });
put(150, LNF, { lab: 'dγ finalize', col: 'attn', op: 'MoE norm bwd', kind: 'glue', b: 2 * f32(D * 32), how: 'aux', after: 2 * f32(D * 32) });
put(151, EW, { lab: 'dγ accumulate', col: 'attn', op: 'MoE norm bwd', kind: 'glue', b: 3 * f32(D), how: 'aux', after: 0 });
put(152, EW, { lab: 'scale backward (× constant)', col: 'attn', op: 'O proj bwd', kind: 'op', b: 2 * bf(S * D), how: 'epilogue', host: 91, after: 0, inf: true });
put(153, /cast_back_special_fmt/, { lab: 'dequantize the E5M6 attention-output stash → bf16', col: 'attn', op: 'O proj bwd', kind: 'op', b: e5m6(S * O) + bf(S * O), how: 'keep', after: e5m6(S * O) + f8(S * O), note: 'no GEMM produces this tensor in the backward: one pass E5M6 → FP8ᵀ remains (today it is three passes)' });
put(154, CAST, { lab: 'quantize d(attention out) → FP8', col: 'attn', op: 'O proj bwd', kind: 'op', b: cast(S * D), how: 'epilogue', host: 91, after: 0 });
put(155, BT, { lab: 'transpose W_o (FP8)', col: 'attn', op: 'O proj bwd', kind: 'op', b: wT(O * D), how: 'cache', after: 0 });
put(156, ST, { lab: 'transpose scales', col: 'attn', op: 'O proj bwd', kind: 'glue', b: scaleT(S * D), how: 'epilogue', host: 91, after: 0 });
put(157, GEMM, { lab: 'O proj: input gradient (dO)', col: 'attn', op: 'O proj bwd', kind: 'gemm', dt: 'fp8', ...gemm(S, O, D), epi: [] });
put(158, CT, { lab: 'attention output → FP8ᵀ (weight-gradient operand)', col: 'attn', op: 'O proj bwd', kind: 'op', b: cast(S * O), how: 'keep', after: 0, note: 'folded into the one E5M6 → FP8ᵀ pass above' });
put(159, CT, { lab: 'd(attention out) → FP8ᵀ (weight-gradient operand)', col: 'attn', op: 'O proj bwd', kind: 'op', b: cast(S * D), how: 'epilogue', host: 91, after: 0 });
put(160, LN, { lab: 'recompute: q-latent RMSNorm', col: 'attn', op: 'recompute', kind: 'op', b: norm(S, QR), how: 'stash', after: 0, note: 'the forward Q-down epilogue\'s FP8 (latent ⊙ γ) and r_q are stashed instead of the bf16 latent: 3.7 MB less, no recompute' });
put(161, EW, { lab: 'recompute: layout copy', col: 'attn', op: 'recompute', kind: 'op', b: copy(S * QR), how: 'stash', after: 0 });
put(162, LN, { lab: 'recompute: kv-latent RMSNorm', col: 'attn', op: 'recompute', kind: 'op', b: norm(S, KVR), how: 'stash', after: 0 });
put(163, CAST, { lab: 'recompute: quantize q latent → FP8', col: 'attn', op: 'recompute', kind: 'op', b: cast(S * QR), how: 'stash', after: 0 });
put(164, ST, { lab: 'transpose scales', col: 'attn', op: 'recompute', kind: 'glue', b: scaleT(S * QR), how: 'stash', after: 0 });
put(165, GEMM, { lab: 'recompute: Q up projection', col: 'attn', op: 'recompute', kind: 'gemm', dt: 'fp8', ...gemm(S, QH, QR), epi: ['× r_q', 'RoPE(q_rope)', 'Q in FA layout'] });
put(166, CAST, { lab: 'recompute: quantize kv latent → FP8', col: 'attn', op: 'recompute', kind: 'op', b: cast(S * KVR), how: 'stash', after: 0 });
put(167, ST, { lab: 'transpose scales', col: 'attn', op: 'recompute', kind: 'glue', b: scaleT(S * KVR), how: 'stash', after: 0 });
put(168, GEMM, { lab: 'recompute: KV up projection', col: 'attn', op: 'recompute', kind: 'gemm', dt: 'fp8', ...gemm(S, KVH, KVR), epi: ['× r_kv', 'K, V in FA layout'] });
put(169, /rotary/, { lab: 'recompute: RoPE on k_rope', col: 'attn', op: 'recompute', kind: 'op', b: rope(S * RO), how: 'stash', after: 0 });
put(170, /_copy_tensor/, { lab: 'recompute: copy q_nope', col: 'attn', op: 'recompute', kind: 'op', b: copy(S * H * NO), how: 'epilogue', host: 165, after: 0, inf: true });
put(171, EW, { lab: 'recompute: copy q_rope', col: 'attn', op: 'recompute', kind: 'op', b: copy(S * H * RO), how: 'epilogue', host: 165, after: 0, inf: true });
put(172, /rotary/, { lab: 'recompute: RoPE on q_rope', col: 'attn', op: 'recompute', kind: 'op', b: rope(S * H * RO), how: 'epilogue', host: 165, after: 0 });
put(173, /FlashAttnBwdPreprocess/, { lab: 'FlashAttention backward: preprocess (D = rowsum O ⊙ dO, zero dQ accumulator)', col: 'attn', op: 'attention bwd', kind: 'op', b: 2 * bf(S * O) + f32(S * H) + f32(S * QH), how: 'keep' });
put(174, /FlashAttnBwd</, { lab: 'FlashAttention backward', col: 'attn', op: 'attention bwd', kind: 'attn', dt: 'bf16', f: S * S * H * (3 * (NO + RO) + 2 * DV),
  b: 2 * bf(S * QH) + 3 * bf(S * O) + 2 * f32(S * H) + 2 * f32(S * QH) + bf(S * QH) + bf(S * O), how: 'keep' });
put(175, /ConvertdQ/, { lab: 'FlashAttention backward: dQ fp32 → bf16', col: 'attn', op: 'attention bwd', kind: 'op', b: f32(S * QH) + bf(S * QH), how: 'keep', epi: ['RoPE backward', 'dQ → FP8, both layouts', 'partial Σ for s_q'], after: f32(S * QH) + 2 * f8(S * QH) });
put(176, /rotary/, { lab: 'RoPE backward on dQ_rope', col: 'attn', op: 'Q up bwd', kind: 'op', b: rope(S * H * RO), how: 'fa', host: 175, after: 0 });
put(177, BT, { lab: 'transpose W_q_up (FP8)', col: 'attn', op: 'Q up bwd', kind: 'op', b: wT(QR * QH), how: 'cache', after: 0 });
put(178, CAST, { lab: 'quantize dQ → FP8 (S × 24576)', col: 'attn', op: 'Q up bwd', kind: 'op', b: cast(S * QH), how: 'fa', host: 175, after: 0 });
put(179, ST, { lab: 'transpose scales', col: 'attn', op: 'Q up bwd', kind: 'glue', b: scaleT(S * QH), how: 'fa', host: 175, after: 0 });
put(180, GEMM, { lab: 'Q up: input gradient', col: 'attn', op: 'Q up bwd', kind: 'gemm', dt: 'fp8', ...gemm(S, QR, QH), epi: ['× r_q', 'q-latent RMSNorm backward', 'd(latent) → FP8, both layouts', 'partial Σ for s (input norm)'], add: 3 * f8(S * QR) + bf(S * QR) - bf(S * QR) });
put(181, EW, { lab: 'sum dk_rope over the 128 heads', col: 'attn', op: 'KV up bwd', kind: 'op', b: bf(S * H * RO) + bf(S * RO), how: 'aux', after: bf(S * H * RO) + bf(S * RO), note: 'a reduction across heads (across tiles): stays a kernel' });
put(182, /rotary/, { lab: 'RoPE backward on dk_rope', col: 'attn', op: 'KV up bwd', kind: 'op', b: rope(S * RO), how: 'aux', after: 0 });
put(183, /_copy_tensor/, { lab: 'copy dk_nope into the dKV buffer', col: 'attn', op: 'KV up bwd', kind: 'op', b: copy(S * H * NO), how: 'fa', host: 174, after: 0, inf: true, note: 'FlashAttention can write dK_nope and dV into one buffer' });
put(184, /_copy_tensor/, { lab: 'copy dV into the dKV buffer', col: 'attn', op: 'KV up bwd', kind: 'op', b: copy(S * H * DV), how: 'fa', host: 174, after: 0, inf: true });
put(185, BT, { lab: 'transpose W_kv_up (FP8)', col: 'attn', op: 'KV up bwd', kind: 'op', b: wT(KVR * KVH), how: 'cache', after: 0 });
put(186, CAST, { lab: 'quantize dKV → FP8 (S × 32768)', col: 'attn', op: 'KV up bwd', kind: 'op', b: cast(S * KVH), how: 'fa', host: 174, after: 0, note: 'one head = one 128-block: tile-local in FlashAttention\'s dK/dV store' });
put(187, ST, { lab: 'transpose scales', col: 'attn', op: 'KV up bwd', kind: 'glue', b: scaleT(S * KVH), how: 'fa', host: 174, after: 0 });
put(188, GEMM, { lab: 'KV up: input gradient', col: 'attn', op: 'KV up bwd', kind: 'gemm', dt: 'fp8', ...gemm(S, KVR, KVH), epi: ['× r_kv', 'kv-latent RMSNorm backward', 'd(kv latent ‖ dk_rope) → FP8, both layouts', 'partial Σ for s (input norm)'], add: 3 * f8(S * KVP) });
put(189, CT, { lab: 'dKV → FP8ᵀ (weight-gradient operand)', col: 'attn', op: 'KV up bwd', kind: 'op', b: cast(S * KVH), how: 'fa', host: 174, after: 0 });
put(190, CT, { lab: 'kv latent → FP8ᵀ (weight-gradient operand)', col: 'attn', op: 'KV up bwd', kind: 'op', b: cast(S * KVR), how: 'stash', after: 0 });
put(191, CT, { lab: 'dQ → FP8ᵀ (weight-gradient operand)', col: 'attn', op: 'Q up bwd', kind: 'op', b: cast(S * QH), how: 'fa', host: 175, after: 0 });
put(192, CT, { lab: 'q latent → FP8ᵀ (weight-gradient operand)', col: 'attn', op: 'Q up bwd', kind: 'op', b: cast(S * QR), how: 'stash', after: 0 });
put(193, LNB, { lab: 'kv-latent RMSNorm backward', col: 'attn', op: 'kv norm bwd', kind: 'op', b: normBwd(S, KVR, 528), how: 'epilogue', host: 188, after: 0 });
put(194, LNF, { lab: 'dγ finalize', col: 'attn', op: 'kv norm bwd', kind: 'glue', b: 2 * f32(KVR * 32), how: 'aux', after: 2 * f32(KVR * 32) });
put(195, EW, { lab: 'dγ accumulate', col: 'attn', op: 'kv norm bwd', kind: 'glue', b: 3 * f32(KVR), how: 'aux', after: 0 });
put(196, LNB, { lab: 'q-latent RMSNorm backward', col: 'attn', op: 'q norm bwd', kind: 'op', b: normBwd(S, QR, 660), how: 'epilogue', host: 180, after: 0 });
put(197, LNF, { lab: 'dγ finalize', col: 'attn', op: 'q norm bwd', kind: 'glue', b: 2 * f32(QR * 32), how: 'aux', after: 2 * f32(QR * 32) });
put(198, EW, { lab: 'dγ accumulate', col: 'attn', op: 'q norm bwd', kind: 'glue', b: 3 * f32(QR), how: 'aux', after: 0 });
for (const [i, l] of [[199, 'fill'], [200, 'copy dk_rope'], [201, 'fill'], [202, 'copy d(kv latent)'], [203, 'add']])
  put(i, EW, { lab: `assemble d(kv packed): ${l}`, col: 'attn', op: 'KV down bwd', kind: 'glue', b: ew(LAUNCHES[i].grid[0], 128, 4), how: 'epilogue', host: 188, after: 0, inf: true });
put(204, LN, { lab: 'recompute: input RMSNorm (the down-projections\' input)', col: 'attn', op: 'recompute', kind: 'op', b: norm(S, D) + bf(S * D), how: 'stash', after: 0, note: 'the forward wrote x ⊙ γ in FP8; stash it in both scale layouts instead of recomputing (+29 MB of stash per layer)' });
put(205, CAST, { lab: 'quantize d(kv packed) → FP8', col: 'attn', op: 'KV down bwd', kind: 'op', b: cast(S * KVP), how: 'epilogue', host: 188, after: 0 });
put(206, BT, { lab: 'transpose W_kv_down (FP8)', col: 'attn', op: 'KV down bwd', kind: 'op', b: wT(KVP * D), how: 'cache', after: 0 });
put(207, ST, { lab: 'transpose scales', col: 'attn', op: 'KV down bwd', kind: 'glue', b: scaleT(S * KVP), how: 'epilogue', host: 188, after: 0 });
put(208, GEMM, { lab: 'KV down: input gradient', col: 'attn', op: 'KV down bwd', kind: 'gemm', dt: 'fp8', ...gemm(S, D, KVP), epi: [] });
put(209, CT, { lab: 'normed input → FP8ᵀ (weight-gradient operand)', col: 'attn', op: 'KV down bwd', kind: 'op', b: cast(S * D), how: 'stash', after: 0 });
put(210, CT, { lab: 'd(kv packed) → FP8ᵀ (weight-gradient operand)', col: 'attn', op: 'KV down bwd', kind: 'op', b: cast(S * KVP), how: 'epilogue', host: 188, after: 0 });
put(211, CAST, { lab: 'quantize d(q latent) → FP8', col: 'attn', op: 'Q down bwd', kind: 'op', b: cast(S * QR), how: 'epilogue', host: 180, after: 0 });
put(212, BT, { lab: 'transpose W_q_down (FP8)', col: 'attn', op: 'Q down bwd', kind: 'op', b: wT(QR * D), how: 'cache', after: 0 });
put(213, ST, { lab: 'transpose scales', col: 'attn', op: 'Q down bwd', kind: 'glue', b: scaleT(S * QR), how: 'epilogue', host: 180, after: 0 });
put(214, GEMM, { lab: 'Q down: input gradient', col: 'attn', op: 'Q down bwd', kind: 'gemm', dt: 'fp8', ...gemm(S, D, QR), epi: ['+ KV down\'s d(input)', 'input RMSNorm backward (needs s, r)', '+ residual gradient', 'partial dγ'], add: 3 * bf(S * D) });
put(215, CT, { lab: 'normed input → FP8ᵀ (again, for Q down\'s weight gradient)', col: 'attn', op: 'Q down bwd', kind: 'op', b: cast(S * D), how: 'stash', after: 0, note: 'the same tensor quantized and transposed twice' });
put(216, CT, { lab: 'd(q latent) → FP8ᵀ (weight-gradient operand)', col: 'attn', op: 'Q down bwd', kind: 'op', b: cast(S * QR), how: 'epilogue', host: 180, after: 0 });
put(217, EW, { lab: 'add the two down-projections\' input gradients', col: 'attn', op: 'input norm bwd', kind: 'op', b: 3 * bf(S * D), how: 'epilogue', host: 214, after: 0 });
put(218, LNB, { lab: 'input RMSNorm backward (+ residual gradient)', col: 'attn', op: 'input norm bwd', kind: 'op', b: normBwd(S, D, 132), how: 'epilogue', host: 214, after: 0, note: 's for this norm comes from the Q-up / KV-up backward epilogues (Σ latent ⊙ d(latent)) — one GEMM boundary later, the paper\'s identity' });
put(219, LNF, { lab: 'dγ finalize', col: 'attn', op: 'input norm bwd', kind: 'glue', b: 2 * f32(D * 32), how: 'aux', after: 2 * f32(D * 32) });
put(220, EW, { lab: 'dγ accumulate', col: 'attn', op: 'input norm bwd', kind: 'glue', b: 3 * f32(D), how: 'aux', after: 0 });

// ======================= the communication stream =======================
put(226, CAST, { lab: 'quantize d(MoE output) → FP8 before the backward dispatch', col: 'comm', op: 'combine bwd', kind: 'op', b: cast(S * D), how: 'epilogue', host: 'grad', after: 0, note: 'produced by the next layer\'s input-RMSNorm-backward epilogue, in FP8 directly' });
const CM = (i, lab, op) => put(i, /dpsk::ep/, { lab, col: 'comm', op, kind: 'comm', b: 0, how: 'keep' });
CM(227, 'DeepEP: notify (backward dispatch)', 'combine bwd'); CM(228, 'DeepEP: dispatch the output gradients (FP8)', 'combine bwd');
CM(229, 'DeepEP: dispatch layout', 'dispatch'); CM(230, 'DeepEP: notify (dispatch)', 'dispatch'); CM(231, 'DeepEP: dispatch tokens (FP8 + scales)', 'dispatch');
CM(232, 'DeepEP: notify (backward combine)', 'dispatch bwd'); CM(234, 'DeepEP: combine the input gradients (BF16)', 'dispatch bwd');
CM(238, 'DeepEP: notify (combine)', 'combine'); CM(239, 'DeepEP: combine the expert outputs (BF16)', 'combine');

for (let i = 0; i < LAUNCHES.length; i++) if (!SPEC[i]) throw new Error(`launch ${i} (${LAUNCHES[i].k}) has no spec`);

// ---- pricing -------------------------------------------------------------------
export const PASS_OF = (ann) => ann.includes('(F)') ? 'F' : ann.includes('(B)') ? 'B' : 'W';
// every launch, with both meters and the CODA state, at the given rates
export function priced(rates = RATES) {
  return LAUNCHES.map((o, i) => {
    const s = SPEC[i];
    const pass = PASS_OF(o.ann), comm = o.s === 27;
    const flop = s.f ? s.f / rates[s.dt] : 0, hbm = s.b / rates.hbm;
    const sol = Math.max(flop, hbm), bound = s.f ? (flop >= hbm ? 'compute' : 'memory') : 'memory';
    const host = s.kind === 'gemm' || s.kind === 'attn' || s.i === 175;
    // after CODA: hosts keep their FLOP time but take on epilogue traffic; removed launches leave `after`
    const bAfter = host ? s.b + (s.add ?? 0) + (s.out ?? 0) : s.how === 'keep' && s.after == null ? s.b : (s.after ?? 0);
    const hbmAfter = bAfter / rates.hbm;
    const solAfter = host ? Math.max(flop, hbmAfter) : hbmAfter;
    // gone: the launch folds away (into a host, a stash, a once-per-step transpose); aux: a small kernel remains in its place
    const gone = !host && !comm && s.how !== 'keep';
    return { ...o, ...s, pass, comm, flop, hbm, sol, bound, x: sol ? o.us * 1e-6 / sol : null, bAfter, hbmAfter, solAfter, gone, aux: gone && bAfter > 0 };
  });
}
// per pass: trace, SOL, bytes — GEMM+attention vs everything else, today and under CODA
export function tally(rates = RATES) {
  const P = priced(rates);
  const T = {};
  for (const pass of ['F', 'B', 'W']) {
    const es = P.filter((e) => e.pass === pass && !e.comm);
    const g = es.filter((e) => e.kind === 'gemm' || e.kind === 'attn'), o = es.filter((e) => !(e.kind === 'gemm' || e.kind === 'attn'));
    const sum = (xs, k) => xs.reduce((t, e) => t + e[k], 0);
    T[pass] = {
      launches: es.length, launchesAfter: es.filter((e) => !e.gone || e.aux).length,
      gemm: { trace: sum(g, 'raw'), traceNorm: sum(g, 'us'), flop: sum(g, 'flop'), hbm: sum(g, 'hbm'), sol: sum(g, 'sol'), bytes: sum(g, 'b'), hbmAfter: sum(g, 'hbmAfter'), solAfter: sum(g, 'solAfter'),
        // hosts whose epilogue traffic outgrows their tensor-core time, and the time that sticks out
        over: g.filter((e) => e.hbmAfter > e.hbm && e.hbmAfter > e.flop).map((e) => e.i), exposed: g.reduce((t, e) => t + (e.hbmAfter > e.hbm ? Math.max(0, e.hbmAfter - Math.max(e.flop, e.hbm)) : 0), 0) },
      other: { trace: sum(o, 'raw'), traceNorm: sum(o, 'us'), hbm: sum(o, 'hbm'), bytes: sum(o, 'b'), bytesAfter: sum(o, 'bAfter'), hbmAfter: sum(o, 'hbmAfter'),
        gone: o.filter((e) => e.gone && !e.aux).length, aux: o.filter((e) => e.aux).length, kept: o.filter((e) => !e.gone).length },
    };
    const t = T[pass];
    t.other.eff = t.other.trace * 1e-6 / t.other.hbm;                       // how far above its HBM floor the non-GEMM work runs today
    t.other.projected = t.other.hbmAfter * t.other.eff;              // the remaining floor at today's efficiency
  }
  return T;
}
export { SPEC };
