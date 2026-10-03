// Every kernel launch of one DeepSeek-V3 MoE layer, priced on both meters and
// wired by data dependency.
// src/kerndata.js carries the 240 launches of one steady-state layer cycle of
// DeepSeek's EP64 profile (traced µs, grids, names); this module says what
// each launch IS — which tensors it reads (`in`) and writes (`out`), in which
// dtype, through which GEMM — and prices it at speed of light: FLOPs ÷ the
// measured GEMM rate against bytes ÷ HBM bandwidth, the longer meter binding
// (03's conventions: the FP8/BF16 rates are 03's measured π^sol, HBM is the
// data-sheet 3.35 TB/s). Then it says where each launch goes under CODA
// (arXiv:2605.19269): into which GEMM's epilogue, into an auxiliary reduction
// over tile partials, into the all-to-all kernels, or nowhere (it stays).
// Operands are read off kernel names, template arguments and grid sizes;
// where a shape is inferred from the grid alone the spec says so (`inf:
// true`) and the page marks it.
//
// The tensor names are the accounting: graph(pass) joins producers to
// consumers by name, and the module refuses to load if a launch reads a
// tensor nothing produces (in this pass, an earlier pass, or the declared
// layer inputs / weights), writes one nothing reads (here, in a later pass,
// or the declared layer outputs), or folds under CODA into a host it is not
// wired to. Node-importable (goldens pin the per-pass totals).
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
// lane (the figure's columns): 0 q · 1 kv — the attention column; 2 routed
// (router, DeepEP, the routed experts) · 3 shared — the MoE column. Within a
// lane launches stack in trace order (`seq` overrides, for the comm stream).
// in / out: the tensors a launch reads and writes, by name — the wires.
// how (CODA): epilogue (into `host`'s epilogue) · aux (a reduction over tile
// partials; `after` = its traffic) · fa (into FlashAttention's own kernels) ·
// a2a (into DeepEP's kernels) · cache (a weight transpose that need not recur
// per microbatch) · stash (replaced by a stash the forward epilogue writes) ·
// keep (unchanged). `add` on a host GEMM = HBM traffic its epilogue takes on;
// `out` on a host = the change in its output bytes.
const SPEC = [];
const put = (i, re, e) => {
  const o = LAUNCHES[i];
  if (!o || !re.test(o.name)) throw new Error(`launch ${i} is ${o?.k}, spec expected ${re}`);
  SPEC[i] = { i, in: [], out: [], ...e };
};
const GEMM = /GemmKernel|sm90_xmma_gemm/, CAST = /per_token_cast_to_fp8_with_channels/, ST = /utils::transpose/, LN = /ln_fwd_kernel/, LNB = /ln_bwd_kernel/, LNF = /ln_bwd_finalize/, CT = /per_channel_cast_and_transpose/, BT = /batched_transpose/, CB = /cast_back_impl/, EW = /elementwise|reduce_kernel|FillFunctor|scatter_gather|_copy_tensor|arange|Distribution|topk|unique_group|aux_fi/, ANY = /./;

// ======================= forward: attention sub-layer (attn(F)) =======================
put(0, LN, { lab: 'residual add + input RMSNorm', lane: 0, op: 'input norm', kind: 'op', b: normRes(S, D), in: ['x', 'y (prev. layer)'], out: ['h₁', 'n₁', 'r₁'],
  how: 'a2a', host: 'combine', after: bf(S * D) + f8(S * D), note: 'the previous layer\'s combine kernel assembles each token\'s full row, so it can add the residual, take the exact row RMS, and emit x ⊙ γ in FP8 with r folded into the scales' });
put(1, CAST, { lab: 'quantize the normed input → FP8 (for Q down)', lane: 0, op: 'Q down', kind: 'op', b: cast(S * D), in: ['n₁'], out: ['n₁ fp8'], how: 'a2a', host: 'combine', after: 0, note: 'emitted by the combine epilogue' });
put(2, ST, { lab: 'transpose its scales', lane: 0, op: 'Q down', kind: 'glue', b: scaleT(S * D), in: ['n₁ fp8'], out: ['n₁ scalesᵀ'], how: 'a2a', host: 'combine', after: 0 });
put(3, GEMM, { lab: 'Q down projection', lane: 0, op: 'Q down', kind: 'gemm', dt: 'fp8', ...gemm(S, QR, D), in: ['n₁ fp8', 'n₁ scalesᵀ'], out: ['q latent'], epi: ['⊙ γ_q', 'partial RMS → r_q', 'FP8 out'], add: f32(S * QR / 128) });
put(4, CAST, { lab: 'quantize the normed input → FP8 (again, for KV down)', lane: 1, op: 'KV down', kind: 'op', b: cast(S * D), in: ['n₁'], out: ['n₁ fp8 (kv)'], how: 'a2a', host: 'combine', after: 0, note: 'the same tensor quantized twice' });
put(5, ST, { lab: 'transpose its scales', lane: 1, op: 'KV down', kind: 'glue', b: scaleT(S * D), in: ['n₁ fp8 (kv)'], out: ['n₁ scalesᵀ (kv)'], how: 'a2a', host: 'combine', after: 0 });
put(6, GEMM, { lab: 'KV down projection', lane: 1, op: 'KV down', kind: 'gemm', dt: 'fp8', ...gemm(S, KVP, D), in: ['n₁ fp8 (kv)', 'n₁ scalesᵀ (kv)'], out: ['kv latent ‖ k_rope'], epi: ['⊙ γ_kv', 'partial RMS → r_kv', 'RoPE(k_rope)', 'FP8 out'], add: f32(S * KVR / 128) });
put(7, EW, { lab: 'position indices (arange)', lane: 1, op: 'RoPE', kind: 'glue', b: 0, out: ['pos (k)'], how: 'keep', note: 'a constant per step' });
put(8, EW, { lab: 'position indices (arange)', lane: 0, op: 'RoPE', kind: 'glue', b: 0, out: ['pos (q)'], how: 'keep' });
put(9, LN, { lab: 'q-latent RMSNorm', lane: 0, op: 'q norm', kind: 'op', b: norm(S, QR), in: ['q latent'], out: ['n_q', 'r_q'], how: 'aux', after: 2 * f32(S * QR / 128) + f32(S), note: 'r_q from the Q-down epilogue\'s 12 partials per row; applied in the Q-up epilogue' });
put(10, EW, { lab: 'layout copy (q latent)', lane: 0, op: 'q norm', kind: 'op', b: copy(S * QR), in: ['n_q'], out: ['n_q (contiguous)'], how: 'epilogue', host: 3, after: 0 });
put(11, LN, { lab: 'kv-latent RMSNorm', lane: 1, op: 'kv norm', kind: 'op', b: norm(S, KVR), in: ['kv latent ‖ k_rope'], out: ['n_kv', 'r_kv'], how: 'aux', after: 2 * f32(S * KVR / 128) + f32(S) });
put(12, EW, { lab: 'layout copy (k_rope slice)', lane: 1, op: 'kv norm', kind: 'op', b: copy(S * RO), in: ['kv latent ‖ k_rope'], out: ['k_rope (contiguous)'], how: 'epilogue', host: 6, after: 0, inf: true });
put(13, CAST, { lab: 'quantize q latent → FP8', lane: 0, op: 'Q up', kind: 'op', b: cast(S * QR), in: ['n_q (contiguous)'], out: ['n_q fp8'], how: 'epilogue', host: 3, after: 0 });
put(14, ST, { lab: 'transpose its scales', lane: 0, op: 'Q up', kind: 'glue', b: scaleT(S * QR), in: ['n_q fp8'], out: ['n_q scalesᵀ'], how: 'epilogue', host: 3, after: 0 });
put(15, GEMM, { lab: 'Q up projection', lane: 0, op: 'Q up', kind: 'gemm', dt: 'fp8', ...gemm(S, QH, QR), in: ['n_q fp8', 'n_q scalesᵀ'], out: ['q (pre-RoPE)'], epi: ['× r_q', 'RoPE(q_rope)', 'Q in FA layout'] });
put(16, CAST, { lab: 'quantize kv latent → FP8', lane: 1, op: 'KV up', kind: 'op', b: cast(S * KVR), in: ['n_kv'], out: ['n_kv fp8'], how: 'epilogue', host: 6, after: 0 });
put(17, ST, { lab: 'transpose its scales', lane: 1, op: 'KV up', kind: 'glue', b: scaleT(S * KVR), in: ['n_kv fp8'], out: ['n_kv scalesᵀ'], how: 'epilogue', host: 6, after: 0 });
put(18, GEMM, { lab: 'KV up projection', lane: 1, op: 'KV up', kind: 'gemm', dt: 'fp8', ...gemm(S, KVH, KVR), in: ['n_kv fp8', 'n_kv scalesᵀ'], out: ['k_nope ‖ v'], epi: ['× r_kv', 'K, V in FA layout'] });
put(19, /rotary/, { lab: 'RoPE on k_rope', lane: 1, op: 'RoPE', kind: 'op', b: rope(S * RO), in: ['k_rope (contiguous)', 'pos (k)'], out: ['k_rope'], how: 'epilogue', host: 6, after: 0 });
put(20, /_copy_tensor/, { lab: 'copy q_nope into the Q buffer', lane: 0, op: 'RoPE', kind: 'op', b: copy(S * H * NO), in: ['q (pre-RoPE)'], out: ['Q (nope half)'], how: 'epilogue', host: 15, after: 0, inf: true });
put(21, EW, { lab: 'copy q_rope out', lane: 0, op: 'RoPE', kind: 'op', b: copy(S * H * RO), in: ['q (pre-RoPE)'], out: ['q_rope'], how: 'epilogue', host: 15, after: 0, inf: true });
put(22, /rotary/, { lab: 'RoPE on q_rope (128 heads)', lane: 0, op: 'RoPE', kind: 'op', b: rope(S * H * RO), in: ['q_rope', 'pos (q)'], out: ['Q (rope half)'], how: 'epilogue', host: 15, after: 0 });
put(23, /compute_attn/, { lab: 'FlashAttention forward', lane: 0, op: 'attention', kind: 'attn', dt: 'bf16', f: S * S * H * (NO + RO + DV),
  b: bf(S * QH) * 2 + bf(S * O) * 2 + f32(S * H), in: ['Q (nope half)', 'Q (rope half)', 'k_nope ‖ v', 'k_rope'], out: ['attn out', 'lse'], how: 'keep', note: 'its own fused kernel; the FP8 and E5M6 casts of its output could be its epilogue (tile-local: one head = one 128-block)' });
put(24, /special_fmt/, { lab: 'stash the attention output as E5M6', lane: 0, op: 'O proj', kind: 'op', b: bf(S * O) + e5m6(S * O), in: ['attn out'], out: ['attn out e5m6'], how: 'fa', after: e5m6(S * O), note: 'written from FlashAttention\'s output tile instead of re-read' });
put(25, CAST, { lab: 'quantize the attention output → FP8', lane: 0, op: 'O proj', kind: 'op', b: cast(S * O), in: ['attn out'], out: ['attn out fp8'], how: 'fa', after: f8(S * O) });
put(26, ST, { lab: 'transpose its scales', lane: 0, op: 'O proj', kind: 'glue', b: scaleT(S * O), in: ['attn out fp8'], out: ['attn out scalesᵀ'], how: 'fa', after: 0 });
put(27, GEMM, { lab: 'attention output projection', lane: 0, op: 'O proj', kind: 'gemm', dt: 'fp8', ...gemm(S, D, O), in: ['attn out fp8', 'attn out scalesᵀ'], out: ['o'],
  epi: ['× c', '+ residual', '⊙ γ', 'partial RMS → r', 'FP8 out (+ bf16 for the router)'], add: 2 * bf(S * D) + f8(S * D) });
put(28, EW, { lab: 'scale the projection output (× constant)', lane: 0, op: 'O proj', kind: 'op', b: 2 * bf(S * D), in: ['o'], out: ['o′'], how: 'epilogue', host: 27, after: 0, inf: true });

// ======================= forward: MoE sub-layer (mlp(F)) =======================
put(29, LN, { lab: 'residual add + MoE-input RMSNorm', lane: 2, op: 'MoE norm', kind: 'op', b: normRes(S, D), in: ['o′', 'h₁'], out: ['h₂', 'n₂', 'r₂'], how: 'epilogue', host: 27, after: 2 * f32(S * D / 128) + f32(S), note: 'residual and γ in the epilogue; r from a reduction over 56 partials per row' });
put(30, GEMM, { lab: 'router logits', lane: 2, op: 'router', kind: 'gemm', dt: 'bf16', ...gemm(S, NE, D, { ab: 2, w: 2, o: 4 }), in: ['n₂'], out: ['logits'], epi: ['× r', 'sigmoid', '+ bias'] });
// the router's glue: a chain of tiny PyTorch launches; each reads the previous one's output unless it says otherwise
let chain = null;
const RG = (i, lab, e = {}) => { put(i, ANY, { lab, lane: 2, op: 'router', kind: 'glue', b: ew(LAUNCHES[i].grid[0] * LAUNCHES[i].grid[1], LAUNCHES[i].block[0], 8), how: 'keep', inf: true, in: chain ? [chain] : [], out: [`router state ${i}`], ...e }); chain = SPEC[i].out[0]; };
RG(31, 'random normal (router noise)', { in: [], out: ['noise'] }); RG(32, 'sigmoid', { in: ['logits', 'noise'], out: ['scores'], how: 'epilogue', host: 30, after: 0 }); RG(33, 'row sum', { out: ['score sums'] }); RG(34, 'divide', { in: ['scores', 'score sums'], out: ['probs'] });
RG(35, 'add the correction bias', { in: ['scores'], out: ['biased probs'], how: 'epilogue', host: 30, after: 0 }); RG(36, 'group scores (top-2 sum per group)', { out: ['group scores'] });
RG(37, 'fill mask', { in: [], out: ['group mask (zeroed)'] }); RG(38, 'scatter group mask', { in: ['group scores', 'group mask (zeroed)'], out: ['group mask'] }); RG(39, 'copy', { out: ['group mask (copy)'] }); RG(40, 'masked fill', { in: ['biased probs', 'group mask (copy)'], out: ['masked probs'] });
put(41, /gatherTopK/, { lab: 'top-8 experts per token (sort)', lane: 2, op: 'router', kind: 'op', b: ew(4096, 256, 8), in: ['masked probs'], out: ['top-k idx'], how: 'keep', inf: true }); chain = 'top-k idx';
RG(42, 'floor-divide', { out: ['group idx'] }); RG(43, 'unique group indices', { out: ['unique groups'] }); RG(44, 'gather', { in: ['probs', 'top-k idx'], out: ['top-k w'] });
chain = 'top-k idx';
for (const [i, l] of [[45, 'fill'], [46, 'fill'], [47, 'add'], [48, 'scatter-add'], [49, 'compare'], [50, 'copy'], [51, 'reduce'], [52, 'multiply'], [53, 'add'], [54, 'add'], [55, 'reduce'], [56, 'add'], [57, 'divide']]) RG(i, `load-balance statistics: ${l}`, i === 45 ? { in: ['top-k idx', 'probs'] } : {});
put(58, /gatherTopK/, { lab: 'load-balance statistics: top-k (sort)', lane: 2, op: 'router', kind: 'op', b: ew(4096, 256, 8), in: [chain], out: ['router state 58'], how: 'keep', inf: true }); chain = 'router state 58';
RG(59, 'load-balance statistics: reduce'); RG(60, 'load-balance statistics: aux_fi'); RG(61, 'load-balance statistics: multiply'); RG(62, 'load-balance statistics: reduce'); RG(63, 'load-balance statistics: multiply', { out: ['aux-loss stats'] });
put(64, CAST, { lab: 'quantize the MoE input → FP8 (dispatch + shared expert)', lane: 3, op: 'dispatch', kind: 'op', b: cast(S * D), in: ['n₂'], out: ['n₂ fp8'], how: 'epilogue', host: 27, after: 0, note: 'the O-proj epilogue emits x ⊙ γ in FP8; r rides the per-token scales through the all-to-all' });
put(102, ST, { lab: 'transpose scales', lane: 3, op: 'shared FC13', kind: 'glue', b: scaleT(S * D), in: ['n₂ fp8'], out: ['n₂ scalesᵀ'], how: 'epilogue', host: 27, after: 0 });
put(103, GEMM, { lab: 'shared expert gate/up', lane: 3, op: 'shared FC13', kind: 'gemm', dt: 'fp8', ...gemm(S, 2 * I, D), in: ['n₂ fp8', 'n₂ scalesᵀ'], out: ['gu_sh'], epi: ['× r', 'SwiGLU (pairwise)', 'FP8 out + FP8 stash'], out8: f8(S * I) + f8(S * 2 * I) - bf(S * 2 * I) });
put(104, /clean_and_count_expert/, { lab: 'count tokens per expert', lane: 2, op: 'pack', kind: 'glue', b: ew(4, 512, 16), in: ['recv idx'], out: ['expert counts'], how: 'a2a', host: 'dispatch', after: 0, note: 'DeepEP bookkeeping, 4 blocks: latency, not bytes', inf: true });
put(105, /get_fused_mapping/, { lab: 'build the token → expert-row mapping', lane: 2, op: 'pack', kind: 'glue', b: ew(4, 512, 16), in: ['recv idx', 'expert counts'], out: ['row map'], how: 'a2a', host: 'dispatch', after: 0, note: '4 blocks: latency, not bytes', inf: true });
put(106, /expand_to_fused/, { lab: 'expand received tokens into expert rows (FP8 + scales)', lane: 2, op: 'pack', kind: 'op', b: 2 * f8(R * D), in: ['recv tokens', 'row map'], out: ['X_e'], how: 'keep', note: 'not tile-local; the dispatch kernel could land tokens in expert order directly, which is a DeepEP change, not an epilogue — counted as staying' });
put(107, BT, { lab: 'transpose the 4 local W13 (FP8)', lane: 2, op: 'pack', kind: 'op', b: wT(EL * 2 * I * D), in: ['W13'], out: ['W13ᵀ'], how: 'cache', after: 0, note: 'weights are constant for the step: transpose once, not per microbatch' });
put(108, EW, { lab: 'copy the scales', lane: 2, op: 'pack', kind: 'glue', b: 2 * f32(R * D / 128), in: ['X_e'], out: ['X_e scales′'], how: 'a2a', host: 'dispatch', after: 0, inf: true });
put(109, GEMM, { lab: 'routed experts gate/up (grouped)', lane: 2, op: 'routed FC13', kind: 'gemm', dt: 'fp8', ...gemm(R, 2 * I, D, { groups: EL }), in: ['X_e', 'X_e scales′', 'W13ᵀ'], out: ['gu_r'], epi: ['× r (in the scales)', 'SwiGLU (pairwise)', '× top-k weight', 'FP8 out + FP8 stash'], out8: f8(R * I) + f8(R * 2 * I) - bf(R * 2 * I), add: f32(R) });
put(110, EW, { lab: 'scale fixup', lane: 2, op: 'routed FC13', kind: 'glue', b: ew(453, 128, 8), in: ['gu_r'], out: ['gu_r′'], how: 'epilogue', host: 109, after: 0, inf: true });
put(111, /swiglu_forward_with_weight/, { lab: 'SwiGLU × routing weight → FP8', lane: 2, op: 'routed SwiGLU', kind: 'op', b: bf(R * 2 * I) + f8(R * I) + f32(R), in: ['gu_r′', 'recv top-k w'], out: ['s_r fp8'], how: 'epilogue', host: 109, after: 0, note: 'already one fused kernel; under CODA the gate/up intermediate is never written' });
put(112, BT, { lab: 'transpose the 4 local W2 (FP8)', lane: 2, op: 'routed FC2', kind: 'op', b: wT(EL * I * D), in: ['W2'], out: ['W2ᵀ'], how: 'cache', after: 0 });
put(113, EW, { lab: 'copy the scales', lane: 2, op: 'routed FC2', kind: 'glue', b: 2 * f32(R * I / 128), in: ['s_r fp8'], out: ['s_r scales′'], how: 'epilogue', host: 109, after: 0, inf: true });
put(114, GEMM, { lab: 'routed experts down (grouped)', lane: 2, op: 'routed FC2', kind: 'gemm', dt: 'fp8', ...gemm(R, D, I, { groups: EL }), in: ['s_r fp8', 's_r scales′', 'W2ᵀ'], out: ['Y_e'], epi: [] });
put(115, /reduce_fused/, { lab: 'sum each token\'s local expert copies', lane: 2, op: 'combine', kind: 'op', b: 2 * bf(R * D), in: ['Y_e', 'row map'], out: ['y (per token)'], how: 'keep', note: 'a cross-row reduction, not tile-local; the combine kernel reads these rows anyway and could sum them — a DeepEP change, counted as staying' });
put(116, /swiglu_forward_with_weight/, { lab: 'shared SwiGLU → FP8', lane: 3, op: 'shared SwiGLU', kind: 'op', b: bf(S * 2 * I) + f8(S * I), in: ['gu_sh'], out: ['s_sh fp8'], how: 'epilogue', host: 103, after: 0 });
put(117, ST, { lab: 'transpose scales', lane: 3, op: 'shared FC2', kind: 'glue', b: scaleT(S * I), in: ['s_sh fp8'], out: ['s_sh scalesᵀ'], how: 'epilogue', host: 103, after: 0 });
put(118, GEMM, { lab: 'shared expert down', lane: 3, op: 'shared FC2', kind: 'gemm', dt: 'fp8', ...gemm(S, D, I), in: ['s_sh fp8', 's_sh scalesᵀ'], out: ['y_sh'], epi: ['+ residual (x after attention)'], add: bf(S * D) });
put(119, CAST, { lab: 'quantize the routed gate/up → FP8 (the SwiGLU-input stash)', lane: 2, op: 'stash', kind: 'op', b: cast(R * 2 * I), in: ['gu_r′'], out: ['gu_r fp8 (stash)'], how: 'epilogue', host: 109, after: 0, inf: true,
  note: 'DeepSeek caches the SwiGLU inputs in FP8 (§3.3.3) and the backward\'s dequantize reads exactly this tensor; grid = routed rows, and the time fits a 4,096-wide cast at ×1.5 its floor' });
put(120, CAST, { lab: 'quantize the shared gate/up → FP8 (the SwiGLU-input stash)', lane: 3, op: 'stash', kind: 'op', b: cast(S * 2 * I), in: ['gu_sh'], out: ['gu_sh fp8 (stash)'], how: 'epilogue', host: 103, after: 0, inf: true });

// ======================= backward: MoE sub-layer (mlp(B)) =======================
put(65, CB, { lab: 'dequantize the shared gate/up stash (FP8 → bf16)', lane: 3, op: 'shared FC2 bwd', kind: 'op', b: castBack(S * 2 * I), in: ['gu_sh fp8 (stash)'], out: ['gu_sh (dequantized)'], how: 'epilogue', host: 68, after: 0, note: 'the SwiGLU-backward epilogue reads the FP8 stash tile directly' });
put(66, BT, { lab: 'transpose the shared W2 (FP8)', lane: 3, op: 'shared FC2 bwd', kind: 'op', b: wT(I * D), in: ['W2 (shared)'], out: ['W2ᵀ (shared)'], how: 'cache', after: 0 });
put(67, ST, { lab: 'transpose scales', lane: 3, op: 'shared FC2 bwd', kind: 'glue', b: scaleT(S * D), in: ['dy fp8'], out: ['dy scalesᵀ'], how: 'epilogue', host: 'grad', after: 0 });
put(68, GEMM, { lab: 'shared FC2: input gradient', lane: 3, op: 'shared FC2 bwd', kind: 'gemm', dt: 'fp8', ...gemm(S, I, D), in: ['dy fp8', 'dy scalesᵀ', 'W2ᵀ (shared)'], out: ['d(s_sh)'], epi: ['SwiGLU backward (pairwise)', 'd(gate/up) → FP8, both layouts'], add: f8(S * 2 * I) + 2 * f8(S * 2 * I) - bf(S * I) });
put(69, CT, { lab: 'd(shared out) → FP8ᵀ (weight-gradient operand)', lane: 3, op: 'shared FC2 bwd', kind: 'op', b: cast(S * D), in: ['dy'], out: ['dy fp8ᵀ'], how: 'epilogue', host: 'grad', after: 0, note: 'the epilogue that produces this gradient can write both scale layouts' });
put(70, /swiglu_and_per_channel/, { lab: 'recompute SwiGLU → FP8ᵀ (weight-gradient operand)', lane: 3, op: 'shared SwiGLU bwd', kind: 'op', b: bf(S * 2 * I) + f8(S * I), in: ['gu_sh (dequantized)'], out: ['s_sh fp8ᵀ'], how: 'epilogue', host: 68, after: 0 });
put(71, /swiglu_backward/, { lab: 'SwiGLU backward', lane: 3, op: 'shared SwiGLU bwd', kind: 'op', b: bf(S * 2 * I) * 2 + bf(S * I), in: ['gu_sh (dequantized)', 'd(s_sh)'], out: ['d(gu_sh)'], how: 'epilogue', host: 68, after: 0 });
put(72, CB, { lab: 'dequantize the routed gate/up stash (FP8 → bf16)', lane: 2, op: 'routed SwiGLU bwd', kind: 'op', b: castBack(R * 2 * I), in: ['gu_r fp8 (stash)'], out: ['gu_r (dequantized)'], how: 'epilogue', host: 75, after: 0, inf: true, note: '112 blocks for a 400 MB pass: 1.4× its floor' });
put(73, /expand_to_fused/, { lab: 'expand the received output gradients into expert rows', lane: 2, op: 'unpack', kind: 'op', b: 2 * f8(R * D), in: ['recv d(y)', 'row map'], out: ['d(Y_e)'], how: 'keep', note: 'not tile-local (see the forward expand)' });
put(74, EW, { lab: 'copy the scales', lane: 2, op: 'unpack', kind: 'glue', b: 2 * f32(R * D / 128), in: ['d(Y_e)'], out: ['d(Y_e) scales′'], how: 'a2a', host: 'dispatch', after: 0, inf: true });
put(75, GEMM, { lab: 'routed FC2: input gradient (grouped)', lane: 2, op: 'routed FC2 bwd', kind: 'gemm', dt: 'fp8', ...gemm(R, I, D, { groups: EL }), in: ['d(Y_e)', 'd(Y_e) scales′'], out: ['d(s_r)'],
  epi: ['SwiGLU backward (pairwise)', '× top-k weight', 'partial Σ for d(weight)', 'd(gate/up) → FP8, both layouts'], add: f8(R * 2 * I) + 2 * f8(R * 2 * I) - bf(R * I) });
put(76, /swiglu_forward_backward_with_weight/, { lab: 'SwiGLU forward + backward × routing weight', lane: 2, op: 'routed SwiGLU bwd', kind: 'op', b: bf(R * 2 * I) * 2 + bf(R * I) + f32(R), in: ['gu_r (dequantized)', 'd(s_r)', 'recv top-k w'], out: ['d(gu_r)', 's_r (recomputed)'], how: 'epilogue', host: 75, after: 0 });
put(77, /swiglu_weight_backward/, { lab: 'routing-weight gradient (row dot products)', lane: 2, op: 'routed SwiGLU bwd', kind: 'op', b: bf(R * I) * 2 + f32(R), in: ['s_r (recomputed)', 'd(s_r)'], out: ['d(top-k w) (routed rows)'], how: 'aux', host: 75, after: f32(R * 16) * 2, note: 'a row reduction: tile partials from the FC2-backward epilogue, summed by a small kernel' });
put(78, CT, { lab: 'SwiGLU output → FP8ᵀ (weight-gradient operand)', lane: 2, op: 'routed SwiGLU bwd', kind: 'op', b: bf(R * I) + f8(R * I), in: ['s_r (recomputed)'], out: ['s_r fp8ᵀ'], how: 'epilogue', host: 75, after: 0 });
put(79, EW, { lab: 'scale fixup', lane: 2, op: 'routed SwiGLU bwd', kind: 'glue', b: ew(502, 128, 8), in: ['d(gu_r)'], out: ['d(gu_r)′'], how: 'epilogue', host: 75, after: 0, inf: true });
put(80, LN, { lab: 'recompute: residual add + MoE-input RMSNorm', lane: 3, op: 'recompute', kind: 'op', b: normRes(S, D), in: ['o′', 'h₁'], out: ['n₂ (recomputed)', 'h₂ (recomputed)'], how: 'epilogue', host: 91, after: 0, note: 'the RMSNorm-backward epilogue reads the saved pre-norm row (h₁) and r instead of recomputing the norm' });
put(81, CAST, { lab: 'quantize the recomputed MoE input → FP8', lane: 3, op: 'shared FC13 bwd', kind: 'op', b: cast(S * D), in: ['n₂ (recomputed)'], out: ['n₂ fp8 (recomputed)'], how: 'stash', after: 0, inf: true, note: 'the forward O-proj epilogue already wrote this tensor in FP8' });
put(82, CAST, { lab: 'quantize d(gate/up) → FP8 (R × 4096)', lane: 2, op: 'routed FC13 bwd', kind: 'op', b: cast(R * 2 * I), in: ['d(gu_r)′'], out: ['d(gu_r) fp8'], how: 'epilogue', host: 75, after: 0 });
put(83, EW, { lab: 'copy the scales', lane: 2, op: 'routed FC13 bwd', kind: 'glue', b: 2 * f32(R * 2 * I / 128), in: ['d(gu_r) fp8'], out: ['d(gu_r) scales′'], how: 'epilogue', host: 75, after: 0, inf: true });
put(84, GEMM, { lab: 'routed FC13: input gradient (grouped)', lane: 2, op: 'routed FC13 bwd', kind: 'gemm', dt: 'fp8', ...gemm(R, D, 2 * I, { groups: EL }), in: ['d(gu_r) fp8', 'd(gu_r) scales′'], out: ['d(X_e)'], epi: ['partial Σ x ⊙ dx → ŝ (rides the combine)'], add: f32(R) });
put(85, /reduce_fused/, { lab: 'sum each token\'s local input-gradient copies', lane: 2, op: 'unpack', kind: 'op', b: 2 * bf(R * D), in: ['d(X_e)', 'row map'], out: ['d(x) (per token)'], how: 'keep', note: 'not tile-local (see the forward reduction)' });
put(86, CT, { lab: 'd(gate/up) → FP8ᵀ (weight-gradient operand)', lane: 2, op: 'routed FC13 bwd', kind: 'op', b: cast(R * 2 * I), in: ['d(gu_r)′'], out: ['d(gu_r) fp8ᵀ'], how: 'epilogue', host: 75, after: 0 });
put(87, CB, { lab: 'dequantize the MoE input (FP8 → bf16)', lane: 3, op: 'shared FC13 bwd', kind: 'op', b: castBack(S * D), in: ['n₂ fp8 (recomputed)'], out: ['n₂ (FP8-rounded)'], how: 'stash', after: 0, inf: true, note: 'a quantize–dequantize round trip: the weight-gradient operand is re-cast from the FP8-rounded value' });
put(88, BT, { lab: 'transpose the shared W13 (FP8)', lane: 3, op: 'shared FC13 bwd', kind: 'op', b: wT(2 * I * D), in: ['W13 (shared)'], out: ['W13ᵀ (shared)'], how: 'cache', after: 0 });
put(89, CAST, { lab: 'quantize d(gate/up) → FP8 (S × 4096)', lane: 3, op: 'shared FC13 bwd', kind: 'op', b: cast(S * 2 * I), in: ['d(gu_sh)'], out: ['d(gu_sh) fp8'], how: 'epilogue', host: 68, after: 0 });
put(90, ST, { lab: 'transpose scales', lane: 3, op: 'shared FC13 bwd', kind: 'glue', b: scaleT(S * 2 * I), in: ['d(gu_sh) fp8'], out: ['d(gu_sh) scalesᵀ'], how: 'epilogue', host: 68, after: 0 });
put(91, GEMM, { lab: 'shared FC13: input gradient', lane: 3, op: 'shared FC13 bwd', kind: 'gemm', dt: 'fp8', ...gemm(S, D, 2 * I), in: ['d(gu_sh) fp8', 'd(gu_sh) scalesᵀ', 'W13ᵀ (shared)'], out: ['d(n₂) shared'],
  epi: ['+ router and routed d(input)', 'RMSNorm backward (needs s, r)', 'partial dγ'], add: 2 * bf(S * D) + bf(S * D) });
put(92, CT, { lab: 'd(gate/up) → FP8ᵀ (weight-gradient operand)', lane: 3, op: 'shared FC13 bwd', kind: 'op', b: cast(S * 2 * I), in: ['d(gu_sh)'], out: ['d(gu_sh) fp8ᵀ'], how: 'epilogue', host: 68, after: 0 });
put(93, CT, { lab: 'MoE input → FP8ᵀ (weight-gradient operand)', lane: 3, op: 'shared FC13 bwd', kind: 'op', b: cast(S * D), in: ['n₂ (FP8-rounded)'], out: ['n₂ fp8ᵀ'], how: 'stash', after: 0, note: 'the forward epilogue can stash both scale layouts (+29 MB of stash per layer)' });

// ======================= weight gradients (mlp(W), attn(W)) =======================
const WG = (i, lab, lane, op, g, ins, outs) => put(i, GEMM, { lab, lane, op, kind: 'gemm', dt: 'fp8', ...g, in: ins, out: outs });
WG(94, 'shared FC2: weight gradient', 2, 'shared FC2 wgrad', gemm(I, D, S, { acc: true }), ['s_sh fp8ᵀ', 'dy fp8ᵀ'], ['dW2 (shared)']);
put(95, CT, { lab: 'd(routed out) → FP8ᵀ (R × 7168)', lane: 2, op: 'routed FC2 wgrad', kind: 'op', b: requantT(R * D), in: ['d(Y_e)'], out: ['d(Y_e) fp8ᵀ'], how: 'a2a', host: 'dispatch', after: 0, note: 'the backward dispatch could deliver the gradient in both scale layouts' });
WG(96, 'routed FC2: weight gradient, expert 1', 2, 'routed FC2 wgrad', gemm(I, D, R / EL, { acc: true }), ['s_r fp8ᵀ', 'd(Y_e) fp8ᵀ'], ['dW2 (expert 1)']);
WG(97, 'routed FC2: weight gradient, expert 2', 2, 'routed FC2 wgrad', gemm(I, D, R / EL, { acc: true }), ['s_r fp8ᵀ', 'd(Y_e) fp8ᵀ'], ['dW2 (expert 2)']);
put(98, CT, { lab: 'dispatched input → FP8ᵀ (R × 7168)', lane: 2, op: 'routed FC13 wgrad', kind: 'op', b: requantT(R * D), in: ['X_e'], out: ['X_e fp8ᵀ'], how: 'a2a', host: 'dispatch', after: 0, note: 'the forward dispatch could stash both scale layouts' });
WG(99, 'routed FC13: weight gradient, expert 1', 2, 'routed FC13 wgrad', gemm(2 * I, D, R / EL, { acc: true }), ['X_e fp8ᵀ', 'd(gu_r) fp8ᵀ'], ['dW13 (expert 1)']);
WG(100, 'routed FC13: weight gradient, expert 2', 2, 'routed FC13 wgrad', gemm(2 * I, D, R / EL, { acc: true }), ['X_e fp8ᵀ', 'd(gu_r) fp8ᵀ'], ['dW13 (expert 2)']);
WG(101, 'shared FC13: weight gradient', 2, 'shared FC13 wgrad', gemm(2 * I, D, S, { acc: true }), ['n₂ fp8ᵀ', 'd(gu_sh) fp8ᵀ'], ['dW13 (shared)']);
WG(233, 'routed FC2: weight gradient, expert 3 (2nd stream)', 2, 'routed FC2 wgrad', gemm(I, D, R / EL, { acc: true }), ['s_r fp8ᵀ', 'd(Y_e) fp8ᵀ'], ['dW2 (expert 3)']);
WG(235, 'routed FC2: weight gradient, expert 4 (2nd stream)', 2, 'routed FC2 wgrad', gemm(I, D, R / EL, { acc: true }), ['s_r fp8ᵀ', 'd(Y_e) fp8ᵀ'], ['dW2 (expert 4)']);
WG(236, 'routed FC13: weight gradient, expert 3 (2nd stream)', 2, 'routed FC13 wgrad', gemm(2 * I, D, R / EL, { acc: true }), ['X_e fp8ᵀ', 'd(gu_r) fp8ᵀ'], ['dW13 (expert 3)']);
WG(237, 'routed FC13: weight gradient, expert 4 (2nd stream)', 2, 'routed FC13 wgrad', gemm(2 * I, D, R / EL, { acc: true }), ['X_e fp8ᵀ', 'd(gu_r) fp8ᵀ'], ['dW13 (expert 4)']);
WG(221, 'O proj: weight gradient', 0, 'O proj wgrad', gemm(O, D, S, { acc: true }), ['attn out fp8ᵀ', 'd(o) fp8ᵀ'], ['dW_o']);
WG(222, 'KV up: weight gradient', 0, 'KV up wgrad', gemm(KVR, KVH, S, { acc: true }), ['n_kv fp8ᵀ', 'dKV fp8ᵀ'], ['dW_kv_up']);
WG(223, 'Q up: weight gradient', 0, 'Q up wgrad', gemm(QR, QH, S, { acc: true }), ['n_q fp8ᵀ', 'dQ fp8ᵀ'], ['dW_q_up']);
WG(224, 'KV down: weight gradient', 0, 'KV down wgrad', gemm(D, KVP, S, { acc: true }), ['n₁ fp8ᵀ', 'd(kv ‖ k_rope) fp8ᵀ'], ['dW_kv_down']);
WG(225, 'Q down: weight gradient', 0, 'Q down wgrad', gemm(D, QR, S, { acc: true }), ['n₁ fp8ᵀ (again)', 'd(q latent) fp8ᵀ'], ['dW_q_down']);

// ======================= backward: attention sub-layer (attn(B)) =======================
// the router's backward: 25 tiny launches from d(top-k w) back to d(logits)
chain = null;
const RB = (i, lab, e = {}) => { put(i, ANY, { lab: `router backward: ${lab}`, lane: 3, op: 'router bwd', kind: 'glue', b: ew(LAUNCHES[i].grid[0], LAUNCHES[i].block[0], 8), how: 'keep', inf: true, in: chain ? [chain] : [], out: [`router bwd state ${i}`], ...e }); chain = SPEC[i].out[0]; };
for (const [i, l] of [[121, 'fill'], [122, 'reduce'], [123, 'multiply'], [124, 'multiply'], [125, 'fill'], [126, 'copy'], [127, 'divide'], [128, 'divide'], [129, 'negate'], [130, 'multiply'], [131, 'divide'], [132, 'reduce'], [133, 'add'], [134, 'fill'], [135, 'scatter-add'],
  [136, 'divide'], [137, 'divide'], [138, 'negate'], [139, 'multiply'], [140, 'divide'], [141, 'reduce'], [142, 'add'], [143, 'add'], [144, 'sigmoid backward'], [145, 'copy']]) RB(i, l, i === 121 ? { in: ['d(top-k w)', 'top-k idx', 'probs'] } : i === 145 ? { out: ['d(logits)'] } : {});
put(146, GEMM, { lab: 'router: input gradient', lane: 3, op: 'router bwd', kind: 'gemm', dt: 'bf16', ...gemm(S, D, NE, { ab: 2, w: 2, o: 2 }), in: ['d(logits)'], out: ['d(n₂) router'], epi: ['× r'] });
put(147, GEMM, { lab: 'router: weight gradient', lane: 3, op: 'router bwd', kind: 'gemm', dt: 'bf16', ...gemm(D, NE, S, { ab: 2, w: 2, o: 4 }), in: ['d(logits)', 'n₂ (recomputed)'], out: ['dW_router'] });
put(148, EW, { lab: 'accumulate d(MoE input) (+ router\'s)', lane: 3, op: 'MoE norm bwd', kind: 'op', b: 3 * bf(S * D), in: ['d(n₂) shared', 'd(n₂) router', 'd(n₂) routed'], out: ['d(n₂)'], how: 'epilogue', host: 91, after: 0, inf: true, note: 'one add of two tensors; the third of the three input gradients is accumulated inside a GEMM or the combine' });
put(149, LNB, { lab: 'MoE-input RMSNorm backward', lane: 3, op: 'MoE norm bwd', kind: 'op', b: normBwd(S, D, 132), in: ['d(n₂)', 'n₂ (recomputed)', 'r₂', 'dy'], out: ['d(x₁)', 'dγ₂ partials'], how: 'epilogue', host: 91, after: 0, note: 's for this norm = Σ over the router, shared and routed outputs ⊙ their gradients: the routed share is accumulated on other ranks and must ride the gradient combine' });
put(150, LNF, { lab: 'dγ finalize', lane: 3, op: 'MoE norm bwd', kind: 'glue', b: 2 * f32(D * 32), in: ['dγ₂ partials'], out: ['dγ₂'], how: 'aux', after: 2 * f32(D * 32) });
put(151, EW, { lab: 'dγ accumulate', lane: 3, op: 'MoE norm bwd', kind: 'glue', b: 3 * f32(D), in: ['dγ₂'], out: ['dγ₂ (accumulated)'], how: 'aux', after: 0 });
put(152, EW, { lab: 'scale backward (× constant)', lane: 0, op: 'O proj bwd', kind: 'op', b: 2 * bf(S * D), in: ['d(x₁)'], out: ['d(o)'], how: 'epilogue', host: 91, after: 0, inf: true });
put(153, /cast_back_special_fmt/, { lab: 'dequantize the E5M6 attention-output stash → bf16', lane: 0, op: 'O proj bwd', kind: 'op', b: e5m6(S * O) + bf(S * O), in: ['attn out e5m6'], out: ['attn out (dequantized)'], how: 'keep', after: e5m6(S * O) + f8(S * O), note: 'no GEMM produces this tensor in the backward: one pass E5M6 → FP8ᵀ remains (today it is three passes)' });
put(154, CAST, { lab: 'quantize d(attention out) → FP8', lane: 0, op: 'O proj bwd', kind: 'op', b: cast(S * D), in: ['d(o)'], out: ['d(o) fp8'], how: 'epilogue', host: 91, after: 0 });
put(155, BT, { lab: 'transpose W_o (FP8)', lane: 0, op: 'O proj bwd', kind: 'op', b: wT(O * D), in: ['W_o'], out: ['W_oᵀ'], how: 'cache', after: 0 });
put(156, ST, { lab: 'transpose scales', lane: 0, op: 'O proj bwd', kind: 'glue', b: scaleT(S * D), in: ['d(o) fp8'], out: ['d(o) scalesᵀ'], how: 'epilogue', host: 91, after: 0 });
put(157, GEMM, { lab: 'O proj: input gradient (dO)', lane: 0, op: 'O proj bwd', kind: 'gemm', dt: 'fp8', ...gemm(S, O, D), in: ['d(o) fp8', 'd(o) scalesᵀ', 'W_oᵀ'], out: ['d(attn out)'], epi: [] });
put(158, CT, { lab: 'attention output → FP8ᵀ (weight-gradient operand)', lane: 0, op: 'O proj bwd', kind: 'op', b: cast(S * O), in: ['attn out (dequantized)'], out: ['attn out fp8ᵀ'], how: 'keep', after: 0, note: 'folded into the one E5M6 → FP8ᵀ pass above' });
put(159, CT, { lab: 'd(attention out) → FP8ᵀ (weight-gradient operand)', lane: 0, op: 'O proj bwd', kind: 'op', b: cast(S * D), in: ['d(o)'], out: ['d(o) fp8ᵀ'], how: 'epilogue', host: 91, after: 0 });
put(160, LN, { lab: 'recompute: q-latent RMSNorm', lane: 0, op: 'recompute', kind: 'op', b: norm(S, QR), in: ['q latent'], out: ['n_q (recomputed)'], how: 'stash', after: 0, note: 'the forward Q-down epilogue\'s FP8 (latent ⊙ γ) and r_q are stashed instead of the bf16 latent: 3.7 MB less, no recompute' });
put(161, EW, { lab: 'recompute: layout copy', lane: 0, op: 'recompute', kind: 'op', b: copy(S * QR), in: ['n_q (recomputed)'], out: ['n_q (recomputed, contiguous)'], how: 'stash', after: 0 });
put(162, LN, { lab: 'recompute: kv-latent RMSNorm', lane: 1, op: 'recompute', kind: 'op', b: norm(S, KVR), in: ['kv latent ‖ k_rope'], out: ['n_kv (recomputed)'], how: 'stash', after: 0 });
put(163, CAST, { lab: 'recompute: quantize q latent → FP8', lane: 0, op: 'recompute', kind: 'op', b: cast(S * QR), in: ['n_q (recomputed, contiguous)'], out: ['n_q fp8 (recomputed)'], how: 'stash', after: 0 });
put(164, ST, { lab: 'transpose scales', lane: 0, op: 'recompute', kind: 'glue', b: scaleT(S * QR), in: ['n_q fp8 (recomputed)'], out: ['n_q scalesᵀ (recomputed)'], how: 'stash', after: 0 });
put(165, GEMM, { lab: 'recompute: Q up projection', lane: 0, op: 'recompute', kind: 'gemm', dt: 'fp8', ...gemm(S, QH, QR), in: ['n_q fp8 (recomputed)', 'n_q scalesᵀ (recomputed)'], out: ['q (pre-RoPE, recomputed)'], epi: ['× r_q', 'RoPE(q_rope)', 'Q in FA layout'] });
put(166, CAST, { lab: 'recompute: quantize kv latent → FP8', lane: 1, op: 'recompute', kind: 'op', b: cast(S * KVR), in: ['n_kv (recomputed)'], out: ['n_kv fp8 (recomputed)'], how: 'stash', after: 0 });
put(167, ST, { lab: 'transpose scales', lane: 1, op: 'recompute', kind: 'glue', b: scaleT(S * KVR), in: ['n_kv fp8 (recomputed)'], out: ['n_kv scalesᵀ (recomputed)'], how: 'stash', after: 0 });
put(168, GEMM, { lab: 'recompute: KV up projection', lane: 1, op: 'recompute', kind: 'gemm', dt: 'fp8', ...gemm(S, KVH, KVR), in: ['n_kv fp8 (recomputed)', 'n_kv scalesᵀ (recomputed)'], out: ['k_nope ‖ v (recomputed)'], epi: ['× r_kv', 'K, V in FA layout'] });
put(169, /rotary/, { lab: 'recompute: RoPE on k_rope', lane: 1, op: 'recompute', kind: 'op', b: rope(S * RO), in: ['kv latent ‖ k_rope', 'pos (k)'], out: ['k_rope (recomputed)'], how: 'stash', after: 0 });
put(170, /_copy_tensor/, { lab: 'recompute: copy q_nope', lane: 0, op: 'recompute', kind: 'op', b: copy(S * H * NO), in: ['q (pre-RoPE, recomputed)'], out: ['Q (nope half, recomputed)'], how: 'epilogue', host: 165, after: 0, inf: true });
put(171, EW, { lab: 'recompute: copy q_rope', lane: 0, op: 'recompute', kind: 'op', b: copy(S * H * RO), in: ['q (pre-RoPE, recomputed)'], out: ['q_rope (recomputed)'], how: 'epilogue', host: 165, after: 0, inf: true });
put(172, /rotary/, { lab: 'recompute: RoPE on q_rope', lane: 0, op: 'recompute', kind: 'op', b: rope(S * H * RO), in: ['q_rope (recomputed)', 'pos (q)'], out: ['Q (rope half, recomputed)'], how: 'epilogue', host: 165, after: 0 });
put(173, /FlashAttnBwdPreprocess/, { lab: 'FlashAttention backward: preprocess (D = rowsum O ⊙ dO, zero dQ accumulator)', lane: 0, op: 'attention bwd', kind: 'op', b: 2 * bf(S * O) + f32(S * H) + f32(S * QH), in: ['attn out (dequantized)', 'd(attn out)'], out: ['δ (rowsum O ⊙ dO)', 'dQ accumulator (zeroed)'], how: 'keep' });
put(174, /FlashAttnBwd</, { lab: 'FlashAttention backward', lane: 0, op: 'attention bwd', kind: 'attn', dt: 'bf16', f: S * S * H * (3 * (NO + RO) + 2 * DV),
  b: 2 * bf(S * QH) + 3 * bf(S * O) + 2 * f32(S * H) + 2 * f32(S * QH) + bf(S * QH) + bf(S * O),
  in: ['Q (nope half, recomputed)', 'Q (rope half, recomputed)', 'k_nope ‖ v (recomputed)', 'k_rope (recomputed)', 'd(attn out)', 'lse', 'δ (rowsum O ⊙ dO)', 'dQ accumulator (zeroed)'], out: ['dQ (fp32 accumulator)', 'dK', 'dV'], how: 'keep' });
put(175, /ConvertdQ/, { lab: 'FlashAttention backward: dQ fp32 → bf16', lane: 0, op: 'attention bwd', kind: 'op', b: f32(S * QH) + bf(S * QH), in: ['dQ (fp32 accumulator)'], out: ['dQ'], how: 'keep', epi: ['RoPE backward', 'dQ → FP8, both layouts', 'partial Σ for s_q'], after: f32(S * QH) + 2 * f8(S * QH) });
put(176, /rotary/, { lab: 'RoPE backward on dQ_rope', lane: 0, op: 'Q up bwd', kind: 'op', b: rope(S * H * RO), in: ['dQ', 'pos (q)'], out: ['dQ (RoPE backward)'], how: 'fa', host: 175, after: 0 });
put(177, BT, { lab: 'transpose W_q_up (FP8)', lane: 0, op: 'Q up bwd', kind: 'op', b: wT(QR * QH), in: ['W_q_up'], out: ['W_q_upᵀ'], how: 'cache', after: 0 });
put(178, CAST, { lab: 'quantize dQ → FP8 (S × 24576)', lane: 0, op: 'Q up bwd', kind: 'op', b: cast(S * QH), in: ['dQ (RoPE backward)'], out: ['dQ fp8'], how: 'fa', host: 175, after: 0 });
put(179, ST, { lab: 'transpose scales', lane: 0, op: 'Q up bwd', kind: 'glue', b: scaleT(S * QH), in: ['dQ fp8'], out: ['dQ scalesᵀ'], how: 'fa', host: 175, after: 0 });
put(180, GEMM, { lab: 'Q up: input gradient', lane: 0, op: 'Q up bwd', kind: 'gemm', dt: 'fp8', ...gemm(S, QR, QH), in: ['dQ fp8', 'dQ scalesᵀ', 'W_q_upᵀ'], out: ['d(n_q)'], epi: ['× r_q', 'q-latent RMSNorm backward', 'd(latent) → FP8, both layouts', 'partial Σ for s (input norm)'], add: 3 * f8(S * QR) + bf(S * QR) - bf(S * QR) });
put(181, EW, { lab: 'sum dk_rope over the 128 heads', lane: 1, op: 'KV up bwd', kind: 'op', b: bf(S * H * RO) + bf(S * RO), in: ['dK'], out: ['dk_rope (summed)'], how: 'aux', after: bf(S * H * RO) + bf(S * RO), note: 'a reduction across heads (across tiles): stays a kernel' });
put(182, /rotary/, { lab: 'RoPE backward on dk_rope', lane: 1, op: 'KV up bwd', kind: 'op', b: rope(S * RO), in: ['dk_rope (summed)', 'pos (k)'], out: ['dk_rope'], how: 'aux', after: 0 });
put(183, /_copy_tensor/, { lab: 'copy dk_nope into the dKV buffer', lane: 1, op: 'KV up bwd', kind: 'op', b: copy(S * H * NO), in: ['dK'], out: ['dKV (nope half)'], how: 'fa', host: 174, after: 0, inf: true, note: 'FlashAttention can write dK_nope and dV into one buffer' });
put(184, /_copy_tensor/, { lab: 'copy dV into the dKV buffer', lane: 1, op: 'KV up bwd', kind: 'op', b: copy(S * H * DV), in: ['dV'], out: ['dKV (v half)'], how: 'fa', host: 174, after: 0, inf: true });
put(185, BT, { lab: 'transpose W_kv_up (FP8)', lane: 1, op: 'KV up bwd', kind: 'op', b: wT(KVR * KVH), in: ['W_kv_up'], out: ['W_kv_upᵀ'], how: 'cache', after: 0 });
put(186, CAST, { lab: 'quantize dKV → FP8 (S × 32768)', lane: 1, op: 'KV up bwd', kind: 'op', b: cast(S * KVH), in: ['dKV (nope half)', 'dKV (v half)'], out: ['dKV fp8'], how: 'fa', host: 174, after: 0, note: 'one head = one 128-block: tile-local in FlashAttention\'s dK/dV store' });
put(187, ST, { lab: 'transpose scales', lane: 1, op: 'KV up bwd', kind: 'glue', b: scaleT(S * KVH), in: ['dKV fp8'], out: ['dKV scalesᵀ'], how: 'fa', host: 174, after: 0 });
put(188, GEMM, { lab: 'KV up: input gradient', lane: 1, op: 'KV up bwd', kind: 'gemm', dt: 'fp8', ...gemm(S, KVR, KVH), in: ['dKV fp8', 'dKV scalesᵀ', 'W_kv_upᵀ'], out: ['d(n_kv)'], epi: ['× r_kv', 'kv-latent RMSNorm backward', 'd(kv latent ‖ dk_rope) → FP8, both layouts', 'partial Σ for s (input norm)'], add: 3 * f8(S * KVP) });
put(189, CT, { lab: 'dKV → FP8ᵀ (weight-gradient operand)', lane: 1, op: 'KV up bwd', kind: 'op', b: cast(S * KVH), in: ['dKV (nope half)', 'dKV (v half)'], out: ['dKV fp8ᵀ'], how: 'fa', host: 174, after: 0 });
put(190, CT, { lab: 'kv latent → FP8ᵀ (weight-gradient operand)', lane: 1, op: 'KV up bwd', kind: 'op', b: cast(S * KVR), in: ['n_kv (recomputed)'], out: ['n_kv fp8ᵀ'], how: 'stash', after: 0 });
put(191, CT, { lab: 'dQ → FP8ᵀ (weight-gradient operand)', lane: 0, op: 'Q up bwd', kind: 'op', b: cast(S * QH), in: ['dQ (RoPE backward)'], out: ['dQ fp8ᵀ'], how: 'fa', host: 175, after: 0 });
put(192, CT, { lab: 'q latent → FP8ᵀ (weight-gradient operand)', lane: 0, op: 'Q up bwd', kind: 'op', b: cast(S * QR), in: ['n_q (recomputed, contiguous)'], out: ['n_q fp8ᵀ'], how: 'stash', after: 0 });
put(193, LNB, { lab: 'kv-latent RMSNorm backward', lane: 1, op: 'kv norm bwd', kind: 'op', b: normBwd(S, KVR, 528), in: ['d(n_kv)', 'kv latent ‖ k_rope', 'r_kv'], out: ['d(kv latent)', 'dγ_kv partials'], how: 'epilogue', host: 188, after: 0 });
put(194, LNF, { lab: 'dγ finalize', lane: 1, op: 'kv norm bwd', kind: 'glue', b: 2 * f32(KVR * 32), in: ['dγ_kv partials'], out: ['dγ_kv'], how: 'aux', after: 2 * f32(KVR * 32) });
put(195, EW, { lab: 'dγ accumulate', lane: 1, op: 'kv norm bwd', kind: 'glue', b: 3 * f32(KVR), in: ['dγ_kv'], out: ['dγ_kv (accumulated)'], how: 'aux', after: 0 });
put(196, LNB, { lab: 'q-latent RMSNorm backward', lane: 0, op: 'q norm bwd', kind: 'op', b: normBwd(S, QR, 660), in: ['d(n_q)', 'q latent', 'r_q'], out: ['d(q latent)', 'dγ_q partials'], how: 'epilogue', host: 180, after: 0 });
put(197, LNF, { lab: 'dγ finalize', lane: 0, op: 'q norm bwd', kind: 'glue', b: 2 * f32(QR * 32), in: ['dγ_q partials'], out: ['dγ_q'], how: 'aux', after: 2 * f32(QR * 32) });
put(198, EW, { lab: 'dγ accumulate', lane: 0, op: 'q norm bwd', kind: 'glue', b: 3 * f32(QR), in: ['dγ_q'], out: ['dγ_q (accumulated)'], how: 'aux', after: 0 });
chain = null;
for (const [i, l, e] of [[199, 'fill', { in: [], out: ['d(kv ‖ k_rope) (zeroed)'] }], [200, 'copy dk_rope', { in: ['dk_rope', 'd(kv ‖ k_rope) (zeroed)'] }], [201, 'fill', { in: [], out: ['d(kv ‖ k_rope) (zeroed, 2)'] }], [202, 'copy d(kv latent)', { in: ['d(kv latent)', 'd(kv ‖ k_rope) (zeroed, 2)'] }], [203, 'add', { in: ['assembly 200', 'assembly 202'], out: ['d(kv ‖ k_rope)'] }]]) {
  put(i, EW, { lab: `assemble d(kv packed): ${l}`, lane: 1, op: 'KV down bwd', kind: 'glue', b: ew(LAUNCHES[i].grid[0], 128, 4), how: 'epilogue', host: 188, after: 0, inf: true, out: [`assembly ${i}`], ...e });
}
put(204, LN, { lab: 'recompute: input RMSNorm (the down-projections\' input)', lane: 1, op: 'recompute', kind: 'op', b: norm(S, D) + bf(S * D), in: ['h₁'], out: ['n₁ (recomputed)'], how: 'stash', after: 0, note: 'the forward wrote x ⊙ γ in FP8; stash it in both scale layouts instead of recomputing (+29 MB of stash per layer)' });
put(205, CAST, { lab: 'quantize d(kv packed) → FP8', lane: 1, op: 'KV down bwd', kind: 'op', b: cast(S * KVP), in: ['d(kv ‖ k_rope)'], out: ['d(kv ‖ k_rope) fp8'], how: 'epilogue', host: 188, after: 0 });
put(206, BT, { lab: 'transpose W_kv_down (FP8)', lane: 1, op: 'KV down bwd', kind: 'op', b: wT(KVP * D), in: ['W_kv_down'], out: ['W_kv_downᵀ'], how: 'cache', after: 0 });
put(207, ST, { lab: 'transpose scales', lane: 1, op: 'KV down bwd', kind: 'glue', b: scaleT(S * KVP), in: ['d(kv ‖ k_rope) fp8'], out: ['d(kv ‖ k_rope) scalesᵀ'], how: 'epilogue', host: 188, after: 0 });
put(208, GEMM, { lab: 'KV down: input gradient', lane: 1, op: 'KV down bwd', kind: 'gemm', dt: 'fp8', ...gemm(S, D, KVP), in: ['d(kv ‖ k_rope) fp8', 'd(kv ‖ k_rope) scalesᵀ', 'W_kv_downᵀ'], out: ['d(n₁) kv'], epi: [] });
put(209, CT, { lab: 'normed input → FP8ᵀ (weight-gradient operand)', lane: 1, op: 'KV down bwd', kind: 'op', b: cast(S * D), in: ['n₁ (recomputed)'], out: ['n₁ fp8ᵀ'], how: 'stash', after: 0 });
put(210, CT, { lab: 'd(kv packed) → FP8ᵀ (weight-gradient operand)', lane: 1, op: 'KV down bwd', kind: 'op', b: cast(S * KVP), in: ['d(kv ‖ k_rope)'], out: ['d(kv ‖ k_rope) fp8ᵀ'], how: 'epilogue', host: 188, after: 0 });
put(211, CAST, { lab: 'quantize d(q latent) → FP8', lane: 0, op: 'Q down bwd', kind: 'op', b: cast(S * QR), in: ['d(q latent)'], out: ['d(q latent) fp8'], how: 'epilogue', host: 180, after: 0 });
put(212, BT, { lab: 'transpose W_q_down (FP8)', lane: 0, op: 'Q down bwd', kind: 'op', b: wT(QR * D), in: ['W_q_down'], out: ['W_q_downᵀ'], how: 'cache', after: 0 });
put(213, ST, { lab: 'transpose scales', lane: 0, op: 'Q down bwd', kind: 'glue', b: scaleT(S * QR), in: ['d(q latent) fp8'], out: ['d(q latent) scalesᵀ'], how: 'epilogue', host: 180, after: 0 });
put(214, GEMM, { lab: 'Q down: input gradient', lane: 0, op: 'Q down bwd', kind: 'gemm', dt: 'fp8', ...gemm(S, D, QR), in: ['d(q latent) fp8', 'd(q latent) scalesᵀ', 'W_q_downᵀ'], out: ['d(n₁) q'], epi: ['+ KV down\'s d(input)', 'input RMSNorm backward (needs s, r)', '+ residual gradient', 'partial dγ'], add: 3 * bf(S * D) });
put(215, CT, { lab: 'normed input → FP8ᵀ (again, for Q down\'s weight gradient)', lane: 0, op: 'Q down bwd', kind: 'op', b: cast(S * D), in: ['n₁ (recomputed)'], out: ['n₁ fp8ᵀ (again)'], how: 'stash', after: 0, note: 'the same tensor quantized and transposed twice' });
put(216, CT, { lab: 'd(q latent) → FP8ᵀ (weight-gradient operand)', lane: 0, op: 'Q down bwd', kind: 'op', b: cast(S * QR), in: ['d(q latent)'], out: ['d(q latent) fp8ᵀ'], how: 'epilogue', host: 180, after: 0 });
put(217, EW, { lab: 'add the two down-projections\' input gradients', lane: 0, op: 'input norm bwd', kind: 'op', b: 3 * bf(S * D), in: ['d(n₁) q', 'd(n₁) kv'], out: ['d(n₁)'], how: 'epilogue', host: 214, after: 0 });
put(218, LNB, { lab: 'input RMSNorm backward (+ residual gradient)', lane: 0, op: 'input norm bwd', kind: 'op', b: normBwd(S, D, 132), in: ['d(n₁)', 'h₁', 'r₁', 'd(x₁)'], out: ['d(x)', 'dγ₁ partials'], how: 'epilogue', host: 214, after: 0, note: 's for this norm comes from the Q-up / KV-up backward epilogues (Σ latent ⊙ d(latent)) — one GEMM boundary later, the paper\'s identity' });
put(219, LNF, { lab: 'dγ finalize', lane: 0, op: 'input norm bwd', kind: 'glue', b: 2 * f32(D * 32), in: ['dγ₁ partials'], out: ['dγ₁'], how: 'aux', after: 2 * f32(D * 32) });
put(220, EW, { lab: 'dγ accumulate', lane: 0, op: 'input norm bwd', kind: 'glue', b: 3 * f32(D), in: ['dγ₁'], out: ['dγ₁ (accumulated)'], how: 'aux', after: 0 });

// ======================= the communication stream =======================
// `seq` places each DeepEP launch in the routed lane: the backward dispatch of
// dy opens the MoE backward; the forward dispatch follows the router; each
// combine closes its pass.
put(226, CAST, { lab: 'quantize d(MoE output) → FP8 before the backward dispatch', lane: 2, col: 'comm', seq: 64.1, op: 'combine bwd', kind: 'op', b: cast(S * D), in: ['dy'], out: ['dy fp8'], how: 'epilogue', host: 'grad', after: 0, note: 'produced by the next layer\'s input-RMSNorm-backward epilogue, in FP8 directly' });
const CM = (i, lab, op, seq, ins, outs) => put(i, /dpsk::ep/, { lab, lane: 2, col: 'comm', seq, op, kind: 'comm', b: 0, in: ins, out: outs, how: 'keep' });
CM(227, 'DeepEP: notify (backward dispatch)', 'combine bwd', 64.2, ['layout'], ['ready (backward dispatch)']); CM(228, 'DeepEP: dispatch the output gradients (FP8)', 'combine bwd', 64.3, ['dy fp8', 'ready (backward dispatch)'], ['recv d(y)']);
CM(229, 'DeepEP: dispatch layout', 'dispatch', 100.1, ['top-k idx'], ['layout']); CM(230, 'DeepEP: notify (dispatch)', 'dispatch', 100.2, ['layout'], ['ready (dispatch)']); CM(231, 'DeepEP: dispatch tokens (FP8 + scales)', 'dispatch', 100.3, ['n₂ fp8', 'top-k w', 'ready (dispatch)'], ['recv tokens', 'recv top-k w', 'recv idx']);
CM(232, 'DeepEP: notify (backward combine)', 'dispatch bwd', 85.1, ['layout'], ['ready (backward combine)']); CM(234, 'DeepEP: combine the input gradients (BF16)', 'dispatch bwd', 85.2, ['d(x) (per token)', 'd(top-k w) (routed rows)', 'ready (backward combine)'], ['d(n₂) routed', 'd(top-k w)']);
CM(238, 'DeepEP: notify (combine)', 'combine', 121.1, ['layout'], ['ready (combine)']); CM(239, 'DeepEP: combine the expert outputs (BF16)', 'combine', 121.2, ['y (per token)', 'y_sh', 'ready (combine)'], ['y (MoE output)']);

for (let i = 0; i < LAUNCHES.length; i++) if (!SPEC[i]) throw new Error(`launch ${i} (${LAUNCHES[i].k}) has no spec`);

// ---- the boundary: what enters a pass from outside it, and what leaves ---------
// Tensors a pass reads that no launch of it writes must be layer inputs or
// weights (LAYER_IN), or outputs of an earlier pass (a stash: F → B, F/B → W).
// Tensors a pass writes that nothing in it reads must be read by a later pass
// (a stash) or be declared layer outputs (LAYER_OUT); dW… / dγ… are weight
// gradients by name.
export const LAYER_IN = {
  F: { x: 'the residual stream entering the layer (the previous layer\'s h₂)', 'y (prev. layer)': 'the previous layer\'s MoE output, straight from its combine', W13: 'weights (the 4 local experts\')', W2: 'weights (the 4 local experts\')' },
  B: { dy: 'd(layer output), from the next layer\'s input-norm backward', 'W2 (shared)': 'weight', 'W13 (shared)': 'weight', W_o: 'weight', W_q_up: 'weight', W_kv_up: 'weight', W_kv_down: 'weight', W_q_down: 'weight' },
  W: {},
};
export const LAYER_OUT = {
  F: { 'h₂': 'the residual stream → the next layer', 'y (MoE output)': '→ the next layer\'s input norm', 'aux-loss stats': '→ the load-balance loss', 'unique groups': 'routing bookkeeping (read by nothing modeled here)'},
  B: { 'd(x)': 'd(residual stream) → the previous layer', 'h₂ (recomputed)': 'the recomputed residual (unread: the backward needs n₂ only)' },
  W: {},
};
const EARLIER = { F: [], B: ['F'], W: ['F', 'B'] };
const WGRAD = (t) => /^d(W|γ)/.test(t);

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
    const bAfter = host ? s.b + (s.add ?? 0) + (s.out8 ?? 0) : s.how === 'keep' && s.after == null ? s.b : (s.after ?? 0);
    const hbmAfter = bAfter / rates.hbm;
    const solAfter = host ? Math.max(flop, hbmAfter) : hbmAfter;
    // gone: the launch folds away (into a host, a stash, a once-per-step transpose); aux: a small kernel remains in its place
    const gone = !host && !comm && s.how !== 'keep';
    const col = s.col ?? (s.lane < 2 ? 'attn' : 'mlp');
    return { ...o, ...s, col, seq: s.seq ?? i, pass, comm, flop, hbm, sol, bound, x: sol ? o.us * 1e-6 / sol : null, bAfter, hbmAfter, solAfter, gone, aux: gone && bAfter > 0 };
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

// ---- the graph ------------------------------------------------------------------
// The launches of one pass as nodes and the tensors between them as edges
// (producer → consumer, by name). Every launch of the pass is a node (the
// DeepEP launches of the pass's all-to-alls included). Each node also lists
// its `sources` (inputs from outside the pass: {t, from: 'layer' | 'weight' |
// 'F' | 'B'}) and `sinks` (outputs nothing in the pass reads: {t, to:
// 'layer' | 'wgrad' | 'B' | 'W'}). Throws if the accounting does not close.
const PASS_LAUNCHES = (P, pass) => P.filter((e) => e.pass === pass || (e.comm && (pass === 'F' ? /\(F\)/.test(e.ann) : pass === 'B' ? /\(B\)/.test(e.ann) : false)));
export function graph(pass, P = priced()) {
  const nodes = PASS_LAUNCHES(P, pass);
  const producer = new Map();
  for (const e of nodes) for (const t of e.out) { if (producer.has(t)) throw new Error(`${pass}: '${t}' is written by launches ${producer.get(t).i} and ${e.i}`); producer.set(t, e); }
  const consumers = new Map();
  for (const e of nodes) for (const t of e.in) { if (!consumers.has(t)) consumers.set(t, []); consumers.get(t).push(e); }
  const outsOf = (p) => new Set(PASS_LAUNCHES(P, p).flatMap((e) => e.out));
  const insOf = (p) => new Set(PASS_LAUNCHES(P, p).flatMap((e) => e.in));
  const earlier = EARLIER[pass].map((p) => [p, outsOf(p)]), later = ['F', 'B', 'W'].filter((p) => EARLIER[p].includes(pass)).map((p) => [p, insOf(p)]);
  const edges = [], problems = [];
  for (const e of nodes) {
    e.sources = []; e.sinks = [];
    for (const t of e.in) {
      if (producer.has(t)) { edges.push({ t, p: producer.get(t), c: e }); continue; }
      const from = LAYER_IN[pass][t] != null ? (/weight/.test(LAYER_IN[pass][t]) ? 'weight' : 'layer') : earlier.find(([, outs]) => outs.has(t))?.[0];
      if (!from) problems.push(`launch ${e.i} reads '${t}', which nothing produces`);
      e.sources.push({ t, from, why: LAYER_IN[pass][t] });
    }
    for (const t of e.out) {
      if (consumers.has(t)) continue;
      const to = t in LAYER_OUT[pass] ? 'layer' : WGRAD(t) ? 'wgrad' : later.find(([, ins]) => ins.has(t))?.[0];
      if (!to) problems.push(`launch ${e.i} writes '${t}', which nothing reads`);
      e.sinks.push({ t, to, why: LAYER_OUT[pass][t] });
    }
  }
  if (problems.length) throw new Error(`kernel graph (${pass}) does not close:\n  ${problems.join('\n  ')}`);
  // CODA folds follow the wires: a launch that folds into a GEMM's epilogue (or
  // into FlashAttention's / the dQ postprocess's) must reach its host through
  // folded launches only — along the data dependencies, in either direction
  const byI = new Map(nodes.map((e) => [e.i, e]));
  const adj = new Map(nodes.map((e) => [e.i, new Set()]));
  for (const { p, c } of edges) { adj.get(p.i).add(c.i); adj.get(c.i).add(p.i); }
  for (const e of nodes) {
    if (typeof e.host !== 'number') continue;
    const h = byI.get(e.host);
    if (!h) { problems.push(`launch ${e.i} folds into launch ${e.host}, which is not in this pass`); continue; }
    const seen = new Set([e.i]), q = [e.i]; let ok = false;
    while (q.length && !ok) for (const n of adj.get(q.shift())) { if (n === h.i) { ok = true; break; } if (!seen.has(n) && byI.get(n).gone) { seen.add(n); q.push(n); } }
    if (!ok) problems.push(`launch ${e.i} (${e.lab}) folds into ${h.i} (${h.lab}) but is not wired to it`);
  }
  if (problems.length) throw new Error(`CODA folds off the wires (${pass}):\n  ${problems.join('\n  ')}`);
  return { pass, nodes, edges };
}
export { SPEC };
