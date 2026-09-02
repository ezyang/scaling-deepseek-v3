// Precision recipes: which dtype each matmul runs in (and stashes its
// backward inputs in). Pure data + one resolver — split from memory.js so
// the WHAT-precision tables stand apart from the memory model that prices
// them. A matmul's dtype governs both its weight copy and the stashed input
// activations its backward needs.

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
    dimsNote: 'hidden 7168 → 256 expert logits + 256-element score-correction bias (top-8 kept, fp32 gating)' },
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
  // even the all-bf16 baseline pins the router fp32: gating is never a
  // precision choice in the Hopper story (the paper keeps it high-precision,
  // production runs it fp32), so the label must not flip between sections.
  // (nv-mxfp8 keeps router bf16 — that IS NVIDIA's choice, a Blackwell-post fact.)
  'bf16': { router: 'fp32' },
  // DeepSeek-V3 paper recipe: linears in tile-scaled fp8 (the Hopper flavor —
  // same bytes as MX, its own key so labels carry provenance); attention core and head high-precision; the router runs
  // fp32 in production; the attn-out linear's stash is the paper's customized
  // E5M6 (§3.3.3) — the GEMM itself runs fp8 (flopEq prices e5m6 at fp8 rate);
  // swiglu_in = the SwiGLU input's SAVE format (free-floating — §3.3.3 caches it fp8).
  'dsv3-fp8': { qkv_down: 'e4m3', q_up: 'e4m3', kv_up: 'e4m3', o_proj: 'e5m6', router: 'fp32', ffn_gate_up: 'e4m3', ffn_down: 'e4m3', swiglu_in: 'e4m3' },
  // dsv3-fp8 without the wide exception: EVERY linear runs (and stashes) fp8,
  // the attn-out included — a production H100 variant (notes.txt: MM5 is
  // fp8_linear). Attention core and head stay bf16, router fp32. Under
  // attention-replay recompute the attn-out stash never materializes, so this
  // recipe's BYTES equal dsv3-fp8's there (sanity pins that) — the difference
  // is the o_proj GEMM's compute pricing.
  'all-fp8': { qkv_down: 'e4m3', q_up: 'e4m3', kv_up: 'e4m3', o_proj: 'e4m3', router: 'fp32', ffn_gate_up: 'e4m3', ffn_down: 'e4m3', swiglu_in: 'e4m3' },
  // NVIDIA NeMo/Megatron-Bridge (MLPerf 6.0 submission) recipe: MXFP8 for every
  // GEMM (32-element MX blocks, UE8M0 scales, via TE) INCLUDING the attention
  // core (Blackwell FP8 attention: q/k/v saved MXFP8). The attention OUTPUT is
  // saved bf16 (o_proj stash wide); router/head bf16.
  'nv-mxfp8': { qkv_down: 'mxfp8', q_up: 'mxfp8', kv_up: 'mxfp8', attn: 'mxfp8', o_proj: 'bf16', ffn_gate_up: 'mxfp8', ffn_down: 'mxfp8', swiglu_in: 'mxfp8' },
};

// each recipe's CANONICAL stash-side checkbox state: the e4m3ᵀ dual stash is
// part of the production H100 recipe (notes.txt quantizes with transpose at
// forward); DeepSeek's own recipe re-quantizes in backward instead (off).
// Recipe recognition compares these too — flip a checkbox and you are custom.
export const RECIPE_T = { 'all-fp8': true };

export function resolveMatmuls(cfg) {
  const recipe = cfg.recipe ?? (cfg.dtype === 'mxfp8' ? 'nv-mxfp8' : 'bf16');
  return {
    ...Object.fromEntries(MATMULS.map(m => [m.id, 'bf16'])),
    // swiglu_in is a STASH-format channel, not a GEMM: the SwiGLU input's
    // save precision is free-floating (its only backward reader is the
    // elementwise SwiGLU backward — no GEMM forces the format)
    swiglu_in: 'bf16',
    ...RECIPES[recipe],
    ...(cfg.matmuls ?? {}),
  };
}
