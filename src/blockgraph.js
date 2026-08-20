// The DSv3 transformer block as an explicit op graph, for recompute planning.
//
// Every compute op is marked `save` (its output tensor is stashed for backward)
// or `recompute` (2x that op's compute: it replays once during backward).
// Backward needs specific tensors: a matmul/norm/swiglu backward needs its
// inputs, flash attention also needs its own output (+lse), residual adds need
// nothing. The gate weights are applied to the swiglu output BEFORE the (linear)
// down-proj — mathematically identical to gating the expert outputs — so the
// combine's backward (router-weight grads) needs the swiglu output and the
// gating probs, NOT the expert outputs. Any needed tensor marked `recompute` pulls
// that producer — and, transitively, its unmarked-unsaved ancestors — into the
// backward replay. The block is the checkpoint region: its INPUT (x0) is always
// saved — the anchor every replay chain terminates at. Its output x2 is the
// next region's x0 and is charged there.
//
// Norm backwards are input-form by design (they need their input, and the
// per-token rstd aux). Aux backward artifacts (rstd, lse) are saved unless
// their op replays — a replay regenerates them as a byproduct.
// Simplification: each replayed op runs exactly once per layer backward.

// MXFP8: 1-byte E4M3 elements + one UE8M0 scale byte per 32-element block
// = 1 + 1/32 B/elem. (DeepSeek's tile-wise recipe — 4 fp32 scale bytes per
// 128 elements — has the identical overhead, so this constant covers both.)
export const DTYPE_BYTES = { bf16: 2, mxfp8: 1 + 1 / 32, fp32: 4 };

// Marking presets. true = save output, false = recompute. Unlisted ops: save.
export const RECOMPUTE_PRESETS = {
  none: {},
  // DeepSeek-paper policy: recompute RMSNorms, MLA up-projections, SwiGLU
  dsv3: { norm1: false, norm2: false, q_up: false, kv_up: false, swiglu: false },
  // aggressive production policy: recompute ALL of attention (norm1 through
  // the residual add) from x0 in backward; the saved attention-side tensor is
  // norm2's OUTPUT (the attention output post-norm), which the MoE half consumes.
  'attn-replay': {
    norm1: false, qkv_down: false, q_up: false, kv_up: false, attn: false, o_proj: false, x1: false,
    swiglu: false,
  },
  // dsv3 + recompute ffn gate/up from the dispatched tokens
  selective: { norm1: false, norm2: false, q_up: false, kv_up: false, swiglu: false, gate_up: false },
  // full checkpointing: replay the whole layer from its input (incl. the a2a!)
  full: {
    norm1: false, qkv_down: false, q_up: false, kv_up: false, attn: false, o_proj: false, x1: false,
    norm2: false, router: false, dispatch: false, gate_up: false, swiglu: false, ffn_down: false, combine: false,
  },
};

export function resolveMarks(cfg) {
  const preset = RECOMPUTE_PRESETS[cfg.recompute ?? 'selective'] ?? {};
  return { ...preset, ...(cfg.saved ?? {}) };
}

// Build the graph for one layer. Sizes are per token (bytes include dtype).
export function blockGraph(kind, a, mm, seqLen) {
  const B = (id) => DTYPE_BYTES[mm[id]];
  const h = a.hidden, qk = a.qkNope + a.qkRope;
  const inter = kind === 'dense' ? a.denseInter : a.moeInter;
  const experts = kind === 'dense' ? 1 : a.topk + a.sharedExperts;
  const moe = kind === 'moe';
  // aux: per-token backward artifacts (rstd, lse) — saved unless the op replays
  const N = (id, label, opKind, inputs, tensor, elems, bytesPer, flopsTok, opts = {}) => ({
    id, label, opKind, inputs, tensor, elems,
    outBytes: elems * bytesPer,
    flopsTok, bucket: opts.bucket ?? 'moe',
    always: opts.always ?? false, needsOwnOutput: opts.needsOwnOutput ?? false,
    aux: opts.aux ?? null, bwdNeeds: opts.bwdNeeds ?? null,
  });
  const nodes = [
    N('x0', 'block input', 'boundary', [], 'x0 (checkpoint anchor)', h, 2, 0, { bucket: 'residual', always: true }),
    N('norm1', 'RMSNorm', 'vector', ['x0'], 'norm1 out', h, B('qkv_down'), 8 * h, { bucket: 'mla', aux: { name: 'rstd', bytes: 4 } }),
    N('qkv_down', 'q/kv down-proj', 'matmul', ['norm1'], 'latents (KV cache)',
      a.qRank + a.kvRank + a.qkRope, B('q_up'), 2 * (h * a.qRank + h * (a.kvRank + a.qkRope)), { bucket: 'mla' }),
    N('q_up', 'q up-proj', 'matmul', ['qkv_down'], 'q', a.heads * qk, B('attn'), 2 * a.qRank * a.heads * qk, { bucket: 'mla' }),
    N('kv_up', 'kv up-proj', 'matmul', ['qkv_down'], 'k,v', a.heads * (qk + a.vHead), B('attn'),
      2 * a.kvRank * a.heads * (a.qkNope + a.vHead), { bucket: 'mla' }),
    N('attn', 'flash attention', 'attn', ['q_up', 'kv_up'], 'attn out', a.heads * a.vHead, B('o_proj'),
      2 * a.heads * (qk + a.vHead) * seqLen / 2, { bucket: 'mla', aux: { name: 'lse', bytes: 4 * a.heads }, needsOwnOutput: true }),
    N('o_proj', 'attn out-proj', 'matmul', ['attn'], 'attn proj out', h, 2, 2 * a.heads * a.vHead * h, { bucket: 'mla' }),
    N('x1', '+ residual', 'add', ['o_proj', 'x0'], 'x1 (residual)', h, 2, 0, { bucket: 'residual' }),
    N('norm2', 'RMSNorm', 'vector', ['x1'], 'norm2 out', h, B('ffn_gate_up'), 8 * h, { bucket: 'moe', aux: { name: 'rstd', bytes: 4 } }),
    ...(moe ? [
      // production routers retain logits + scores (both fp32, even when their
      // loss weights are zero) + top-k weights/indices + int32 routing mappings
      // ≈ 2.1 KiB/token, not just the 1 KiB of probs. Implementation-specific —
      // Megatron will need its own retention set when we study it.
      N('router', 'router', 'matmul', ['norm2'], 'router state (logits, scores, top-k)',
        2 * a.routedExperts + 4 * a.topk, 4, 2 * h * a.routedExperts),
      N('dispatch', 'a2a dispatch', 'comm', ['norm2', 'router'], 'dispatched tokens', a.topk * h, B('ffn_gate_up'), 0),
    ] : []),
    N('gate_up', 'ffn gate/up', 'matmul', moe ? ['dispatch', 'norm2'] : ['norm2'], 'gate, up',
      experts * 2 * inter, B('ffn_down'), 2 * 2 * h * inter * experts),
    N('swiglu', 'SwiGLU', 'vector', ['gate_up'], 'swiglu out', experts * inter, B('ffn_down'), 6 * experts * inter),
    N('ffn_down', 'ffn down', 'matmul', ['swiglu'], 'expert outputs', moe ? a.topk * h : h, 2, 2 * h * inter * experts),
    ...(moe ? [N('combine', 'a2a combine', 'comm', ['ffn_down', 'router'], 'moe out', h, 2, 0,
      { bwdNeeds: ['swiglu', 'router'] })] : []),
    N('x2', '+ residual (out)', 'add', [moe ? 'combine' : 'ffn_down', 'x1'], 'x2 → next block', h, 2, 0,
      { bucket: 'residual' }),
  ];
  return nodes;
}

const BWD_NEEDS_INPUTS = { matmul: true, vector: true, attn: true, boundary: false, add: false, comm: false };
const bwdNeedsInputs = (n) => BWD_NEEDS_INPUTS[n.opKind];

// Given markings ({id: false} = recompute), compute what is stashed and what
// replays. Returns per-token byte totals by bucket, the saved/replayed node
// sets, and the replay flop/comm overhead.
export function analyze(nodes, marks) {
  const byId = Object.fromEntries(nodes.map(n => [n.id, n]));
  const saved = (n) => n.always || (marks[n.id] !== false);
  const neededSaved = new Set(), replayed = new Set();
  const neededBy = new Map();                        // tensor id -> Set of ops whose backward/replay reads it
  const need = (id, by) => {
    const n = byId[id];
    if (!n) return;
    if (!neededBy.has(id)) neededBy.set(id, new Set());
    neededBy.get(id).add(by);
    if (saved(n)) { neededSaved.add(id); return; }
    if (replayed.has(id)) return;
    replayed.add(id);
    for (const i of n.inputs) need(i, id);           // replay needs its inputs in turn
  };
  for (const n of nodes) {
    if (n.bwdNeeds) { for (const i of n.bwdNeeds) need(i, n.id); }
    else if (bwdNeedsInputs(n)) for (const i of n.inputs) need(i, n.id);
    if (n.needsOwnOutput) need(n.id, n.id);
  }
  // the checkpoint region saves its own input: the anchor every replay reads
  neededSaved.add('x0');
  if (!neededBy.has('x0')) neededBy.set('x0', new Set(['replay anchor']));
  const buckets = { mla: 0, moe: 0, residual: 0 };
  let savedBytes = 0;
  for (const id of neededSaved) {
    const n = byId[id];
    buckets[n.bucket] += n.outBytes;
    savedBytes += n.outBytes;
  }
  for (const n of nodes) {         // aux artifacts (rstd, lse): a replay regenerates them
    if (n.aux && !replayed.has(n.id)) { buckets[n.bucket] += n.aux.bytes; savedBytes += n.aux.bytes; }
  }
  const replayFlopsTok = [...replayed].reduce((t, id) => t + byId[id].flopsTok, 0);
  const fwdFlopsTok = nodes.reduce((t, n) => t + n.flopsTok, 0);
  const replayComm = ['dispatch', 'combine'].filter(id => replayed.has(id));
  return {
    buckets, savedBytes, replayFlopsTok, fwdFlopsTok,
    replayFrac: fwdFlopsTok ? replayFlopsTok / fwdFlopsTok : 0,
    neededSaved, replayed, replayComm, neededBy, byId,
  };
}
