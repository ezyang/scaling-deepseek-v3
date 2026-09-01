// The DSv3 transformer block as an explicit op graph, for recompute planning.
//
// SAVE-DRIVEN marking (torch_remat's authoring direction): the block starts
// as recompute-everything; a policy writes `save` marks in ({id: true}).
// A saved op's output is stashed for backward (if backward reads it); an
// unmarked op RECOMPUTES — literally, replaying once during backward.
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

// e4m3: DeepSeek's Hopper flavor — 1-byte E4M3 elements + 4 fp32 scale bytes
// per 1×128 tile = 1 + 1/32 B/elem. MXFP8: the Blackwell flavor — E4M3 +
// one UE8M0 scale byte per 32-element block, the IDENTICAL overhead. Two
// keys, same bytes: the labels carry provenance (and the fp8ᵀ transpose tax
// applies only to the tile-scaled flavor's per-row scales).
// E5M6: DeepSeek's customized 12-bit format (§3.3.3) exclusively for the
// attention output — read by BOTH attention backward and the attn-out
// linear's wgrad, too precision-sensitive for fp8. 1.5 B/elem; the paper
// implies these tiles ARE 1×128-scaled with power-of-two scales (that's
// what makes the 1×128 → 128×1 backward flip lossless) — the scale bytes
// (≤1/128) and physical 12-bit packing are our inference. The GEMM that
// reads it still RUNS fp8 (e5m6 names the stash).
export const DTYPE_BYTES = { bf16: 2, e4m3: 1 + 1 / 32, mxfp8: 1 + 1 / 32, e5m6: 1.5, fp32: 4 };

// Marking is SAVE-driven (torch_remat's authoring direction): the checkpoint
// region starts as RECOMPUTE-EVERYTHING, and a policy writes saves in.
// marks: {id: true} = save this op's output; anything unlisted RECOMPUTES
// (literally — see analyze). Recompute marks are literal; saves stay
// demand-gated (autograd never materializes an output backward won't read).
export const MARKABLE = ['norm1', 'qkv_down', 'q_norm', 'kv_norm', 'q_up', 'kv_up', 'rope_q', 'rope_kv',
  'attn', 'o_proj', 'x1', 'norm2', 'router', 'dispatch', 'gate_up', 'swiglu', 'ffn_down', 'combine', 'moe_add'];
const saveAllExcept = (...redo) => Object.fromEntries(MARKABLE.filter(i => !redo.includes(i)).map(i => [i, true]));
export const RECOMPUTE_PRESETS = {
  // ORDERED as the authoring direction reads: start from the empty policy
  // (recompute everything) and write saves in — the stash grows rightward,
  // like every linear bar on the site. Every preset except 'none' re-runs
  // RoPE in backward (production fuses it into the attention prologue);
  // 'none' is naive autograd — it saves the rotated q/k (attention's actual
  // inputs), the same bytes either way.
  // full checkpointing: NO saves — replay the whole layer from its input
  // (incl. the a2a!). The canvas every policy starts from.
  full: {},
  // aggressive production policy: recompute ALL of attention (norm1 through
  // the residual add) from x0 in backward; the saved attention-side tensor is
  // norm2's OUTPUT (the attention output post-norm), which the MoE half consumes.
  'attn-replay': saveAllExcept('norm1', 'qkv_down', 'q_norm', 'kv_norm', 'q_up', 'kv_up',
    'rope_q', 'rope_kv', 'attn', 'o_proj', 'x1', 'swiglu'),
  // dsv3 + recompute ffn gate/up from the dispatched tokens
  selective: saveAllExcept('norm1', 'norm2', 'q_norm', 'kv_norm', 'q_up', 'kv_up',
    'rope_q', 'rope_kv', 'swiglu', 'gate_up'),
  // DeepSeek-paper policy, two verbatim sources: \u00a73.2.3 "We recompute all
  // RMSNorm operations and MLA up-projections during back-propagation"
  // (ALL norms — the latent norms included), and \u00a73.3.3 "we cache the
  // inputs of the SwiGLU operator and recompute its output in the backward
  // pass" (hence gate/up-out stays stashed here while swiglu goes \u21bb).
  // RoPE \u21bb is OUR extension (fused production kernels; a zero-byte choice
  // either way, so it cannot move the ledger) — the paper doesn't name it.
  dsv3: saveAllExcept('norm1', 'norm2', 'q_norm', 'kv_norm', 'q_up', 'kv_up', 'rope_q', 'rope_kv', 'swiglu'),
  none: saveAllExcept(),
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
    always: opts.always ?? false, nomark: opts.nomark ?? false, needsOwnOutput: opts.needsOwnOutput ?? false,
    aux: opts.aux ?? null, bwdNeeds: opts.bwdNeeds ?? null,
    tdims: opts.tdims ?? String(elems),   // unitless per-token size, factored like the op dims
  });
  const nodes = [
    N('x0', 'block input', 'boundary', [], 'x0 (checkpoint anchor)', h, 2, 0, { bucket: 'residual', always: true }),
    N('norm1', 'RMSNorm', 'vector', ['x0'], 'norm1 out', h, B('qkv_down'), 8 * h, { bucket: 'mla', aux: { name: 'rstd', bytes: 4 } }),
    // stash excludes the k_rope dims: RoPE's backward is a transposed rotation
    // (needs no input), and wkv_a's wgrad needs norm1-out, not its own output
    // Pre-norm latents kept in bf16 (the latent norms' backward input).
    N('qkv_down', 'q/kv down-proj', 'matmul', ['norm1'], 'latents',
      a.qRank + a.kvRank, 2, 2 * (h * a.qRank + h * (a.kvRank + a.qkRope)),
      { bucket: 'mla', tdims: `${a.qRank} + ${a.kvRank}` }),
    // the MLA-internal latent norms are real ops: at no-AC both the pre-norm
    // latent (their backward input) and their normed output (the up-proj wgrad
    // activation, TE's cached fp8 copy) are stashed; every recompute preset
    // replays them (the DSv3 paper names RMSNorms explicitly)
    N('q_norm', 'RMSNorm (q latent)', 'vector', ['qkv_down'], 'norm(q latent)', a.qRank, B('q_up'),
      8 * a.qRank, { bucket: 'mla', aux: { name: 'rstd', bytes: 4 } }),
    N('kv_norm', 'RMSNorm (kv latent)', 'vector', ['qkv_down'], 'norm(kv latent)', a.kvRank, B('kv_up'),
      8 * a.kvRank, { bucket: 'mla', aux: { name: 'rstd', bytes: 4 } }),
    N('q_up', 'q up-proj', 'matmul', ['q_norm'], 'q (pre-RoPE)', a.heads * qk, B('attn'), 2 * a.qRank * a.heads * qk,
      { bucket: 'mla', tdims: `${a.heads}\u00d7${qk}` }),
    N('kv_up', 'kv up-proj', 'matmul', ['kv_norm'], 'k,v (pre-RoPE)', a.heads * (qk + a.vHead), B('attn'),
      2 * a.kvRank * a.heads * (a.qkNope + a.vHead), { bucket: 'mla', tdims: `${a.heads}\u00d7(${qk}+${a.vHead})` }),
    // RoPE as REAL nodes: the mark is a genuine (zero-byte) choice — save the
    // rotated q/k (attention's actual inputs; same size as pre-RoPE) or
    // re-run the rotation in backward (cheap bandwidth-bound vector work).
    // Its own backward needs NOTHING (a fixed rotation: dL/dx = R\u1d40\u00b7dL/dy),
    // hence bwdNeeds: [] — only a REPLAY pulls the pre-RoPE input.
    N('rope_q', 'RoPE (q)', 'vector', ['q_up'], 'q', a.heads * qk, B('attn'),
      6 * a.heads * a.qkRope, { bucket: 'mla', bwdNeeds: [], tdims: `${a.heads}\u00d7${qk}` }),
    N('rope_kv', 'RoPE (k, build K,V)', 'vector', ['kv_up'], 'k,v', a.heads * (qk + a.vHead), B('attn'),
      6 * a.qkRope, { bucket: 'mla', bwdNeeds: [], tdims: `${a.heads}\u00d7(${qk}+${a.vHead})` }),
    N('attn', 'attention', 'attn', ['rope_q', 'rope_kv'], 'attn out', a.heads * a.vHead, B('o_proj'),
      2 * a.heads * (qk + a.vHead) * seqLen / 2, { bucket: 'mla', aux: { name: 'lse', bytes: 4 * a.heads }, needsOwnOutput: true, tdims: `${a.heads}\u00d7${a.vHead}` }),
    N('o_proj', 'attn out-proj', 'matmul', ['attn'], 'attn proj out', h, 2, 2 * a.heads * a.vHead * h, { bucket: 'mla' }),
    N('x1', '+ residual', 'add', ['o_proj', 'x0'], 'x1 (residual)', h, 2, 0, { bucket: 'residual' }),
    N('norm2', 'RMSNorm', 'vector', ['x1'], 'norm2 out', h, B('ffn_gate_up'), 8 * h, { bucket: 'moe', aux: { name: 'rstd', bytes: 4 } }),
    ...(moe ? [
      // production routers retain logits + scores (both fp32, even when their
      // loss weights are zero) + top-k weights/indices + int32 routing mappings
      // ≈ 2.1 KiB/token, not just the 1 KiB of probs. Implementation-specific —
      // Megatron will need its own retention set when we study it.
      N('router', 'router', 'matmul', ['norm2'], 'router state (logits, scores, top-k)',
        2 * a.routedExperts + 4 * a.topk, 4, 2 * h * a.routedExperts,
        { tdims: `2\u00d7${a.routedExperts} + 4\u00d7${a.topk}` }),
      N('dispatch', 'a2a dispatch', 'comm', ['norm2', 'router'], 'dispatched tokens', a.topk * h, B('ffn_gate_up'), 0,
        { tdims: `${a.topk}\u00d7${h}` }),
    ] : []),
    // the gate/up stash (= the SwiGLU input) has a FREE-FLOATING save format
    // (mm.swiglu_in): its only backward reader is the elementwise SwiGLU
    // backward \u2014 no GEMM ever consumes it, so no GEMM forces its precision
    // (the paper CHOOSES fp8 for it, \u00a73.3.3). ?? ffn_down covers stale
    // hand-rolled matmul dicts from before the channel existed.
    N('gate_up', 'ffn gate/up', 'matmul', moe ? ['dispatch', 'norm2'] : ['norm2'], 'gate, up',
      experts * 2 * inter, DTYPE_BYTES[mm.swiglu_in ?? mm.ffn_down], 2 * 2 * h * inter * experts,
      { tdims: moe ? `${experts}\u00d72\u00d7${inter}` : `2\u00d7${inter}` }),
    N('swiglu', 'SwiGLU', 'vector', ['gate_up'], 'swiglu out', experts * inter, B('ffn_down'), 6 * experts * inter,
      { tdims: moe ? `${experts}\u00d7${inter}` : String(inter) }),
    N('ffn_down', 'ffn down', 'matmul', ['swiglu'], moe ? 'expert outputs' : 'ffn out',
      moe ? a.topk * h : h, 2, 2 * h * inter * experts,
      { tdims: moe ? `${a.topk}\u00d7${h}` : String(h) }),
    ...(moe ? [N('combine', 'a2a combine', 'comm', ['ffn_down', 'router'], 'moe out', h, 2, 0,
      { bwdNeeds: ['swiglu', 'router'] }),
    // the routed+shared sum is a REAL node (we model no fusion — Megatron's
    // add_shared_and_residual is trivia, not an excuse): an add, so its
    // backward needs nothing; a ↻ mark is representable and honestly
    // wasteful (the replay pulls combine-out into the stash for nothing)
    N('moe_add', '+ routed + shared', 'add', ['combine'], 'ffn out (routed + shared)', h, 2, 0,
      { bucket: 'moe' })] : []),
    N('x2', '+ residual (out)', 'add', [moe ? 'moe_add' : 'ffn_down', 'x1'], 'x2 → next block', h, 2, 0,
      { bucket: 'residual', nomark: true }),   // the block boundary: its output is the next region's x0 (charged there) — no mark exists
  ];
  return nodes;
}

const BWD_NEEDS_INPUTS = { matmul: true, vector: true, attn: true, boundary: false, add: false, comm: false };
const bwdNeedsInputs = (n) => BWD_NEEDS_INPUTS[n.opKind];

// Given markings ({id: false} = recompute), compute what is stashed and what
// replays. Returns per-token byte totals by bucket, the saved/replayed node
// sets, and the replay flop/comm overhead.
//
// transposedStash models Hopper tile-scaled fp8 (1×128 per-row scales): an fp8
// stash consumed by a wgrad GEMM is kept in BOTH quantization orientations,
// because per-row scales don't transpose exactly. Blackwell MXFP8's power-of-2
// (UE8M0) block scales requantize the transpose exactly, so one copy suffices —
// leave it off there. Elementwise consumers (swiglu backward reading gate/up)
// need no transpose, so those stashes are exempt either way.
export function analyze(nodes, marks, transposedStash = false) {
  const byId = Object.fromEntries(nodes.map(n => [n.id, n]));
  const saved = (n) => n.always || n.nomark || marks[n.id] === true;
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
  // torch_remat semantics: a \u21bb mark MEANS recompute — the op replays even
  // if no backward consumer demands its output (so marking everything costs
  // exactly 1\u00d7 fwd), and its inputs must be available (a useless mark can
  // honestly INCREASE the stash). Demand still decides what is stashed.
  for (const n of nodes) if (!saved(n) && !replayed.has(n.id)) {
    replayed.add(n.id);
    for (const i of n.inputs) need(i, n.id);
  }
  // the checkpoint region saves its own input: the anchor every replay reads
  neededSaved.add('x0');
  if (!neededBy.has('x0')) neededBy.set('x0', new Set(['replay anchor']));
  const buckets = { mla: 0, moe: 0, residual: 0 };
  const savedById = {};                              // per-tensor stash bytes (incl. fp8ᵀ dual copies and aux artifacts)
  let savedBytes = 0;
  const dual = new Set();                            // stashes kept in both fp8 orientations
  for (const id of neededSaved) {
    const n = byId[id];
    let bytes = n.outBytes;
    // < 1.2 B/elem = the tile-scaled fp8 class only: the transpose problem is
    // the 1×128 per-row scales, so E5M6 (1.5 B, no tile scales) is exempt
    if (transposedStash && n.outBytes / n.elems < 1.2
      && [...(neededBy.get(id) ?? [])].some(c => c !== id && ['matmul', 'attn'].includes(byId[c]?.opKind))) {
      dual.add(id);
      bytes *= 2;
    }
    buckets[n.bucket] += bytes;
    savedById[id] = (savedById[id] ?? 0) + bytes;
    savedBytes += bytes;
  }
  for (const n of nodes) {         // aux artifacts (rstd, lse): a replay regenerates them
    if (n.aux && !replayed.has(n.id)) {
      buckets[n.bucket] += n.aux.bytes;
      savedById[n.id] = (savedById[n.id] ?? 0) + n.aux.bytes;
      savedBytes += n.aux.bytes;
    }
  }
  const replayFlopsTok = [...replayed].reduce((t, id) => t + byId[id].flopsTok, 0);
  const fwdFlopsTok = nodes.reduce((t, n) => t + n.flopsTok, 0);
  const replayComm = ['dispatch', 'combine'].filter(id => replayed.has(id));
  // POINTLESS recomputes: marked \u21bb, so they replay (literal semantics),
  // but nothing in backward — no bwd op, no other replay — reads the output.
  // A demand-driven planner would skip them; torch_remat does what you said.
  const pointless = new Set([...replayed].filter(id => !neededBy.has(id)));
  return {
    buckets, savedBytes, savedById, replayFlopsTok, fwdFlopsTok,
    replayFrac: fwdFlopsTok ? replayFlopsTok / fwdFlopsTok : 0,
    neededSaved, replayed, replayComm, neededBy, byId, dual, pointless,
  };
}
