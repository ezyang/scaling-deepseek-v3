// The quantity graph ("cells") behind the full-model fit chart: every number
// the chart shows is a cell — a value computed by evaluating a FORMULA
// STRING over other cells. The string the reader sees IS the expression the
// engine evaluates (one source of truth: a label can't diverge from the
// math), so the chart, the hover tooltips, and the spreadsheet figure all
// agree by construction. Pure module (no DOM) so Node can cross-check it.
//
// Cell ids double as the display names (spreadsheet coordinates,
// [A-Z]\d+[a-z]? — the trailing letter is a per-tensor sub-row), lettered
// by SECTION:
//   H the architecture (hidden · experts · ranks · head geometry — constants)
//   P parallelism & schedule (GPUs · PP · EP · DP · EDP · mb in flight · tokens/mb)
//   S sharding (S1 the ZeRO level; S2–S7 the per-component shard groups)
//   L this rank's layout (MoE/dense layers · vocab matrices · emb?/head?)
//   N parameter counts · Q params on this GPU · F format flags (F1 fp8ᵀ
//   -resident params) · W/G/O weights/gradients/optimizer bytes ·
//   D per-token stash rates · A activations (A1 total · A2… buckets ·
//   A2a… tensors) · R kept? choices and B precision inputs, numbered to
//   MATCH their A row · T1 the total
// A cell without an expr is a LEAF: its value is injected by the caller
// (slot-split layer counts, op-graph stash rates) and drill-down ends there.

// ---- the mini formula language: numbers, cell ids, + - × / ( ) and ≥ ------
// (≥ evaluates to 0/1 — indicator arithmetic keeps piecewise rules, like the
// ZeRO shard groups, expressible without the formula changing shape)
import { ACT_BUCKETS, actBucketsOf, ppStage, LOCAL_PAR, ilvPeak } from './localmodel.js';
import { PARAMS } from './params.js';
import { markKey } from './blockgraph.js';
import { DSV3 } from './model.js';

// H — the architecture: named dimensions as CELLS, so parameter counts and
// stash formulas read semantically (H3 × 3 × H1 × H2) instead of as opaque
// numerals. Values come from the same DSV3 dict the whole engine runs on.
export const ARCH_CELLS = [
  ['H1', 'hidden', 'hidden dim'],
  ['H2', 'moeInter', 'expert FFN inter dim'],
  ['H3', 'routedExperts', 'routed experts'],
  ['H4', 'topk', 'top-k routed per token'],
  ['H5', 'denseInter', 'dense-layer FFN inter dim'],
  ['H6', 'vocab', 'vocabulary'],
  ['H7', 'qRank', 'q latent rank'],
  ['H8', 'kvRank', 'kv latent rank'],
  ['H9', 'heads', 'attention heads'],
  ['H10', 'qkNope', 'head dim · qk (nope)'],
  ['H11', 'qkRope', 'head dim · rope'],
  ['H12', 'vHead', 'head dim · v'],
  ['H13', 'sharedExperts', 'shared experts'],
];
const archGet = (id) => { const r = ARCH_CELLS.find((x) => x[0] === id); return r ? DSV3[r[1]] : NaN; };

const AST = new Map();   // expr string → parsed tree (exprs are static per state)
function parse(src) {
  if (AST.has(src)) return AST.get(src);
  let i = 0;
  const ws = () => { while (src[i] === ' ') i++; };
  const expr = () => {
    let v = add();
    ws();
    if (src[i] === '≥') { i++; v = { op: '≥', a: v, b: add() }; }
    return v;
  };
  const add = () => {
    let v = term();
    for (ws(); src[i] === '+' || src[i] === '-'; ws()) { const op = src[i++]; v = { op, a: v, b: term() }; }
    return v;
  };
  const term = () => {
    let v = factor();
    for (ws(); src[i] === '×' || src[i] === '/'; ws()) { const op = src[i++]; v = { op, a: v, b: factor() }; }
    return v;
  };
  const factor = () => {
    ws();
    if (src[i] === '(') { i++; const v = expr(); ws(); if (src[i++] !== ')') throw new Error(`cells: missing ) in "${src}"`); return v; }
    const m = /^(\d+(?:\.\d+)?)/.exec(src.slice(i));
    if (m) { i += m[0].length; return { num: +m[0] }; }
    const c = /^([A-Z]\d+[a-z]?)/.exec(src.slice(i));
    if (c) { i += c[0].length; return { ref: c[0] }; }
    throw new Error(`cells: bad token at "${src.slice(i)}"`);
  };
  const tree = expr();
  ws();
  if (i !== src.length) throw new Error(`cells: trailing "${src.slice(i)}"`);
  AST.set(src, tree);
  return tree;
}
export function evalExpr(src, get) {
  const go = (n) => n.num != null ? n.num
    : n.ref ? get(n.ref)
      : n.op === '≥' ? (go(n.a) >= go(n.b) ? 1 : 0)
        : n.op === '+' ? go(n.a) + go(n.b)
          : n.op === '-' ? go(n.a) - go(n.b)
            : n.op === '×' ? go(n.a) * go(n.b) : go(n.a) / go(n.b);
  return go(parse(src));
}
export const refsOf = (src) => [...new Set(src.match(/[A-Z]\d+[a-z]?/g) ?? [])];

// ---- the cell definitions, built for one widget state ----------------------
// env: { world, pp, ep, zero, sched, fp8p, g: {moe, dense, emb, head},
//        aM, aD (stash B/token per layer kind),
//        bM, bD (per-ACT-BUCKET B/token rates), bLabels (the bucket names),
//        N: { restLayer, denseLayer } (parameter-count leaves, from PARAMS) }
// Units: 'B' bytes · 'p' parameter counts · 'B/tok' · absent = plain counts.
export function buildCells(env) {
  const { g, zero, fp8p } = env;
  // schedule-dependent shape: under Megatron's interleaved 1F1B the two layer
  // kinds see DIFFERENT in-flight counts (chunk-weighted: P6 for MoE layers,
  // P6d for dense), so every activation formula wears the in-flight factor
  // per kind — (L1 × P6) × rate + (L2 × P6d) × rate — instead of one trailing
  // × P6. DualPipeV / ×1 mb keep 02's shape byte-for-byte.
  const ILV = env.sched === 'interleaved';
  const gradB = env.gradB ?? 4;   // the gradient buffer's bytes/param (02: fp32 = 4; Megatron's perf recipes: bf16 = 2)
  const L1w = ILV ? '(L1 × P6)' : 'L1', L2w = ILV ? '(L2 × P6d)' : 'L2', P6t = ILV ? '' : ' × P6';
  // one CLASS of one component (the accordion sub-rows): q = Q1/Q2/Q3,
  // S• = its ZeRO shard-group INPUT (the group size when this level shards
  // the component, else 1 — so the formula never changes shape, only the
  // input's value), w8 = the fp8-resident weights get the F1 flag IN the
  // formula — both e4m3 orientations pay one fp32 scale per 1×128 tile,
  // 2 + F1 × 2 × 4/128. The component TOTALS are the sums of these rows:
  // the accordion IS the computation.
  const cls = (bpp, q, S, w8) => `${w8 ? (env.mx ? '(2 + F1 × 2 × 1/32)' : '(2 + F1 × 2 × 4/128)') : String(bpp)} × ${q} / ${S}`;
  // sheet-side alternate names (the diagram keeps the graph's tensor
  // names): dispatched tokens ARE the routed experts' gate/up input; the
  // shared expert reads the post-norm2 stream directly (pre-a2a)
  const ALIAS = {
    'dispatched tokens': 'dispatched tokens (routed experts’ input)',
    'norm2 out': 'norm2 out (shared expert input)',
  };
  const alias = (l2) => ALIAS[l2] ?? l2;
  // noScale (a sheet display mode, like simplify): count fp8 ACTIVATION
  // stashes at their payload rate — the 1×128 tile-scale share (1/32 B/elem
  // per copy) drops out of the B• inputs; weights (F1's factor) keep theirs
  const descale = !env.noScale ? (v) => v : (v) => v === 1 + 1 / 32 ? 1 : v === 2 + 2 / 32 ? 2 : v;
  const nBk = env.bM?.length ?? 0;   // act buckets (A2…)
  const bucketSum = Array.from({ length: nBk }, (_, i) => `A${i + 2}`).join(' + ');
  // per-bucket rows: the recompute CHOICE is an explicit 0/1 input (R•)
  // multiplying the bucket's save-everything rates at the current recipe —
  // flipping the policy flips R, not the formula. A bucket a policy keeps
  // only PARTIALLY (the catch-all's aux artifacts; x1 under full) falls
  // back to its as-is rates, labeled so.
  const buckets = (env.bM ?? []).flatMap((rM, i) => {
    const rD = env.bD[i], fM = env.bMF?.[i] ?? rM, fD = env.bDF?.[i] ?? rD;
    const last = i === nBk - 1, tail = last ? ' + D3 × P7' : '';
    const lbl = `${alias(env.bLabels[i])}${last ? ' (+ vocab D3)' : ''}`;   // indented under A1 — no 'stash ·' prefix
    const R = env.bRate?.[i];
    // BREAKOUT buckets (residual, norm outs, the remainder): one sub-cell
    // per TENSOR, each a whole 0/1 kept? choice — no bucket ever reads
    // 'partial' just because a policy split it (x0 vs x1; norm1 vs norm2)
    if (R?.tensors) {
      const L = 'abcdefgh';
      // simplify drops the aux rows (lse, rstd) — negligible terms; the
      // parent sum shrinks with them, so the simplified sheet's totals may
      // drift a little from the (always exact) chart
      const shown = R.tensors.map((t, j) => ({ t, sid: `A${i + 2}${L[j]}` }))
        .filter(({ t }) => !(env.simplify && t.aux));
      const dtEdit = (dtc) => dtc == null ? undefined
        : dtc === 'o_proj' ? { t: 'cb', k: 'e5m6' } : { t: 'dt', k: dtc };
      let lastRid = null;   // an aux artifact is gated by ITS TENSOR's kept? —
      // one choice controls both (an aux row never gets its own R)
      const rows = shown.flatMap(({ t, sid }) => {
        const rid = `R${sid.slice(1)}`, ui = { c: t.id };
        const mkEdit = t.aux ? undefined : { t: 'mark', k: markKey(t.id) };
        if (t.aux) {
          const gate = lastRid ?? rid;
          const rate = [t.fMv ? `${L1w} × ${t.fMv}` : null, t.fDv ? `${L2w} × ${t.fDv}` : null].filter(Boolean).join(' + ');
          return [
            { id: sid, depth: 2, unit: 'B', label: t.label, ui, expr: `${gate} × (${rate}) × P7${P6t}` },
            ...(lastRid ? [] : [{ id: rid, depth: 3, label: 'kept?', ui, value: t.r }]),
          ];
        }
        lastRid = null;
        if (!t.whole) return [{ id: sid, depth: 2, unit: 'B', label: `${alias(t.label)} (partial under policy)`, ui,
          expr: `(${L1w} × ${t.cMv} + ${L2w} × ${t.cDv}) × P7${P6t}` }];
        const p1 = t.fMv ? `${L1w} × ${t.tM ? `(${t.tM})` : t.fMv}` : null;
        const p2 = t.fDv ? `${L2w} × ${t.tD ? `(${t.tD})` : t.fDv}` : null;
        const rate = [p1, p2].filter(Boolean).join(' + ');
        if (!rate) return [{ id: sid, depth: 2, unit: 'B', label: alias(t.label), ui, value: 0 }];
        lastRid = rid;
        return [
          { id: sid, depth: 2, unit: 'B', label: alias(t.label), ui, expr: `${rid} × (${rate}) × P7${P6t}` },
          { id: rid, depth: 3, label: 'kept?', ui, edit: mkEdit, value: t.r },
          ...(t.prec != null ? [{ id: t.bref, depth: 3, unit: 'B/e', label: 'precision (B/elem)', ui, edit: dtEdit(t.dtc), value: descale(t.prec) }] : []),
        ];
      });
      const subIds = shown.map(({ sid }) => sid);
      return [
        { id: `A${i + 2}`, unit: 'B', depth: 1, label: lbl, ui: { c: R.tensors[0]?.id ?? env.bIds?.[i] },
          expr: (subIds.length ? subIds.join(' + ') : '0') + tail },
        ...rows,
      ];
    }
    const whole = (fM > 0 || fD > 0) && ((rM === fM && rD === fD) || (rM === 0 && rD === 0));
    if (!whole) return [{ id: `A${i + 2}`, unit: 'B', depth: 1, label: `${lbl} (partial under policy)`,
      ui: env.bIds?.[i] ? { c: env.bIds[i] } : undefined,
      expr: `(${L1w} × ${rM} + ${L2w} × ${rD}) × P7${P6t}${tail}` }];
    // the gate/up bucket is ONE stashed graph node (the SwiGLU-input
    // quantize's output) whose elems span routed + shared (+ the dense MLP
    // in dense layers): split it for display — validated: the sub-dims must
    // sum exactly to the node's rates
    const GS = env.gateSplit;
    if (GS && i === GS.i && R?.prec != null) {
      const B9 = `B${i + 2}`, R9 = `R${i + 2}`;
      const rB = evalExpr(GS.routed, archGet) * R.prec, sB = evalExpr(GS.shared, archGet) * R.prec;
      const dB = evalExpr(GS.dense, archGet) * R.prec;
      if (rB + sB === fM && dB === fD) {
        const kept = rM === fM && rD === fD ? 1 : 0;
        const ui9 = { c: env.bIds[i] };
        return [
          { id: `A${i + 2}`, unit: 'B', depth: 1, label: lbl, ui: ui9,
            expr: `A${i + 2}a + A${i + 2}b + A${i + 2}c${tail}` },
          { id: `A${i + 2}a`, depth: 2, unit: 'B', label: 'gate, up · routed (routed experts’ hidden, pre-SwiGLU)', ui: ui9,
            expr: `${R9} × ${L1w} × (${GS.routed} × ${B9}) × P7${P6t}` },
          { id: `A${i + 2}b`, depth: 2, unit: 'B', label: 'gate, up · shared (shared expert hidden, pre-SwiGLU)', ui: { c: `${env.bIds[i]}:sh` },
            expr: `${R9} × ${L1w} × (${GS.shared} × ${B9}) × P7${P6t}` },
          { id: `A${i + 2}c`, depth: 2, unit: 'B', label: 'gate, up · dense MLP (dense layers’ hidden)', ui: ui9,
            expr: `${R9} × ${L2w} × (${GS.dense} × ${B9}) × P7${P6t}` },
          { id: R9, depth: 2, label: 'kept?', ui: ui9, edit: { t: 'mark', k: markKey(env.bIds[i]) }, value: kept },
          { id: B9, depth: 2, unit: 'B/e', label: 'precision (B/elem)', ui: ui9,
            edit: R.dtc === 'o_proj' ? { t: 'cb', k: 'e5m6' } : { t: 'dt', k: R.dtc }, value: descale(R.prec) },
        ];
      }
    }
    // the rate DECOMPOSITION (per saved tensor: dims × B•, + fp32 aux; the
    // ᵀ dual folds into B•'s value) when the caller validated one;
    // zero-rate kinds drop out
    const t1 = fM ? `${L1w} × ${R?.eM ? `(${R.eM})` : fM}` : null;
    const t2 = fD ? `${L2w} × ${R?.eD ? `(${R.eD})` : fD}` : null;
    const rate = [t1, t2].filter(Boolean).join(' + ') || 'L1 × 0 + L2 × 0';
    const ui = env.bIds?.[i] ? { c: env.bIds[i] } : undefined;
    const dtE = R?.dtc == null ? undefined
      : R.dtc === 'o_proj' ? { t: 'cb', k: 'e5m6' } : { t: 'dt', k: R.dtc };
    return [
      { id: `A${i + 2}`, unit: 'B', depth: 1, label: lbl, ui,
        expr: `R${i + 2} × (${rate}) × P7${P6t}${tail}` },
      { id: `R${i + 2}`, depth: 2, label: 'kept?', ui,
        edit: env.bIds?.[i] ? { t: 'mark', k: markKey(env.bIds[i]) } : undefined,
        value: rM === fM && rD === fD ? 1 : 0 },
      ...(R ? [{ id: `B${i + 2}`, depth: 2, unit: 'B/e', label: 'precision (B/elem)', ui, edit: dtE, value: descale(R.prec) }] : []),
    ];
  });
  // N sub-rows: the per-weight decomposition, written in H refs and
  // VALIDATED against the graph-derived totals the env carries (sanity
  // additionally proves those against the checkpoint-exact PARAMS) — a
  // mismatch falls back to opaque value rows, never a wrong formula.
  const N2ROWS = [
    ['N2a', 'q/kv down-proj', 'H1 × H7 + H1 × (H8 + H11)'],
    ['N2b', 'q up-proj', 'H7 × H9 × (H10 + H11)'],
    ['N2c', 'kv up-proj', 'H8 × H9 × (H10 + H12)'],
    ['N2d', 'attn out-proj', 'H9 × H12 × H1'],
    ['N2e', 'RMSNorms (norm1 · norm2 · 2 latent norms)', '2 × H1 + H7 + H8'],
    ['N2f', 'router (weight + bias)', '(H1 + 1) × H3'],
    ['N2g', 'shared expert (gate/up/down)', 'H13 × 3 × H1 × H2'],
  ];
  const N3A = '3 × H1 × H5';
  const nv = (e) => evalExpr(e, archGet);
  const n1ok = env.N.routed == null || nv('H3 × 3 × H1 × H2') === env.N.routed;   // no target (the oracle's minimal env) = trust the arch
  const nOk = N2ROWS.reduce((t, [, , e]) => t + nv(e), 0) === env.N.restLayer
    && N2ROWS.slice(0, 5).reduce((t, [, , e]) => t + nv(e), 0) + nv(N3A) === env.N.denseLayer;
  const defs = [
    { id: 'P1', label: 'GPUs in the cluster', value: env.world, ui: { k: 'gpus' }, edit: { t: 'step', k: 'gpus' } },
    { id: 'P2', label: 'pipeline stages (PP)', value: env.pp, ui: { k: 'pp' }, edit: { t: 'step', k: 'pp' } },
    { id: 'P3', label: 'expert parallelism (EP)', value: env.ep, ui: { k: 'ep' }, edit: { t: 'step', k: 'ep' } },
    { id: 'P4', label: 'data parallelism (non-expert params: GPUs ÷ PP ÷ TP)', expr: 'P1 / (P2 × P11)' },
    { id: 'P5', label: 'expert data parallelism (GPUs ÷ PP ÷ EP — TP widens it: expert-tensor-parallel is 1)', expr: 'P4 × P11 / P3' },
    ...(ILV ? [
      // Megatron interleaved 1F1B: the rank's peak chunk-stashes (P10) are
      // dealt over its VP chunks in groups of PP, so each layer KIND's
      // in-flight count is the chunk-weighted mean over that kind's layers
      // — the literals are (layers × stashes) per chunk, grouped by count
      { id: 'P6', label: `microbatches in flight · MoE layers (chunks hold ${(env.ifLive ?? []).join(',')} at the byte peak)`, ui: { k: 'sched' }, edit: { t: 'flip', k: 'sched' },
        ...(env.ifExpr?.moe ? { expr: env.ifExpr.moe } : { value: env.ifM ?? 1 }) },
      { id: 'P6d', depth: 1, label: 'microbatches in flight · dense layers (same moment)', ui: { k: 'sched' },
        ...(env.ifExpr?.dense ? { expr: env.ifExpr.dense } : { value: env.ifD ?? 1 }) },
      { id: 'P8', label: 'virtual pipeline stages per rank (VP)', value: env.vpp ?? 1, ui: { k: 'vpp' }, edit: { t: 'step', k: 'vpp' } },
      { id: 'P9', label: 'this rank (0-based)', value: Math.min(env.stage ?? 0, env.pp - 1), ui: { k: 'rank' }, edit: { t: 'step', k: 'rank' } },
      { id: 'P10', label: `chunk-stashes at the peak on this rank (warmup 2(PP−r−1) + (VP−1)·PP${env.a2a ? ' + 1 (a2a overlap)' : ''}, plus the steady-state forward)`,
        expr: env.pp === 1 ? String(env.vpp ?? 1) : env.vpp === 1 ? 'P2 - P9' : env.a2a ? 'P2 × P8 + P2 - 2 × P9' : 'P2 × P8 + P2 - 2 × P9 - 1', ui: { k: 'sched' } },
    ] : [
      { id: 'P6', label: 'microbatches in flight' + (env.sched !== 'one' && env.pp > 1 ? ' (DualPipeV: PP + ½)' : ''),
        expr: env.sched === 'one' || env.pp === 1 ? '1' : 'P2 + 0.5', ui: { k: 'sched' }, edit: { t: 'flip', k: 'sched' } },
    ]),
    // formula-switching INPUTS get explicit rows: the ZeRO level picks which
    // components wear a /P4·/P5 sharding term; the fp8-params flag rides the
    // weights formulas as a 0/1 factor
    // one 4096-token sequence per microbatch; under TP every stash divides by
    // TP — sequence parallel shards the residual/MoE path by tokens, attention
    // shards heads, the head shards the vocabulary (same bytes either way)
    { id: 'P7', label: 'tokens per microbatch, this GPU\u2019s share (seq 4096 · mbs 1, ÷ TP under sequence parallel)', expr: '4096 / P11' },
    { id: 'P11', label: 'tensor parallelism (TP; sequence parallel on, expert-tensor-parallel 1)', value: env.tp ?? 1, ui: { k: 'tp' }, edit: { t: 'step', k: 'tp' } },
    { id: 'S1', label: 'ZeRO level (1 optim · 2 +grads · 3 +weights)', value: zero, ui: { k: 'zero' }, edit: { t: 'seg', k: 'zero' } },
    // the level resolves to per-component SHARD GROUPS (1 = unsharded) via
    // indicator arithmetic — the byte formulas below never change shape
    // when Z1 moves, and neither do these
    { id: 'S2', depth: 1, ui: { k: 'zero' }, label: 'shard group · weights, experts', expr: '(S1 ≥ 3) × (P5 - 1) + 1' },
    { id: 'S3', depth: 1, ui: { k: 'zero' }, label: 'shard group · weights, others', expr: '(S1 ≥ 3) × (P4 - 1) + 1' },
    { id: 'S4', depth: 1, ui: { k: 'zero' }, label: 'shard group · gradients, experts', expr: '(S1 ≥ 2) × (P5 - 1) + 1' },
    { id: 'S5', depth: 1, ui: { k: 'zero' }, label: 'shard group · gradients, others', expr: '(S1 ≥ 2) × (P4 - 1) + 1' },
    { id: 'S6', depth: 1, ui: { k: 'zero' }, label: 'shard group · optimizer, experts', expr: '(S1 ≥ 1) × (P5 - 1) + 1' },
    { id: 'S7', depth: 1, ui: { k: 'zero' }, label: 'shard group · optimizer, others', expr: '(S1 ≥ 1) × (P4 - 1) + 1' },
    { id: 'F1', label: env.mx ? 'mxfp8-resident params (row + column copies)? (0/1)' : 'e4m3+ᵀ-resident params? (0/1)', value: fp8p ? 1 : 0, ui: { k: 'fp8params' }, edit: { t: 'cb', k: 'fp8params' } },
    { id: 'L1', label: 'MoE layers on this rank (slot split)', value: g.moe, ui: { k: 'rank' }, edit: { t: 'flip', k: 'rank' } },
    { id: 'L2', label: 'dense layers on this rank (slot split)', value: g.dense, ui: { k: 'rank' }, edit: { t: 'flip', k: 'rank' } },
    { id: 'L3', label: 'vocab matrices on this rank', expr: 'L4 + L5' },
    { id: 'L4', depth: 1, label: 'embedding on this rank? (0/1)', value: g.emb ? 1 : 0, ui: { k: 'rank' }, edit: { t: 'flip', k: 'rank' } },
    { id: 'L5', depth: 1, label: 'lm head on this rank? (0/1)', value: g.head ? 1 : 0, ui: { k: 'rank' }, edit: { t: 'flip', k: 'rank' } },
    ...ARCH_CELLS.map(([id, key, label]) => ({ id, label, value: DSV3[key], note: '(architecture)' })),
    { id: 'N1', label: 'params · routed experts, one MoE layer', unit: 'p',
      expr: n1ok ? 'H3 × 3 × H1 × H2' : undefined, value: n1ok ? undefined : env.N.routed },
    ...(nOk ? [
      { id: 'N2', label: 'params · rest of a MoE layer', unit: 'p', expr: 'N2a + N2b + N2c + N2d + N2e + N2f + N2g' },
      ...N2ROWS.map(([id, label, expr]) => ({ id, depth: 1, unit: 'p', label, expr })),
      { id: 'N3', label: 'params · one dense layer', unit: 'p', expr: 'N2a + N2b + N2c + N2d + N2e + N3a' },
      { id: 'N3a', depth: 1, unit: 'p', label: 'FFN gate/up/down (dense)', expr: N3A },
      // the TP split of those (Megatron's DSv3 spec): up-projections, out-proj,
      // shared expert and dense FFN shard over TP; the down-projections, norms
      // and router are replicated on every TP rank
      { id: 'N5', label: 'params · MoE layer, TP-sharded (q/kv up, attn out, shared expert)', unit: 'p', expr: 'N2b + N2c + N2d + N2g' },
      { id: 'N6', label: 'params · MoE layer, TP-replicated (q/kv down, norms, router)', unit: 'p', expr: 'N2a + N2e + N2f' },
      { id: 'N7', label: 'params · dense layer, TP-sharded (q/kv up, attn out, dense FFN)', unit: 'p', expr: 'N2b + N2c + N2d + N3a' },
      { id: 'N8', label: 'params · dense layer, TP-replicated (q/kv down, norms)', unit: 'p', expr: 'N2a + N2e' },
    ] : [
      { id: 'N2', label: 'params · rest of a MoE layer', unit: 'p', value: env.N.restLayer },
      { id: 'N3', label: 'params · one dense layer', unit: 'p', value: env.N.denseLayer },
      { id: 'N5', label: 'params · MoE layer, TP-sharded', unit: 'p', value: env.N.restLayerTp ?? env.N.restLayer },
      { id: 'N6', label: 'params · MoE layer, TP-replicated', unit: 'p', value: env.N.restLayer - (env.N.restLayerTp ?? env.N.restLayer) },
      { id: 'N7', label: 'params · dense layer, TP-sharded', unit: 'p', value: env.N.denseLayerTp ?? env.N.denseLayer },
      { id: 'N8', label: 'params · dense layer, TP-replicated', unit: 'p', value: env.N.denseLayer - (env.N.denseLayerTp ?? env.N.denseLayer) },
    ]),
    { id: 'N4', label: 'params · one vocab matrix', unit: 'p', expr: 'H6 × H1' },
    { id: 'Q1', label: 'expert params on this GPU', unit: 'p', expr: 'L1 × N1 / P3' },
    { id: 'Q2', label: 'non-expert block params on this GPU (sharded parts ÷ TP, replicated parts whole)', unit: 'p', expr: 'L1 × (N5 / P11 + N6) + L2 × (N7 / P11 + N8)' },
    { id: 'Q3', label: 'vocab params on this GPU (+ final norm)', unit: 'p',
      expr: env.simplify ? 'L3 × N4 / P11' : 'L3 × N4 / P11 + L5 × H1' },   // vocab-parallel; the 7 K final norm (replicated) is a simplify casualty
    { id: 'W1', label: env.mx ? 'weights (2 B bf16; F1 flips block params mxfp8 row+col)' : 'weights (2 B bf16; F1 flips block params e4m3+ᵀ)', unit: 'B', expr: 'W2 + W3 + W4' },
    { id: 'W2', depth: 1, label: 'experts', unit: 'B', expr: cls(2, 'Q1', 'S2', true) },
    { id: 'W3', depth: 1, label: 'non-expert blocks', unit: 'B', expr: cls(2, 'Q2', 'S3', true) },
    { id: 'W4', depth: 1, label: 'emb + lm head (bf16 always)', unit: 'B', expr: cls(2, 'Q3', 'S3', false) },
    { id: 'G1', label: gradB === 2 ? 'gradients (bf16 grad buffer, 2 B/param)' : 'gradients (fp32, 4 B/param)', unit: 'B', expr: 'G2 + G3 + G4' },
    { id: 'G2', depth: 1, label: 'experts', unit: 'B', expr: cls(gradB, 'Q1', 'S4', false) },
    { id: 'G3', depth: 1, label: 'non-expert blocks', unit: 'B', expr: cls(gradB, 'Q2', 'S5', false) },
    { id: 'G4', depth: 1, label: 'emb + lm head', unit: 'B', expr: cls(gradB, 'Q3', 'S5', false) },
    { id: 'O1', label: 'optimizer states (8 B/param)', unit: 'B', expr: 'O2 + O3 + O4' },
    { id: 'O2', depth: 1, label: 'experts', unit: 'B', expr: cls(8, 'Q1', 'S6', false) },
    { id: 'O3', depth: 1, label: 'non-expert blocks', unit: 'B', expr: cls(8, 'Q2', 'S7', false) },
    { id: 'O4', depth: 1, label: 'emb + lm head', unit: 'B', expr: cls(8, 'Q3', 'S7', false) },
    { id: 'D1', label: 'stash/token · one MoE layer (the chips’ sum)', unit: 'B/tok', value: env.aM },
    { id: 'D2', label: 'stash/token · one dense layer', unit: 'B/tok', value: env.aD },
    { id: 'D3', label: 'stash/token · vocab side (x0 / logits + loss)', unit: 'B/tok',
      expr: 'L4 × 2 × H1 + L5 × 6 × H6' },
    // the acts total is the SUM OF ITS ACCORDION (the buckets partition the
    // op graph's savedBytes exactly, so this equals (L1×D1 + L2×D2) × 4096
    // × P6 + D3 × 4096 — D1/D2 stay as the per-layer summary rates)
    { id: 'A1', label: 'saved activations', unit: 'B',
      expr: nBk ? bucketSum : `(${L1w} × D1 + ${L2w} × D2) × P7${P6t} + D3 × P7` },
    // one cell per stash BUCKET (+ its 0/1 recompute-choice row): the
    // per-token rates ride the formula as exact literals (dyadic — their
    // decimal strings round-trip), sourced from the op graph per layer kind
    ...buckets,
    { id: 'T1', label: 'total resident bytes on this GPU', unit: 'B', expr: 'W1 + G1 + O1 + A1' },
  ];
  const byId = new Map(defs.map((d) => [d.id, d]));
  const get = (id) => {
    const d = byId.get(id);
    if (!d) throw new Error(`cells: unknown ${id}`);
    if (d.value === undefined) d.value = evalExpr(d.expr, get);
    return d.value;
  };
  for (const d of defs) { get(d.id); d.refs = d.expr ? refsOf(d.expr) : []; }
  return { cells: defs, byId, get };
}

// per-bucket rate DECOMPOSITIONS: instead of an opaque literal
// (59136), the formula shows where the bytes come from — per saved
// tensor, dims × B• where B• is the bucket's PRECISION INPUT cell
// (B/elem: 2 bf16 · 1.03125 e4m3+scales · 1.5 e5m6 · 4 fp32; a ᵀ
// dual stash FOLDS into the value — both orientations counted — so
// toggling ᵀ or the recipe changes the input, never the formula),
// + the fp32 aux artifacts (lse/rstd) as literals. Built from the
// SAVE-EVERYTHING analyses (the R• factorization multiplies full
// rates) and VALIDATED per kind: if a string does not evaluate back
// to the exact rate, the literal stands. A bucket whose saved
// tensors MIX effective precisions gets no B cell (none do today).
// BREAKOUT buckets — the ones a preset can keep only partially — get
// per-TENSOR sub-cells instead, so every row stays a whole 0/1 choice:
// residual (x0 pinned under full while x1 replays), norm outs (attn-
// replay keeps norm2 as the anchor while norm1 replays), mla latents
// (dsv3 keeps the latents but replays their norms — and it MIXES
// precisions, so precision inputs go per tensor there), and the
// remainder catch-all (its member set comes from the save-everything
// analysis, so it is policy-stable).
const BREAKOUT = new Set([0, 1, 2, 4, ACT_BUCKETS.length - 1]);
const NAMED = new Set(ACT_BUCKETS.flatMap((b) => b.ids));
const rateExprs = (AM, AD, curM, curD, bFM, bFD) => ACT_BUCKETS.map((b, k) => {
  // the catch-all's members: everything the save-everything analysis
  // stashes outside the named buckets (a policy-independent set)
  const ids = b.ids.length ? b.ids
    : [...new Set([...Object.keys(AM.savedById ?? {}), ...Object.keys(AD.savedById ?? {})])]
      .filter((id) => !NAMED.has(id));
  if (!ids.length && !BREAKOUT.has(k)) return null;
  let prec = null, mixed = false;
  // one tensor's rate expression for one layer KIND (dims × B• + aux);
  // bref names the precision input the terms reference
  const tExpr = (A, id, bref, withAux = false) => {
    const n2 = A.byId[id];
    if (!n2) return null;                       // the dense graph lacks router/dispatch
    const terms = [];
    if (A.neededSaved.has(id)) {
      const bpe = n2.outBytes / n2.elems * A.mul(id);   // ᵀ dual folds into the input
      if (prec == null) prec = bpe;
      else if (prec !== bpe) mixed = true;
      let dims = String(n2.elems);
      try { if (n2.tdims && evalExpr(n2.tdims, () => NaN) === n2.elems) dims = n2.tdims; } catch { /* keep the literal */ }
      if (dims.includes('+')) dims = `(${dims})`;   // multi-term tdims must bind before × B•
      terms.push(`${dims} × ${bref}`);
    }
    if (withAux && n2.aux && !A.replayed.has(id)) terms.push(String(n2.aux.bytes));
    return terms.length ? terms.join(' + ') : null;
  };
  const val = (e2, bref, p2) => {
    const get = (id2) => { if (id2 !== bref) throw new Error(id2); return p2; };
    try { return e2 == null ? 0 : evalExpr(e2, get); } catch { return NaN; }
  };
  if (BREAKOUT.has(k)) {
    // per-tensor precision inputs (B2a…): a breakout bucket may mix
    // formats (mla latents: bf16 latents + e4m3-rate norm outs). An
    // aux artifact (lse, rstd) SPLITS OUT as its own row — it is a
    // different quantity than the tensor it rides (and the simplify
    // view drops exactly these rows).
    let li = 0;
    const tensors = ids.flatMap((id) => {
      const nM = AM.byId[id], nD = AD.byId[id], n2 = nM ?? nD;
      if (!n2) return [];
      const bref = `B${k + 2}${'abcdefgh'[li]}`;
      prec = null; mixed = false;
      const tM = tExpr(AM, id, bref), tD = tExpr(AD, id, bref);
      const auxOf = (A, n3) => n3?.aux && !A.replayed.has(id) ? n3.aux.bytes : 0;
      // core bytes = the stash minus its aux (the aux gets its own row)
      const fMv = (AM.savedById?.[id] ?? 0) - auxOf(AM, nM), fDv = (AD.savedById?.[id] ?? 0) - auxOf(AD, nD);
      const cMv = (curM.savedById?.[id] ?? 0) - auxOf(curM, curM.byId[id]), cDv = (curD.savedById?.[id] ?? 0) - auxOf(curD, curD.byId[id]);
      const whole = (cMv === fMv && cDv === fDv) || (cMv === 0 && cDv === 0);
      const out = [{ id, bref, prec: mixed ? null : prec, dtc: n2.dtc ?? null,
        label: (n2.tensor?.replace(' (checkpoint anchor)', '') ?? id)
          + ((AM.copies?.[id] ?? 1) > 1 ? ` (×${AM.copies[id]}: q and kv down-proj each keep a copy)` : ''),
        tM, tD, fMv, fDv, cMv, cDv, whole, r: cMv === fMv && cDv === fDv ? 1 : 0 }];
      li++;
      if (n2.aux) {
        out.push({ aux: true, id: `${id}:aux`, label: `${n2.aux.name} (fp32) · ${out[0].label}`,
          fMv: nM?.aux?.bytes ?? 0, fDv: nD?.aux?.bytes ?? 0,
          whole: true, r: curM.replayed.has(id) || curD.replayed.has(id) ? 0 : 1 });
        li++;
      }
      return out;
    });
    // validation, per core tensor per kind — a miss drops the breakout
    for (const t of tensors) {
      if (t.aux) continue;
      const strip = (e2) => e2;   // tExpr excludes aux below — nothing to strip
      if (t.prec == null || val(strip(t.tM), t.bref, t.prec) !== t.fMv || val(strip(t.tD), t.bref, t.prec) !== t.fDv) return null;
    }
    // an EMPTY remainder is a valid breakout: the parent reads 0 (+ D3)
    return { tensors };
  }
  // NON-breakout buckets: one whole-bucket expression over a shared B•
  const bref = `B${k + 2}`;
  const kindExpr = (A) => {
    const terms = ids.map((id) => tExpr(A, id, bref, true)).filter(Boolean);
    return terms.length ? terms.join(' + ') : null;
  };
  const eM = kindExpr(AM), eD = kindExpr(AD);
  if (mixed || prec == null) return null;
  if (val(eM, bref, prec) !== bFM[k] || val(eD, bref, prec) !== bFD[k]) return null;
  return { eM, eD, prec, dtc: (AM.byId[ids[0]] ?? AD.byId[ids[0]])?.dtc ?? null };
});
// the CELL GRAPH (src/cells.js): every chart number is a cell — a
// value computed by evaluating the same formula string the tooltips
// and the spreadsheet display, so the numbers and their shown
// derivations cannot diverge (scripts/sanity.mjs replays the shard
// math independently and asserts === across a config matrix).
// assemble the FULL cell env from plain state S {world?, pp, ep, zero?,
// sched?, fp8p?, stage, vpp?, fold?} + the four analyses (current policy
// and save-everything, per kind). This is the production assembly the
// widget, the sheet, and the goldens all share.
export const cellsEnv = (S, anaM, anaD, anaMF, anaDF) => ({
  world: S.world ?? LOCAL_PAR.world, pp: S.pp, ep: S.ep, zero: S.zero ?? 1,
  sched: S.sched ?? '1f1b', fp8p: !!S.fp8p,
  vpp: S.vpp ?? 1, stage: S.stage, hw: S.hw ?? 'h100',
  g: ppStage(Math.min(S.stage, S.pp - 1), S.pp, S.vpp, S.fold, S.layout),
  a2a: !!S.a2a, gradB: S.gradB ?? 4, mx: !!S.mx, tp: S.tp ?? 1,   // mx: the instance's fp8 flavour is MXFP8 (F1's story + formula wording)
  ...ilvEnv(S, anaM.savedBytes, anaD.savedBytes),
  aM: anaM.savedBytes, aD: anaD.savedBytes,
  bM: actBucketsOf(anaM), bD: actBucketsOf(anaD), bLabels: ACT_BUCKETS.map((b) => b.label),
  bIds: ACT_BUCKETS.map((b) => b.ids[0] ?? null),
  gateSplit: { i: ACT_BUCKETS.findIndex((b) => b.ids[0] === 'quant'),
    routed: 'H4×2×H2', shared: 'H13×2×H2', dense: '2×H5' },
  bMF: actBucketsOf(anaMF ?? anaM), bDF: actBucketsOf(anaDF ?? anaD),
  bRate: anaMF && anaDF
    ? rateExprs(anaMF, anaDF, anaM, anaD, actBucketsOf(anaMF), actBucketsOf(anaDF))
    : null,
  N: { restLayer: PARAMS.moeBlock - moeExp, denseLayer: PARAMS.denseBlock, routed: moeExp },
});
const moeExp = PARAMS.expert * DSV3.routedExperts;
// interleaved-schedule inputs for the cells: the per-chunk live stashes at
// the rank's BYTE-peak moment (ilvPeak at the current policy's per-layer
// rates) and each kind's mean over its own layers as a formula string —
// (Σ layers × stashes, grouped by stash count) / the rank's layer count of
// that kind; a kind the rank lacks gets a plain value (its L• is 0)
const ilvEnv = (S, aM, aD) => {
  if ((S.sched ?? '1f1b') !== 'interleaved') return {};
  const pp = S.pp, vpp = S.vpp ?? 1, st = Math.min(S.stage, pp - 1);
  const pk = ilvPeak(st, pp, vpp, S.layout, { a2a: !!S.a2a, rateM: aM, rateD: aD });
  const out = { ifM: pk.moe, ifD: pk.dense, ifLive: pk.live, ifExpr: {} };
  for (const [kind, Lc] of [['moe', 'L1'], ['dense', 'L2']]) {
    const by = new Map();
    pk.segs.forEach((sg, c) => { if (sg[kind]) by.set(pk.live[c], (by.get(pk.live[c]) ?? 0) + sg[kind]); });
    if (!by.size) continue;
    out.ifExpr[kind] = `(${[...by.entries()].sort((a, b) => b[0] - a[0]).map(([k, L]) => `${L} × ${k}`).join(' + ')}) / ${Lc}`;
  }
  return out;
};
