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
//   P parallelism & schedule (GPUs · PP · EP · DP · EDP · mb in flight)
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
import { ACT_BUCKETS, actBucketsOf, ppStage, LOCAL_PAR } from './localmodel.js';
import { PARAMS } from './params.js';
import { DSV3 } from './model.js';

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
  // one CLASS of one component (the accordion sub-rows): q = Q1/Q2/Q3,
  // S• = its ZeRO shard-group INPUT (the group size when this level shards
  // the component, else 1 — so the formula never changes shape, only the
  // input's value), w8 = the fp8-resident weights get the F1 flag IN the
  // formula — both e4m3 orientations pay one fp32 scale per 1×128 tile,
  // 2 + F1 × 2 × 4/128. The component TOTALS are the sums of these rows:
  // the accordion IS the computation.
  const cls = (bpp, q, S, w8) => `${w8 ? '(2 + F1 × 2 × 4/128)' : String(bpp)} × ${q} / ${S}`;
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
    const last = i === nBk - 1, tail = last ? ' + D3 × 4096' : '';
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
        const mkEdit = t.aux ? undefined : { t: 'mark', k: t.id };
        if (t.aux) {
          const gate = lastRid ?? rid;
          const rate = [t.fMv ? `L1 × ${t.fMv}` : null, t.fDv ? `L2 × ${t.fDv}` : null].filter(Boolean).join(' + ');
          return [
            { id: sid, depth: 2, unit: 'B', label: t.label, ui, expr: `${gate} × (${rate}) × 4096 × P6` },
            ...(lastRid ? [] : [{ id: rid, depth: 3, label: 'kept?', ui, value: t.r }]),
          ];
        }
        lastRid = null;
        if (!t.whole) return [{ id: sid, depth: 2, unit: 'B', label: `${alias(t.label)} (partial under policy)`, ui,
          expr: `(L1 × ${t.cMv} + L2 × ${t.cDv}) × 4096 × P6` }];
        const p1 = t.fMv ? `L1 × ${t.tM ? `(${t.tM})` : t.fMv}` : null;
        const p2 = t.fDv ? `L2 × ${t.tD ? `(${t.tD})` : t.fDv}` : null;
        const rate = [p1, p2].filter(Boolean).join(' + ');
        if (!rate) return [{ id: sid, depth: 2, unit: 'B', label: alias(t.label), ui, value: 0 }];
        lastRid = rid;
        return [
          { id: sid, depth: 2, unit: 'B', label: alias(t.label), ui, expr: `${rid} × (${rate}) × 4096 × P6` },
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
      expr: `(L1 × ${rM} + L2 × ${rD}) × 4096 × P6${tail}` }];
    // the gate/up bucket is ONE graph node whose elems span routed + shared
    // (+ the dense MLP in dense layers): split it for display — validated:
    // the sub-dims must sum exactly to the node's rates
    const GS = env.gateSplit;
    if (GS && i === GS.i && R?.prec != null) {
      const B9 = `B${i + 2}`, R9 = `R${i + 2}`;
      const rB = evalExpr(GS.routed, () => NaN) * R.prec, sB = evalExpr(GS.shared, () => NaN) * R.prec;
      const dB = evalExpr(GS.dense, () => NaN) * R.prec;
      if (rB + sB === fM && dB === fD) {
        const kept = rM === fM && rD === fD ? 1 : 0;
        const ui9 = { c: env.bIds[i] };
        return [
          { id: `A${i + 2}`, unit: 'B', depth: 1, label: lbl, ui: ui9,
            expr: `A${i + 2}a + A${i + 2}b + A${i + 2}c${tail}` },
          { id: `A${i + 2}a`, depth: 2, unit: 'B', label: 'gate, up · routed (routed experts’ hidden, pre-SwiGLU)', ui: ui9,
            expr: `${R9} × L1 × (${GS.routed} × ${B9}) × 4096 × P6` },
          { id: `A${i + 2}b`, depth: 2, unit: 'B', label: 'gate, up · shared (shared expert hidden, pre-SwiGLU)', ui: { c: `${env.bIds[i]}:sh` },
            expr: `${R9} × L1 × (${GS.shared} × ${B9}) × 4096 × P6` },
          { id: `A${i + 2}c`, depth: 2, unit: 'B', label: 'gate, up · dense MLP (dense layers’ hidden)', ui: ui9,
            expr: `${R9} × L2 × (${GS.dense} × ${B9}) × 4096 × P6` },
          { id: R9, depth: 2, label: 'kept?', ui: ui9, edit: { t: 'mark', k: env.bIds[i] }, value: kept },
          { id: B9, depth: 2, unit: 'B/e', label: 'precision (B/elem)', ui: ui9,
            edit: R.dtc === 'o_proj' ? { t: 'cb', k: 'e5m6' } : { t: 'dt', k: R.dtc }, value: descale(R.prec) },
        ];
      }
    }
    // the rate DECOMPOSITION (per saved tensor: dims × B•, + fp32 aux; the
    // ᵀ dual folds into B•'s value) when the caller validated one;
    // zero-rate kinds drop out
    const t1 = fM ? `L1 × ${R?.eM ? `(${R.eM})` : fM}` : null;
    const t2 = fD ? `L2 × ${R?.eD ? `(${R.eD})` : fD}` : null;
    const rate = [t1, t2].filter(Boolean).join(' + ') || 'L1 × 0 + L2 × 0';
    const ui = env.bIds?.[i] ? { c: env.bIds[i] } : undefined;
    const dtE = R?.dtc == null ? undefined
      : R.dtc === 'o_proj' ? { t: 'cb', k: 'e5m6' } : { t: 'dt', k: R.dtc };
    return [
      { id: `A${i + 2}`, unit: 'B', depth: 1, label: lbl, ui,
        expr: `R${i + 2} × (${rate}) × 4096 × P6${tail}` },
      { id: `R${i + 2}`, depth: 2, label: 'kept?', ui,
        edit: env.bIds?.[i] ? { t: 'mark', k: env.bIds[i] } : undefined,
        value: rM === fM && rD === fD ? 1 : 0 },
      ...(R ? [{ id: `B${i + 2}`, depth: 2, unit: 'B/e', label: 'precision (B/elem)', ui, edit: dtE, value: descale(R.prec) }] : []),
    ];
  });
  const defs = [
    { id: 'P1', label: 'GPUs in the cluster', value: env.world, ui: { k: 'gpus' }, edit: { t: 'step', k: 'gpus' } },
    { id: 'P2', label: 'pipeline stages (PP)', value: env.pp, ui: { k: 'pp' }, edit: { t: 'step', k: 'pp' } },
    { id: 'P3', label: 'expert parallelism (EP)', value: env.ep, ui: { k: 'ep' }, edit: { t: 'step', k: 'ep' } },
    { id: 'P4', label: 'data parallelism', expr: 'P1 / P2' },
    { id: 'P5', label: 'expert data parallelism', expr: 'P4 / P3' },
    { id: 'P6', label: 'microbatches in flight' + (env.sched !== 'one' && env.pp > 1 ? ' (DualPipeV: PP + ½)' : ''),
      expr: env.sched === 'one' || env.pp === 1 ? '1' : 'P2 + 0.5', ui: { k: 'sched' }, edit: { t: 'flip', k: 'sched' } },
    // formula-switching INPUTS get explicit rows: the ZeRO level picks which
    // components wear a /P4·/P5 sharding term; the fp8-params flag rides the
    // weights formulas as a 0/1 factor
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
    { id: 'F1', label: 'e4m3+ᵀ-resident params? (0/1)', value: fp8p ? 1 : 0, ui: { k: 'fp8params' }, edit: { t: 'cb', k: 'fp8params' } },
    { id: 'L1', label: 'MoE layers on this rank (slot split)', value: g.moe, ui: { k: 'rank' }, edit: { t: 'flip', k: 'rank' } },
    { id: 'L2', label: 'dense layers on this rank (slot split)', value: g.dense, ui: { k: 'rank' }, edit: { t: 'flip', k: 'rank' } },
    { id: 'L3', label: 'vocab matrices on this rank', expr: 'L4 + L5' },
    { id: 'L4', depth: 1, label: 'embedding on this rank? (0/1)', value: g.emb ? 1 : 0, ui: { k: 'rank' }, edit: { t: 'flip', k: 'rank' } },
    { id: 'L5', depth: 1, label: 'lm head on this rank? (0/1)', value: g.head ? 1 : 0, ui: { k: 'rank' }, edit: { t: 'flip', k: 'rank' } },
    { id: 'N1', label: 'params · routed experts, one MoE layer', unit: 'p', expr: '256 × 3 × 7168 × 2048' },
    { id: 'N2', label: 'params · rest of a MoE layer', unit: 'p', value: env.N.restLayer },
    { id: 'N3', label: 'params · one dense layer', unit: 'p', value: env.N.denseLayer },
    { id: 'N4', label: 'params · one vocab matrix', unit: 'p', expr: '129280 × 7168' },
    { id: 'Q1', label: 'expert params on this GPU', unit: 'p', expr: 'L1 × N1 / P3' },
    { id: 'Q2', label: 'non-expert block params on this GPU', unit: 'p', expr: 'L2 × N3 + L1 × N2' },
    { id: 'Q3', label: 'vocab params on this GPU (+ final norm)', unit: 'p',
      expr: env.simplify ? 'L3 × N4' : 'L3 × N4 + L5 × 7168' },   // the 7 K final norm is a simplify casualty
    { id: 'W1', label: 'weights (2 B bf16; F1 flips block params e4m3+ᵀ)', unit: 'B', expr: 'W2 + W3 + W4' },
    { id: 'W2', depth: 1, label: 'experts', unit: 'B', expr: cls(2, 'Q1', 'S2', true) },
    { id: 'W3', depth: 1, label: 'non-expert blocks', unit: 'B', expr: cls(2, 'Q2', 'S3', true) },
    { id: 'W4', depth: 1, label: 'emb + lm head (bf16 always)', unit: 'B', expr: cls(2, 'Q3', 'S3', false) },
    { id: 'G1', label: 'gradients (fp32, 4 B/param)', unit: 'B', expr: 'G2 + G3 + G4' },
    { id: 'G2', depth: 1, label: 'experts', unit: 'B', expr: cls(4, 'Q1', 'S4', false) },
    { id: 'G3', depth: 1, label: 'non-expert blocks', unit: 'B', expr: cls(4, 'Q2', 'S5', false) },
    { id: 'G4', depth: 1, label: 'emb + lm head', unit: 'B', expr: cls(4, 'Q3', 'S5', false) },
    { id: 'O1', label: 'optimizer states (8 B/param)', unit: 'B', expr: 'O2 + O3 + O4' },
    { id: 'O2', depth: 1, label: 'experts', unit: 'B', expr: cls(8, 'Q1', 'S6', false) },
    { id: 'O3', depth: 1, label: 'non-expert blocks', unit: 'B', expr: cls(8, 'Q2', 'S7', false) },
    { id: 'O4', depth: 1, label: 'emb + lm head', unit: 'B', expr: cls(8, 'Q3', 'S7', false) },
    { id: 'D1', label: 'stash/token · one MoE layer (the chips’ sum)', unit: 'B/tok', value: env.aM },
    { id: 'D2', label: 'stash/token · one dense layer', unit: 'B/tok', value: env.aD },
    { id: 'D3', label: 'stash/token · vocab side (x0 / logits + loss)', unit: 'B/tok',
      expr: 'L4 × 2 × 7168 + L5 × 6 × 129280' },
    // the acts total is the SUM OF ITS ACCORDION (the buckets partition the
    // op graph's savedBytes exactly, so this equals (L1×D1 + L2×D2) × 4096
    // × P6 + D3 × 4096 — D1/D2 stay as the per-layer summary rates)
    { id: 'A1', label: 'saved activations', unit: 'B',
      expr: nBk ? bucketSum : '(L1 × D1 + L2 × D2) × 4096 × P6 + D3 × 4096' },
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
      const bpe = n2.outBytes / n2.elems * (A.dual.has(id) ? 2 : 1);   // ᵀ dual folds into the input
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
        label: n2.tensor?.replace(' (checkpoint anchor)', '') ?? id,
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
  g: ppStage(Math.min(S.stage, S.pp - 1), S.pp, S.vpp, S.fold),
  aM: anaM.savedBytes, aD: anaD.savedBytes,
  bM: actBucketsOf(anaM), bD: actBucketsOf(anaD), bLabels: ACT_BUCKETS.map((b) => b.label),
  bIds: ACT_BUCKETS.map((b) => b.ids[0] ?? null),
  gateSplit: { i: ACT_BUCKETS.findIndex((b) => b.ids[0] === 'gate_up'),
    routed: `${DSV3.topk}×2×${DSV3.moeInter}`, shared: `2×${DSV3.moeInter}`, dense: `2×${DSV3.denseInter}` },
  bMF: actBucketsOf(anaMF ?? anaM), bDF: actBucketsOf(anaDF ?? anaD),
  bRate: anaMF && anaDF
    ? rateExprs(anaMF, anaDF, anaM, anaD, actBucketsOf(anaMF), actBucketsOf(anaDF))
    : null,
  N: { restLayer: PARAMS.moeBlock - moeExp, denseLayer: PARAMS.denseBlock },
});
const moeExp = PARAMS.expert * DSV3.routedExperts;
