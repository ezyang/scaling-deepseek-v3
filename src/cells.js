// The quantity graph ("cells") behind the full-model fit chart: every number
// the chart shows is a cell — a value computed by evaluating a FORMULA
// STRING over other cells. The string the reader sees IS the expression the
// engine evaluates (one source of truth: a label can't diverge from the
// math), so the chart, the hover tooltips, and the spreadsheet figure all
// agree by construction. Pure module (no DOM) so Node can cross-check it.
//
// Cell ids double as the display names (spreadsheet coordinates, [A-Z]\d):
//   P* parallelism · L* this rank's layers · N* parameter counts ·
//   Q* params on this GPU · D* activation rates · W/G/O/A/T byte totals
// A cell without an expr is a LEAF: its value is injected by the caller
// (slot-split layer counts, op-graph stash rates) and drill-down ends there.

// ---- the mini formula language: numbers, cell ids, + - × / ( ) and ≥ ------
// (≥ evaluates to 0/1 — indicator arithmetic keeps piecewise rules, like the
// ZeRO shard groups, expressible without the formula changing shape)
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
    const lbl = `${env.bLabels[i]}${last ? ' (+ vocab D3)' : ''}`;   // indented under A1 — no 'stash ·' prefix
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
      const rows = shown.flatMap(({ t, sid }) => {
        const rid = `R${sid.slice(1)}`, ui = { c: t.id };
        const mkEdit = t.aux ? undefined : { t: 'mark', k: t.id };
        if (t.aux) {   // an aux artifact row: literal fp32 bytes, its own kept?
          const rate = [t.fMv ? `L1 × ${t.fMv}` : null, t.fDv ? `L2 × ${t.fDv}` : null].filter(Boolean).join(' + ');
          return [
            { id: sid, depth: 2, unit: 'B', label: t.label, ui, expr: `${rid} × (${rate}) × 4096 × P6` },
            { id: rid, depth: 3, label: 'kept?', ui, value: t.r },
          ];
        }
        if (!t.whole) return [{ id: sid, depth: 2, unit: 'B', label: `${t.label} (partial under policy)`, ui,
          expr: `(L1 × ${t.cMv} + L2 × ${t.cDv}) × 4096 × P6` }];
        const p1 = t.fMv ? `L1 × ${t.tM ? `(${t.tM})` : t.fMv}` : null;
        const p2 = t.fDv ? `L2 × ${t.tD ? `(${t.tD})` : t.fDv}` : null;
        const rate = [p1, p2].filter(Boolean).join(' + ');
        if (!rate) return [{ id: sid, depth: 2, unit: 'B', label: t.label, ui, value: 0 }];
        return [
          { id: sid, depth: 2, unit: 'B', label: t.label, ui, expr: `${rid} × (${rate}) × 4096 × P6` },
          { id: rid, depth: 3, label: 'kept?', ui, edit: mkEdit, value: t.r },
          ...(t.prec != null ? [{ id: t.bref, depth: 3, unit: 'B/e', label: 'precision (B/elem)', ui, edit: dtEdit(t.dtc), value: t.prec }] : []),
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
      ...(R ? [{ id: `B${i + 2}`, depth: 2, unit: 'B/e', label: 'precision (B/elem)', ui, edit: dtE, value: R.prec }] : []),
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
    { id: 'Z1', label: 'ZeRO level (1 optim · 2 +grads · 3 +weights)', value: zero, ui: { k: 'zero' }, edit: { t: 'seg', k: 'zero' } },
    // the level resolves to per-component SHARD GROUPS (1 = unsharded) via
    // indicator arithmetic — the byte formulas below never change shape
    // when Z1 moves, and neither do these
    { id: 'S1', depth: 1, ui: { k: 'zero' }, label: 'shard group · weights, experts', expr: '(Z1 ≥ 3) × (P5 - 1) + 1' },
    { id: 'S2', depth: 1, ui: { k: 'zero' }, label: 'shard group · weights, others', expr: '(Z1 ≥ 3) × (P4 - 1) + 1' },
    { id: 'S3', depth: 1, ui: { k: 'zero' }, label: 'shard group · gradients, experts', expr: '(Z1 ≥ 2) × (P5 - 1) + 1' },
    { id: 'S4', depth: 1, ui: { k: 'zero' }, label: 'shard group · gradients, others', expr: '(Z1 ≥ 2) × (P4 - 1) + 1' },
    { id: 'S5', depth: 1, ui: { k: 'zero' }, label: 'shard group · optimizer, experts', expr: '(Z1 ≥ 1) × (P5 - 1) + 1' },
    { id: 'S6', depth: 1, ui: { k: 'zero' }, label: 'shard group · optimizer, others', expr: '(Z1 ≥ 1) × (P4 - 1) + 1' },
    { id: 'F1', label: 'e4m3+ᵀ-resident params? (0/1)', value: fp8p ? 1 : 0, ui: { k: 'fp8params' }, edit: { t: 'cb', k: 'fp8params' } },
    { id: 'L1', label: 'MoE layers on this rank (slot split)', value: g.moe, ui: { k: 'rank' }, edit: { t: 'flip', k: 'rank' } },
    { id: 'L2', label: 'dense layers on this rank (slot split)', value: g.dense, ui: { k: 'rank' }, edit: { t: 'flip', k: 'rank' } },
    { id: 'L3', label: 'vocab matrices on this rank', expr: 'E1 + H1' },
    { id: 'E1', depth: 1, label: 'embedding on this rank? (0/1)', value: g.emb ? 1 : 0, ui: { k: 'rank' }, edit: { t: 'flip', k: 'rank' } },
    { id: 'H1', depth: 1, label: 'lm head on this rank? (0/1)', value: g.head ? 1 : 0, ui: { k: 'rank' }, edit: { t: 'flip', k: 'rank' } },
    { id: 'N1', label: 'params · routed experts, one MoE layer', unit: 'p', expr: '256 × 3 × 7168 × 2048' },
    { id: 'N2', label: 'params · rest of a MoE layer', unit: 'p', value: env.N.restLayer },
    { id: 'N3', label: 'params · one dense layer', unit: 'p', value: env.N.denseLayer },
    { id: 'N4', label: 'params · one vocab matrix', unit: 'p', expr: '129280 × 7168' },
    { id: 'Q1', label: 'expert params on this GPU', unit: 'p', expr: 'L1 × N1 / P3' },
    { id: 'Q2', label: 'non-expert block params on this GPU', unit: 'p', expr: 'L2 × N3 + L1 × N2' },
    { id: 'Q3', label: 'vocab params on this GPU (+ final norm)', unit: 'p',
      expr: env.simplify ? 'L3 × N4' : 'L3 × N4 + H1 × 7168' },   // the 7 K final norm is a simplify casualty
    { id: 'W1', label: 'weights (2 B bf16; F1 flips block params e4m3+ᵀ)', unit: 'B', expr: 'W2 + W3 + W4' },
    { id: 'W2', depth: 1, label: 'experts', unit: 'B', expr: cls(2, 'Q1', 'S1', true) },
    { id: 'W3', depth: 1, label: 'non-expert blocks', unit: 'B', expr: cls(2, 'Q2', 'S2', true) },
    { id: 'W4', depth: 1, label: 'emb + lm head (bf16 always)', unit: 'B', expr: cls(2, 'Q3', 'S2', false) },
    { id: 'G1', label: 'gradients (fp32, 4 B/param)', unit: 'B', expr: 'G2 + G3 + G4' },
    { id: 'G2', depth: 1, label: 'experts', unit: 'B', expr: cls(4, 'Q1', 'S3', false) },
    { id: 'G3', depth: 1, label: 'non-expert blocks', unit: 'B', expr: cls(4, 'Q2', 'S4', false) },
    { id: 'G4', depth: 1, label: 'emb + lm head', unit: 'B', expr: cls(4, 'Q3', 'S4', false) },
    { id: 'O1', label: 'optimizer states (8 B/param)', unit: 'B', expr: 'O2 + O3 + O4' },
    { id: 'O2', depth: 1, label: 'experts', unit: 'B', expr: cls(8, 'Q1', 'S5', false) },
    { id: 'O3', depth: 1, label: 'non-expert blocks', unit: 'B', expr: cls(8, 'Q2', 'S6', false) },
    { id: 'O4', depth: 1, label: 'emb + lm head', unit: 'B', expr: cls(8, 'Q3', 'S6', false) },
    { id: 'D1', label: 'stash/token · one MoE layer (the chips’ sum)', unit: 'B/tok', value: env.aM },
    { id: 'D2', label: 'stash/token · one dense layer', unit: 'B/tok', value: env.aD },
    { id: 'D3', label: 'stash/token · vocab side (x0 / logits + loss)', unit: 'B/tok',
      expr: 'E1 × 2 × 7168 + H1 × 6 × 129280' },
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
