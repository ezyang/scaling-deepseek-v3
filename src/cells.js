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

// ---- the mini formula language: numbers, cell ids, + - × / ( ) ------------
const AST = new Map();   // expr string → parsed tree (exprs are static per state)
function parse(src) {
  if (AST.has(src)) return AST.get(src);
  let i = 0;
  const ws = () => { while (src[i] === ' ') i++; };
  const expr = () => {
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
    const c = /^([A-Z]\d+)/.exec(src.slice(i));
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
      : n.op === '+' ? go(n.a) + go(n.b)
        : n.op === '-' ? go(n.a) - go(n.b)
          : n.op === '×' ? go(n.a) * go(n.b) : go(n.a) / go(n.b);
  return go(parse(src));
}
export const refsOf = (src) => [...new Set(src.match(/[A-Z]\d+/g) ?? [])];

// ---- the cell definitions, built for one widget state ----------------------
// env: { world, pp, ep, zero, sched, fp8p, g: {moe, dense, emb, head},
//        aM, aD (stash B/token per layer kind),
//        N: { restLayer, denseLayer } (parameter-count leaves, from PARAMS) }
// Units: 'B' bytes · 'p' parameter counts · 'B/tok' · absent = plain counts.
export function buildCells(env) {
  const { g, zero, fp8p } = env;
  const V = (g.emb ? 1 : 0) + (g.head ? 1 : 0);
  // sharded vs replicated spellings (ZeRO level picks per component)
  const sh = (bpp) => `${bpp} × (Q1 / P5 + (Q2 + Q3) / P4)`;
  const rep = (bpp) => `${bpp} × (Q1 + Q2 + Q3)`;
  const defs = [
    { id: 'P1', label: 'GPUs in the cluster', value: env.world },
    { id: 'P2', label: 'pipeline stages (PP)', value: env.pp },
    { id: 'P3', label: 'expert parallelism (EP)', value: env.ep },
    { id: 'P4', label: 'data parallelism (replicas of a stage)', expr: 'P1 / P2' },
    { id: 'P5', label: 'expert data parallelism (replicas of an expert shard)', expr: 'P4 / P3' },
    { id: 'P6', label: 'microbatches in flight on this rank' + (env.sched !== 'one' && env.pp > 1 ? ' (DualPipeV: PP + ½)' : ''),
      expr: env.sched === 'one' || env.pp === 1 ? '1' : 'P2 + 0.5' },
    { id: 'L1', label: 'MoE layers on this rank (slot split)', value: g.moe },
    { id: 'L2', label: 'dense layers on this rank (slot split)', value: g.dense },
    { id: 'L3', label: 'vocab matrices on this rank (embedding / lm head)', value: V },
    { id: 'N1', label: 'params: routed experts of one MoE layer', unit: 'p', expr: '256 × 3 × 7168 × 2048' },
    { id: 'N2', label: 'params: rest of a MoE layer (MLA + shared expert + router + norms)', unit: 'p', value: env.N.restLayer },
    { id: 'N3', label: 'params: one dense layer (MLA + dense FFN + norms)', unit: 'p', value: env.N.denseLayer },
    { id: 'N4', label: 'params: one vocab matrix', unit: 'p', expr: '129280 × 7168' },
    { id: 'Q1', label: 'expert params on this GPU', unit: 'p', expr: 'L1 × N1 / P3' },
    { id: 'Q2', label: 'non-expert block params on this GPU', unit: 'p', expr: 'L2 × N3 + L1 × N2' },
    { id: 'Q3', label: 'vocab params on this GPU' + (g.head ? ' (+ the final norm)' : ''), unit: 'p',
      expr: V ? `L3 × N4${g.head ? ' + 7168' : ''}` : null, value: V ? undefined : 0 },
    // fp8-resident: BOTH e4m3 orientations, each 1 B/elem + one fp32 scale
    // per 1×128 tile → 2 × (1 + 4/128) = 2.0625 B on the block params
    { id: 'W1', label: fp8p ? 'weights (e4m3+ᵀ resident, fp32 scale per 1×128 tile; vocab stays bf16)' : 'weights (bf16, 2 B/param)', unit: 'B',
      expr: fp8p
        ? (zero >= 3 ? '2 × (1 + 4/128) × (Q1 / P5 + Q2 / P4) + 2 × Q3 / P4' : '2 × (1 + 4/128) × (Q1 + Q2) + 2 × Q3')
        : (zero >= 3 ? sh(2) : rep(2)) },
    { id: 'G1', label: 'gradients (fp32, 4 B/param)', unit: 'B', expr: zero >= 2 ? sh(4) : rep(4) },
    { id: 'O1', label: 'optimizer states (fp32 master + moments, 8 B/param)', unit: 'B', expr: zero >= 1 ? sh(8) : rep(8) },
    { id: 'D1', label: 'stash per token, one MoE layer (sum of the diagram’s chips)', unit: 'B/tok', value: env.aM },
    { id: 'D2', label: 'stash per token, one dense layer (same policy + recipe)', unit: 'B/tok', value: env.aD },
    { id: 'D3', label: 'vocab-side activations per token (embed x0 / logits + fp32 loss)', unit: 'B/tok',
      expr: V ? [g.emb && '2 × 7168', g.head && '6 × 129280'].filter(Boolean).join(' + ') : null,
      value: V ? undefined : 0 },
    { id: 'A1', label: 'saved activations (4096-token microbatches)', unit: 'B',
      expr: '(L1 × D1 + L2 × D2) × 4096 × P6 + D3 × 4096' },
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
