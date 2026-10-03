// <dsv3-kernels pass=F|B|W mode=trace|coda>: one DeepSeek-V3 MoE layer at
// kernel-launch granularity, drawn as the block diagram draws the layer
// (01's grammar): a dataflow graph, not a timeline. Every launch of the
// DeepSeek EP64 profile is a node — GEMMs and FlashAttention as .box,
// memory-bound kernels as .op pills, runs of tiny glue collapsed into one
// italic .micro, DeepEP violet — and every tensor a launch writes for another
// to read is a wire (producer → consumer, arrowhead at the consumer, a dot
// where one tensor forks to several readers). Two columns (attention left,
// MoE right) of two lanes each; a lane stacks its launches in trace order,
// rows come from the longest dependency path, so launches with no dependency
// between them sit side by side. Wires that leave a column (the residual, the
// MoE input, d(x₁)) run down to a basement and up the destination's gutter,
// as 01's residual rail does. Stubs mark tensors that cross passes (amber:
// a stash the forward writes for the backward, or the backward for the weight
// gradients) or the layer boundary / weights (grey). Inside each node a two-bar
// meter on a log µs axis: traced (grey) over its speed of light (bronze when
// HBM binds, the precision color when the tensor cores do). The CODA flip
// keeps every anchor: folded launches ghost where they stand and say where
// they went, hosts grow an epilogue tag, meters move. Hover a node for the
// kernel, what it reads and writes, both meters and its CODA fate. State
// rides the hash (`k:<id>`).
import { priced, tally, graph, RATES } from './kerntrace.js';
import { tokensCss } from './viewer.js';
import { C } from './theme.js';
import { knobCss } from './ui.js';
import { attachTip } from './tip.js';

export const KERNELS_DEFAULTS = { pass: 'F', mode: 'trace' };
const PASSES = [['F', 'forward'], ['B', 'backward (input gradients)'], ['W', 'weight gradients']];
const MODES = [['trace', 'as traced'], ['coda', 'with CODA']];
const LANE_HEADS = { F: ['q', 'kv', 'router · DeepEP · routed experts', 'shared expert'], B: ['dq', 'dkv', 'DeepEP · routed experts', 'shared expert · router'], W: ['attention', '', 'MoE', ''] };

// ---- nodes: launches grouped into drawable nodes, per pass ---------------------
// consecutive glue launches of one op in one lane collapse into a single
// `.micro` node (the tooltip lists them); everything else is one node per launch.
// Edges are re-derived at node granularity (tensors internal to a group vanish).
export function nodesFor(pass) {
  const G = graph(pass);
  const nodes = [];
  const byLane = [[], [], [], []];
  for (const e of [...G.nodes].sort((a, b) => a.seq - b.seq)) {
    const lane = byLane[e.lane], last = lane.at(-1);
    if (e.kind === 'glue' && last?.kind === 'glue' && last.op === e.op) {
      last.es.push(e); last.us += e.us; last.raw += e.raw; last.b += e.b; last.sol += e.sol; last.hbm += e.hbm; last.hbmAfter += e.hbmAfter; last.solAfter += e.solAfter;
      last.gone = last.gone && e.gone; last.aux = last.aux || e.aux; last.key += '+' + e.i; last.lab = `${e.op}: ${last.es.length} small launches`;
      continue;
    }
    const n = { key: String(e.i), es: [e], lab: e.lab, kind: e.kind, op: e.op, lane: e.lane, seq: e.seq, us: e.us, raw: e.raw, b: e.b, sol: e.sol, hbm: e.hbm, flop: e.flop, bound: e.bound, dt: e.dt,
      host: e.kind === 'gemm' || e.kind === 'attn' || e.i === 175, gone: e.gone, aux: e.aux, hbmAfter: e.hbmAfter, solAfter: e.solAfter, epi: e.epi, how: e.how, comm: e.comm };
    lane.push(n); nodes.push(n);
  }
  const nodeOf = new Map(nodes.flatMap((n) => n.es.map((e) => [e, n])));
  const consumersOf = new Map();
  for (const e of G.edges) (consumersOf.get(e.t) ?? consumersOf.set(e.t, []).get(e.t)).push(nodeOf.get(e.c));
  for (const n of nodes) {
    const inside = new Set(n.es.flatMap((e) => e.out));
    n.in = [...new Set(n.es.flatMap((e) => e.in))].filter((t) => !inside.has(t));
    n.out = [...new Set(n.es.flatMap((e) => e.out))].filter((t) => !(consumersOf.get(t) ?? []).length || (consumersOf.get(t) ?? []).some((c) => c !== n));
    n.sources = n.es.flatMap((e) => e.sources); n.sinks = n.es.flatMap((e) => e.sinks);
  }
  const prod = new Map();
  for (const n of nodes) for (const t of n.out) prod.set(t, n);
  const edges = [];
  for (const c of nodes) for (const t of c.in) { const p = prod.get(t); if (p && p !== c) edges.push({ t, p, c }); }
  for (const n of nodes) { n.outs = n.out.filter((t) => edges.some((e) => e.p === n && e.t === t)); }
  byLane.forEach((lane) => lane.forEach((n, j) => { n.laneIdx = j; n.prev = lane[j - 1] ?? null; }));
  return { nodes, edges, byLane };
}

// ---- geometry ----------------------------------------------------------------
const NW = 212, GUT = 38, LANE = NW + GUT, W = 4 * LANE + 12, PH = 22, BH = 34, GAP = 12, TOP = 68, FOOT = 48, BASE = 10;
const laneX = (l) => l * LANE + GUT;
const SPX = 12;                                                   // the spine: a node's first output leaves, and its lane predecessor's wire arrives, 12px in from its left edge
const LOG0 = 1e-6, DOUBLINGS = 12, MW = NW - 12, PXD = MW / DOUBLINGS;   // the meter: 1 µs … 4,096 µs across the node
const mx = (s) => Math.max(0, Math.min(MW, Math.log2(Math.max(s, LOG0) / LOG0) * PXD));
const n = (x, d = 0) => x.toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d });
const fmtUs = (s) => s >= 1e-3 ? `${n(s * 1e3, 2)} ms` : s >= 1e-4 ? `${n(s * 1e6)} µs` : `${n(s * 1e6, s >= 1e-5 ? 1 : 2)} µs`;
const fmtB = (b) => b >= 1e9 ? `${n(b / 1e9, 2)} GB` : b >= 1e6 ? `${n(b / 1e6, 1)} MB` : b >= 1e3 ? `${n(b / 1e3, 1)} KB` : `${n(b)} B`;
const fmtF = (f) => f >= 1e12 ? `${n(f / 1e12, 2)} TFLOP` : f >= 1e9 ? `${n(f / 1e9, 1)} GFLOP` : `${n(f / 1e6)} MFLOP`;
const DT = { fp8: '#d6408b', bf16: '#52514e' };
const BRONZE = '#8c5a19', AMBER = '#eda100', GREY = '#898781';
const ell = (s, k) => s.length > k ? s.slice(0, Math.max(1, k - 1)).trimEnd() + '…' : s;
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
const fitEase = (p) => 1 - (1 - p) ** 3;
const lerp = (a, b, t) => a + (b - a) * t;
const f1 = (v) => (+v).toFixed(1);
const HOW = { epilogue: 'into the epilogue', aux: 'an auxiliary reduction over tile partials', fa: 'into FlashAttention\'s own kernels', a2a: 'into DeepEP\'s kernels', cache: 'a weight transpose: once per step, not per microbatch', stash: 'a stash the forward epilogue writes instead', keep: 'stays' };
const FATE = (r, P) => r.how === 'epilogue' ? (typeof r.es[0].host === 'number' ? `→ ${P[r.es[0].host].op} epilogue` : '→ the producing epilogue') : r.how === 'aux' ? '→ reduction over partials' : r.how === 'fa' ? '→ FlashAttention' : r.how === 'a2a' ? `→ DeepEP ${r.es[0].host ?? ''}` : r.how === 'cache' ? 'once per step' : r.how === 'stash' ? '→ a stash instead' : '';
const FROM = { layer: 'a layer input', weight: 'a weight', F: 'the forward', B: 'the backward' };
const TO = { layer: 'the layer boundary', wgrad: 'a weight gradient', B: 'the backward', W: 'the weight gradients' };

const CSS = `
dsv3-kernels { display: block; margin: 14px 0 26px; }
.kn { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-0b0b0b);
  border: 1px solid var(--c-e1e0d9); border-radius: 6px; background: var(--c-fcfcfb); padding: 8px 10px; position: relative;
  width: max-content; max-width: 100%; box-sizing: border-box; }
.kn .top { display: flex; align-items: stretch; gap: 10px; padding-bottom: 8px; flex-wrap: wrap; }
${knobCss('.kn .top')}
.kn svg { display: block; }
${tokensCss('.kn')}
.kn .op { stroke-width: 1; }
.kn .lab { font: 10.5px system-ui; fill: var(--c-0b0b0b); }
.kn text.bold { font-weight: 600; }
.kn .dims { font: 9px system-ui; fill: var(--c-898781); }
.kn .gl { font: italic 9.5px system-ui; fill: var(--c-898781); }
.kn .microlabel { font: italic 10px system-ui; fill: var(--c-52514e); }
.kn .colhead { font: 600 11px system-ui; fill: var(--c-52514e); }
.kn .tick { stroke: var(--c-e1e0d9); stroke-width: 1; }
.kn .epi { font: 9px system-ui; fill: var(--c-7a5200); }
.kn .fate { font: italic 9px system-ui; fill: var(--c-7a5200); }
.kn .epibox { fill: var(--c-fff3d1); stroke: var(--c-eda100); stroke-width: 0.8; }
.kn .stash { stroke: var(--c-eda100); }
.kn .foot { font: 11px system-ui; fill: var(--c-52514e); }
.kn .legend { font: 9.5px system-ui; fill: var(--c-52514e); }
.kn .ro { font-size: 11.5px; color: var(--c-52514e); margin-top: 6px; max-width: ${W}px; line-height: 1.45; min-height: 34px; }
.kn .ro b { color: var(--c-0b0b0b); font-weight: 600; }
.kn .dsv3-tip { max-width: 440px; }
.kn .dsv3-tip b { font-weight: 600; }
.kn .dsv3-tip .k { font: 10px ui-monospace, Menlo, monospace; color: var(--c-52514e); word-break: break-all; display: block; margin-top: 3px; }
.kn .dsv3-tip .io { color: var(--c-52514e); }
`;

class Dsv3Kernels extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    const st = this.id ? readState('k:' + this.id) : null;
    this.cfg = { ...KERNELS_DEFAULTS };
    for (const k of Object.keys(KERNELS_DEFAULTS)) { const v = st?.[k] ?? this.getAttribute(k); if (v != null) this.cfg[k] = v; }
    this.fixed = this.hasAttribute('fixed');   // no knobs: a pinned figure
    const style = document.createElement('style'); style.textContent = CSS;
    this._root = el('div', 'kn');
    this._top = el('div', 'top');
    this._chart = el('div');
    this._ro = el('div', 'ro');
    this._root.append(this._top, this._chart, this._ro);
    this.append(style, this._root);
    this._ui = {};
    if (!this.fixed) this._buildKnobs(); else this._top.remove();
    attachTip(this._chart, (ev) => this._tipFor(ev.target.closest?.('g[data-node]')), { parent: this._root });
    this._L = null;
    this.render();
    addEventListener('dsv3-theme', () => this.render());
  }
  _buildKnobs() {
    const grp = (label) => { const g = el('span', 'pargrp'); const l = el('div', 'parlab'); l.textContent = label; g.append(l); return g; };
    const row = (...kids) => { const r = el('div', 'parrow'); r.append(...kids); return r; };
    const seg = (key, opts) => {
      const eg = el('span', 'stp'); eg.dataset.knob = key;
      for (const [v, lab] of opts) { const x = document.createElement('button'); x.type = 'button'; x.textContent = lab; x.dataset.v = v; x.onclick = () => this._set(key, v); eg.append(x); }
      this._ui[key] = eg; return eg;
    };
    const g1 = grp('pass'); g1.append(row(seg('pass', PASSES)));
    const g2 = grp('kernels'); g2.append(row(seg('mode', MODES)));
    this._top.append(g1, g2);
    this._syncKnobs();
  }
  _syncKnobs() { for (const [k, eg] of Object.entries(this._ui)) for (const b of eg.querySelectorAll('button')) b.classList.toggle('on', b.dataset.v === this.cfg[k]); }
  _set(k, v) {
    if (this.cfg[k] === v) return;
    this.cfg[k] = v; this._syncKnobs();
    if (this.id) writeState('k:' + this.id, this.cfg);
    this.dispatchEvent(new CustomEvent('dsv3-kernels', { detail: { ...this.cfg } }));
    if (k === 'pass') this.render(); else this._animateTo(this._layout());
  }
  // pixel-space layout: nodes keyed by launch ids, wires keyed by producer>consumer; the mode decides opacities and meters
  _layout() {
    const coda = this.cfg.mode === 'coda', pass = this.cfg.pass;
    const P = priced();
    const { nodes, edges, byLane } = (this._graph ??= {})[pass] ??= nodesFor(pass);
    // rows: per column, the longest path through in-column wires and lane order
    const colOf = (l) => l < 2 ? 0 : 1;
    const row = new Map();
    for (const nd of [...nodes].sort((a, b) => a.seq - b.seq)) {
      let r = nd.prev ? row.get(nd.prev) + 1 : 0;
      for (const e of edges) if (e.c === nd && colOf(e.p.lane) === colOf(nd.lane)) r = Math.max(r, row.get(e.p) + 1);
      row.set(nd, r);
    }
    const rowH = [[], []], rowY = [[], []];
    for (const nd of nodes) { const c = colOf(nd.lane), r = row.get(nd); rowH[c][r] = Math.max(rowH[c][r] ?? 0, nd.host && !nd.comm ? BH : PH); }
    const colH = [0, 0];
    for (const c of [0, 1]) { let y = TOP; for (let r = 0; r < rowH[c].length; r++) { rowY[c][r] = y; y += (rowH[c][r] ?? PH) + GAP; } colH[c] = y - GAP; }
    const baseY = Math.max(colH[0], colH[1]) + BASE;
    const L = { coda, pass, nodes: [], edges: [], dots: [], stubs: [], h: baseY, T: tally()[pass], colH, lanes: byLane.map((l) => l.length) };
    const pos = new Map();
    for (const nd of nodes) {
      const c = colOf(nd.lane), r = row.get(nd), h = nd.host && !nd.comm ? BH : PH;
      const ghost = coda && nd.gone;
      const sol = coda && nd.host ? nd.solAfter : nd.sol, hbm = coda ? nd.hbmAfter : nd.hbm;
      const aux = ghost && nd.aux;
      const q = { ...nd, x: laneX(nd.lane), y: rowY[c][r], h, row: r, col: c, ghost, a: ghost ? (aux ? 0.6 : 0.3) : 1,
        tx: nd.us ? mx(nd.us * 1e-6) : 0, sx: sol > 0 && (!ghost || aux) ? mx(sol) : 0, hx: hbm > 0 && !ghost ? mx(hbm) : 0, showEpi: coda && nd.host && nd.epi?.length, fate: ghost ? FATE(nd, P) : '' };
      pos.set(nd, q); L.nodes.push(q);
    }
    // wires. Exits: a node's k-th consumed output leaves its bottom edge at SPX + 14k. Entries: the lane
    // predecessor's wire arrives at the top (once); every other wire arrives at the left edge, from a rail
    // in the consumer's gutter; farther rails enter lower so none crosses another's last leg.
    const chans = new Map();    // `${col}:${row}` → [{x1, x2, k}]  horizontal runs in the gap below a row
    const rails = [[], [], [], []];   // per gutter: [{y1, y2, slot}]
    const bases = [];           // basement runs: [{x1, x2, k}]
    const chan = (col, r, x1, x2) => { const key = `${col}:${r}`, used = chans.get(key) ?? []; chans.set(key, used); const lo = Math.min(x1, x2) - 2, hi = Math.max(x1, x2) + 2; let k = 0; while (used.some((u) => u.k === k && u.x1 < hi && u.x2 > lo)) k++; used.push({ x1: lo, x2: hi, k }); return rowY[col][r] + rowH[col][r] + 2 + 3 * Math.min(k, 3); };
    const rail = (g, y1, y2) => { const lo = Math.min(y1, y2) - 3, hi = Math.max(y1, y2) + 3; let s = 0; while (rails[g].some((u) => u.slot === s && u.y1 < hi && u.y2 > lo)) s++; rails[g].push({ y1: lo, y2: hi, slot: s }); return laneX(g) - 6 - 4 * s; };
    const base = (x1, x2) => { const lo = Math.min(x1, x2) - 2, hi = Math.max(x1, x2) + 2; let k = 0; while (bases.some((u) => u.k === k && u.x1 < hi && u.x2 > lo)) k++; bases.push({ x1: lo, x2: hi, k }); return baseY + 3 * k; };
    const topTaken = new Set(), side = new Map();   // consumer → [{e, railX}]
    const E = [...edges].sort((a, b) => a.p.seq - b.p.seq || a.p.outs.indexOf(a.t) - b.p.outs.indexOf(b.t));
    const plan = [];
    for (const e of E) {
      const p = pos.get(e.p), c = pos.get(e.c);
      const ex = p.x + SPX + 14 * p.outs.indexOf(e.t), ey = p.y + p.h;
      const w = { e, p, c, ex, ey, turns: [] };
      if (c.col === p.col && e.c.prev === e.p && !topTaken.has(c)) {
        topTaken.add(c); w.top = true; w.enX = c.x + SPX;
        if (ex !== w.enX) { w.chY = chan(p.col, p.row, ex, w.enX); w.turns.push(w.chY); }
        w.endY = c.y;
      } else if (c.col === p.col) {
        w.chY = chan(p.col, p.row, ex, laneX(c.lane) - 6); w.turns.push(w.chY);
        w.railX = rail(c.lane, w.chY, c.y + c.h / 2);
        (side.get(c) ?? side.set(c, []).get(c)).push(w);
      } else {   // across the columns: down this column's gutter to the basement, along it, up the destination's gutter
        w.chY = chan(p.col, p.row, ex, laneX(p.lane) - 6); w.turns.push(w.chY);
        w.rail1X = rail(p.lane, w.chY, baseY + 12);
        w.baseY = base(w.rail1X, laneX(c.lane) - 6);
        w.railX = rail(c.lane, w.baseY, c.y + c.h / 2);
        (side.get(c) ?? side.set(c, []).get(c)).push(w);
      }
      plan.push(w);
    }
    // side entries, sources (stubs in) and sinks (stubs out) share each node's left/right edge
    const offsets = (h, k) => { const span = Math.min(h - 8, 4 * (k - 1)); return Array.from({ length: k }, (_, i) => k === 1 ? 0 : -span / 2 + span * i / (k - 1)); };
    for (const q of L.nodes) {
      const ws = (side.get(q) ?? []).sort((a, b) => b.railX - a.railX);   // nearest rail first (enters highest)
      const srcs = q.sources;
      const offs = offsets(q.h, ws.length + srcs.length);
      ws.forEach((w, i) => { w.endY = q.y + q.h / 2 + offs[srcs.length + i]; });
      srcs.forEach((s, i) => L.stubs.push({ key: `${q.key}<${s.t}`, dir: 'in', x: q.x, y: q.y + q.h / 2 + offs[i], stash: s.from === 'F' || s.from === 'B', a: q.a, t: s.t, node: q }));
      const so = offsets(q.h, q.sinks.length);
      q.sinks.forEach((s, i) => L.stubs.push({ key: `${q.key}>${s.t}`, dir: 'out', x: q.x + NW, y: q.y + q.h / 2 + so[i], stash: s.to === 'B' || s.to === 'W', a: q.a, t: s.t, node: q }));
    }
    for (const w of plan) {
      const { e, p, c } = w;
      let d = `M ${f1(w.ex)} ${f1(w.ey)}`;
      if (w.top) d += w.chY != null ? ` V ${f1(w.chY)} H ${f1(w.enX)} V ${f1(w.endY)}` : ` V ${f1(w.endY)}`;
      else if (w.rail1X == null) d += ` V ${f1(w.chY)} H ${f1(w.railX)} V ${f1(w.endY)} H ${f1(c.x)}`;
      else d += ` V ${f1(w.chY)} H ${f1(w.rail1X)} V ${f1(w.baseY)} H ${f1(w.railX)} V ${f1(w.endY)} H ${f1(c.x)}`;
      L.edges.push({ key: `${p.key}>${c.key}`, t: e.t, d, a: Math.min(p.a, c.a), ex: w.ex, top: !!w.top, turnY: w.top && w.chY == null ? w.endY : w.chY, p, c });
    }
    // forks: wires sharing an exit split at distinct turning heights; a dot marks every split a wire continues past
    const byExit = new Map();
    for (const w of L.edges) { const k = `${w.p.key}@${w.ex}`; (byExit.get(k) ?? byExit.set(k, []).get(k)).push(w); }
    for (const ws of byExit.values()) {
      if (ws.length < 2) continue;
      const deepest = Math.max(...ws.map((w) => w.turnY));
      for (const w of ws) if (w.turnY < deepest) L.dots.push({ key: `${w.key}:${w.t}`, x: w.ex, y: w.turnY, a: w.a });
    }
    return L;
  }
  _animateTo(B) {
    const A = this._L; const N = 12; let f = 0;
    const gen = this._gen = (this._gen ?? 0) + 1;
    const step = () => {
      if (this._gen !== gen) return;
      f++; const t = fitEase(Math.min(1, f / N));
      this._draw(blend(A, B, t));
      if (f < N) setTimeout(step, 16); else { this._L = B; this._draw(B); }
    };
    setTimeout(step, 16);
  }
  render() { this._graph = null; this._L = this._layout(); this._draw(this._L); }
  _draw(L) {
    const B = [];
    B.push(`<defs><marker id="kn-arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 8 4 L 0 8 z" fill="${C(GREY)}"/></marker>` +
      `<marker id="kn-arrs" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 8 4 L 0 8 z" fill="${C(AMBER)}"/></marker></defs>`);
    // legend: the grammar, then the meter's axis at the nodes' own scale
    B.push(`<text class="legend" x="${laneX(0)}" y="12">one node per launch: <tspan font-weight="600">GEMM</tspan> boxes · memory-bound pills · <tspan font-style="italic">collapsed glue</tspan> · DeepEP violet · a wire per tensor (dot = fork) · stubs: <tspan fill="${C(AMBER)}">▸ from / to another pass</tspan> · ▸ the layer or a weight</text>`);
    const lx = laneX(2) + 20;
    B.push(`<text class="legend" x="${laneX(0)}" y="27">in each node, traced <tspan fill="${C(GREY)}">▬</tspan> over its floor: <tspan fill="${C(BRONZE)}">▬</tspan> HBM · <tspan fill="${C(DT.fp8)}">▬</tspan> tensor cores fp8 · <tspan fill="${C(DT.bf16)}">▬</tspan> bf16, on a log axis:</text>`);
    for (const us of [1, 10, 100, 1000]) { const x = lx + mx(us * 1e-6); B.push(`<line class="tick" x1="${f1(x)}" y1="29" x2="${f1(x)}" y2="34"/><text class="dims" x="${f1(x)}" y="27" text-anchor="middle">${us >= 1000 ? '1 ms' : us + ' µs'}</text>`); }
    B.push(`<line class="tick" x1="${lx}" y1="34" x2="${f1(lx + MW)}" y2="34"/>`);
    for (const [c, head] of [[0, 'attention sub-layer'], [1, 'MoE sub-layer']]) {
      if (!L.lanes[2 * c] && !L.lanes[2 * c + 1]) continue;
      B.push(`<text class="colhead" x="${laneX(2 * c)}" y="${TOP - 22}">${head}</text>`);
      for (const l of [2 * c, 2 * c + 1]) if (L.lanes[l] && LANE_HEADS[L.pass][l]) B.push(`<text class="gl" x="${laneX(l)}" y="${TOP - 9}">${esc(LANE_HEADS[L.pass][l])}</text>`);
    }
    // wires first (nodes paint over the spine), then dots, stubs, nodes
    for (const w of L.edges) B.push(`<path class="wire" data-edge="${esc(w.key)}" data-t="${esc(w.t)}" d="${w.d}" marker-end="url(#kn-arr)" opacity="${w.a.toFixed(3)}"/>`);
    for (const d of L.dots) B.push(`<circle data-dot="${esc(d.key)}" cx="${f1(d.x)}" cy="${f1(d.y)}" r="2.5" fill="${C(GREY)}" opacity="${d.a.toFixed(3)}"/>`);
    for (const s of L.stubs) B.push(`<path class="wire${s.stash ? ' stash' : ''}" data-stub="${s.dir}" data-t="${esc(s.t)}" d="${s.dir === 'in' ? `M ${s.x - 9} ${f1(s.y)} H ${s.x}` : `M ${s.x} ${f1(s.y)} H ${s.x + 9}`}" marker-end="url(#kn-arr${s.stash ? 's' : ''})" opacity="${s.a.toFixed(3)}"/>`);
    for (const r of L.nodes) {
      const g = [];
      const { x, y, h } = r;
      const cls = r.comm ? 'comm' : r.host && !r.comm ? 'box' : r.kind === 'glue' ? 'micro' : 'op';
      const rx = cls === 'op' || cls === 'micro' ? 6 : 2;
      g.push(`<rect class="${cls}" x="${x}" y="${f1(y)}" width="${NW}" height="${h}" rx="${rx}"${r.ghost ? ' stroke-dasharray="3 2"' : ''}${cls === 'micro' ? ` fill="${C('#f7f6f1')}" stroke="${C('#d8d6cb')}"` : ''}/>`);
      const e0 = r.es[0];
      const tcls = cls === 'micro' ? 'microlabel' : cls === 'box' ? 'lab bold' : 'lab';
      const right = r.ghost ? r.fate : r.out.length ? `→ ${r.out.join(', ')}` : '';
      const rightN = Math.min(right.length, 26);
      if (cls === 'box') {
        g.push(`<text class="${tcls}" x="${x + 5}" y="${y + 12}">${esc(ell(r.lab, 34))}</text>`);
        if (r.showEpi) {
          const txt = ell(r.epi.join(' · '), Math.floor((NW - 16) / 4.7));
          g.push(`<rect class="epibox" x="${x + 4}" y="${y + 15}" width="${NW - 8}" height="10" rx="3"/>`);
          g.push(`<text class="epi" x="${x + 7}" y="${y + 22.5}">${esc(txt)}</text>`);
        } else {
          const dims = e0.kind === 'gemm' ? `${n(e0.M)}×${n(e0.Kd)}→${n(e0.N)}${e0.groups > 1 ? ` ×${e0.groups}` : ''}` : e0.f ? fmtF(e0.f) : fmtB(e0.b);
          g.push(`<text class="dims" x="${x + 5}" y="${y + 23}">${esc(dims)}</text>`);
          if (right) g.push(`<text class="dims" x="${x + NW - 5}" y="${y + 23}" text-anchor="end">${esc(ell(right, 24))}</text>`);
        }
      } else {
        const labN = Math.max(8, Math.floor((NW - 12 - rightN * 4.4) / 5.6));
        g.push(`<text class="${tcls}" x="${x + 5}" y="${y + 11}">${esc(ell(r.lab, labN))}</text>`);
        if (right) g.push(`<text class="${r.ghost ? 'fate' : 'dims'}" x="${x + NW - 5}" y="${y + 11}" text-anchor="end">${esc(ell(right, 26))}</text>`);
      }
      // the meter: traced over its floor, in the node's bottom band
      const my = y + h - 8, mx0 = x + 6;
      if (!r.ghost && r.tx > 0) g.push(`<rect data-bar="trace" x="${mx0}" y="${f1(my)}" width="${f1(Math.max(0.6, r.tx))}" height="2.6" fill="${C(GREY)}"/>`);
      if ((!r.ghost || (r.aux && r.sx > 0.5)) && !r.comm) {
        const fill = r.host ? (r.sol === r.flop ? C(DT[r.dt] ?? '#52514e') : C(BRONZE)) : C(BRONZE);
        g.push(`<rect data-bar="sol" x="${mx0}" y="${f1(my + 3.4)}" width="${f1(Math.max(0.6, r.sx))}" height="2.6" fill="${fill}"/>`);
        if (r.host && r.flop) g.push(`<rect data-bar="hbm" x="${f1(mx0 + r.hx - 1)}" y="${f1(my + 2.4)}" width="2" height="4.6" fill="${C(BRONZE)}"/>`);
      }
      B.push(`<g data-node="${r.key}" data-kind="${r.kind}" data-lane="${r.lane}" data-row="${r.row}" data-gone="${r.ghost ? 1 : 0}" opacity="${r.a.toFixed(3)}">${g.join('')}</g>`);
    }
    // the footer: the pass's sums
    const T = L.T, fy = L.h + 18;
    const nonGemm = (k) => k === 'trace' ? T.other.trace : k === 'hbm' ? T.other.hbm * 1e6 : T.other.hbmAfter * 1e6;
    const b = (t) => `<tspan font-weight="600" fill="${C('#0b0b0b')}">${t}</tspan>`;
    B.push(`<text class="foot" x="${laneX(0)}" y="${fy}" data-foot="trace">traced, this pass: ${b(n(T.gemm.trace / 1e3, 2) + ' ms')} in GEMMs and attention, ${b(n(T.other.trace / 1e3, 2) + ' ms')} in the other ${T.other.kept + T.other.gone + T.other.aux} launches</text>`);
    B.push(`<text class="foot" x="${laneX(0)}" y="${fy + 15}" data-foot="floor">HBM floor of those ${T.other.kept + T.other.gone + T.other.aux}: ${b(n(nonGemm('hbm') / 1e3, 2) + ' ms')}${L.coda ? ` → with CODA ${b(n(nonGemm('after') / 1e3, 2) + ' ms')} in ${T.other.kept + T.other.aux} launches; hosts: ${n(T.gemm.hbmAfter * 1e3, 1)} ms of HBM under ${n(T.gemm.flop * 1e3, 1)} ms on the tensor cores${T.gemm.over.length ? `, ${T.gemm.over.length} tips HBM-bound (+${n(T.gemm.exposed * 1e6)} µs)` : ''}` : ` — traced ×${n(T.other.eff, 1)} above it`}</text>`);
    const H = L.h + FOOT;
    this._chart.innerHTML = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">${B.join('')}</svg>`;
    this._readout(L);
  }
  _readout(L) {
    const T = L.T, p = PASSES.find(([k]) => k === L.pass)[1];
    const g = T.gemm, o = T.other;
    this._ro.innerHTML = `<b>${p}</b>, one MoE layer, one 4,096-token microbatch, one GPU of 64: ${T.launches} kernel launches, ${L.edges.length} tensors between them. ` +
      `GEMMs and attention run ×${n(g.trace * 1e-6 / g.sol, 2)} their speed of light (${fmtUs(g.sol)} at 03's measured rates); the other ${o.kept + o.gone + o.aux} launches take ${fmtUs(o.trace * 1e-6)} against an HBM floor of ${fmtUs(o.hbm)}, ×${n(o.eff, 1)}. ` +
      (L.coda ? `Under CODA ${o.gone} of them fold away, ${o.aux} shrink to reductions over tile partials, and the floor of what remains is ${fmtUs(o.hbmAfter)}; at today's ×${n(o.eff, 1)} that is ${fmtUs(o.projected)} of kernel time instead of ${fmtUs(o.trace * 1e-6)}.` : 'Flip to CODA to see which fold into GEMM epilogues — every fold follows a wire.');
  }
  _tipFor(g) {
    if (!g) return null;
    const L = this._L, key = g.dataset.node;
    const r = L.nodes.find((q) => q.key === key);
    if (!r) return null;
    const P = priced();
    const d = document.createElement('div');
    const es = r.es;
    const lines = [];
    lines.push(`<b>${esc(r.lab)}</b>${es.length === 1 && es[0].inf ? ' <i>(operand inferred from the launch shape)</i>' : ''}`);
    if (es.length === 1) {
      const e = es[0];
      lines.push(`${e.ann} · stream ${e.s} · grid ${e.grid.join('×')} · block ${e.block.join('×')}`);
      lines.push(`traced ${fmtUs(e.raw * 1e-6)}${e.rows ? ` (normalized to 32,768 routed rows: ${fmtUs(e.us * 1e-6)})` : ''}, mean of 4 cycles`);
      if (e.kind === 'gemm') lines.push(`${n(e.M)} × ${n(e.Kd)} → ${n(e.N)}${e.groups > 1 ? ` (${e.groups} experts)` : ''}: ${fmtF(e.f)} ÷ ${n(RATES[e.dt] / 1e12)} TFLOP/s = ${fmtUs(e.flop)}; ${fmtB(e.b)} ÷ 3.35 TB/s = ${fmtUs(e.hbm)} → <b>${e.bound}-bound</b>, traced ×${n(e.x, 2)}`);
      else if (e.kind === 'attn') lines.push(`${fmtF(e.f)} ÷ ${n(RATES[e.dt] / 1e12)} TFLOP/s = ${fmtUs(e.flop)}; ${fmtB(e.b)} ÷ 3.35 TB/s = ${fmtUs(e.hbm)} → traced ×${n(e.x, 2)}`);
      else if (!e.comm) lines.push(`${fmtB(e.b)} through HBM ÷ 3.35 TB/s = <b>${fmtUs(e.hbm)}</b>${e.hbm > 0 ? ` → traced ×${n(e.x, 1)}` : ''}`);
      else lines.push('communication (priced on the network in 03, not on HBM)');
    } else {
      lines.push(`${es.length} launches, ${fmtUs(r.raw * 1e-6)} traced in all; HBM floor ${fmtUs(r.hbm)} — latency-bound, not bandwidth-bound`);
      for (const e of es) lines.push(`· ${esc(e.lab.replace(/^[^:]*: /, ''))} — ${fmtUs(e.raw * 1e-6)}, grid ${e.grid[0]}`);
    }
    // what it reads and writes, and from / for whom
    const G = this._graph[L.pass];
    const who = (q) => `#${q.key.split('+')[0]}${q.es.length > 1 ? '…' : ''} ${esc(q.lab)}`;
    const ins = r.in.map((t) => { const e = G.edges.find((w) => w.c.key === r.key && w.t === t); const s = r.sources.find((s) => s.t === t); return `${esc(t)} <span class="io">(${e ? `from ${who(e.p)}` : s ? `from ${FROM[s.from]}${s.why ? `: ${esc(s.why)}` : ''}` : '?'})</span>`; });
    const outs = r.out.map((t) => { const cs = G.edges.filter((w) => w.p.key === r.key && w.t === t).map((w) => who(w.c)); const s = r.sinks.find((s) => s.t === t); return `${esc(t)} <span class="io">(${cs.length ? `→ ${cs.join(', ')}` : s ? `→ ${TO[s.to]}${s.why ? `: ${esc(s.why)}` : ''}` : '?'})</span>`; });
    if (ins.length) lines.push(`reads: ${ins.join(' · ')}`);
    if (outs.length) lines.push(`writes: ${outs.join(' · ')}`);
    if (es.length === 1) {
      const e = es[0];
      if (e.kind === 'gemm' || e.kind === 'attn' || e.i === 175) { if (e.epi?.length) lines.push(`CODA epilogue: ${e.epi.join(' · ')}${e.add ? ` (+${fmtB(e.add)} of epilogue traffic)` : ''}${e.out8 ? ` (output ${fmtB(Math.abs(e.out8))} ${e.out8 < 0 ? 'smaller' : 'larger'})` : ''} → HBM ${fmtUs(e.hbmAfter)}, ${e.hbmAfter <= e.flop ? 'still under' : 'OVER'} ${fmtUs(e.flop)}`); }
      else if (!e.comm) lines.push(`CODA: ${HOW[e.how]}${typeof e.host === 'number' ? ` of <i>${esc(P[e.host].lab)}</i>` : e.host ? ` (${e.host})` : ''}${e.after > 0 ? ` — ${fmtB(e.after)} remains` : ''}`);
      if (e.note) lines.push(`<i>${esc(e.note)}</i>`);
      lines.push(`<span class="k">${esc(e.name.length > 200 ? e.name.slice(0, 200) + '…' : e.name)}</span>`);
    } else lines.push(`CODA: ${es.every((e) => e.gone) ? 'fold away' : es.some((e) => e.gone) ? 'some fold into the router GEMM\'s epilogue; the rest stay' : 'stay (not GEMM-adjacent)'}`);
    d.innerHTML = lines.join('<br>');
    return d;
  }
}

function blend(A, B, t) {
  if (!A || t >= 1 || A.pass !== B.pass) return B;
  const mix = (as, bs) => bs.map((b, i) => { const a = as[i]; return a ? { ...b, a: lerp(a.a, b.a, t), sx: b.sx != null ? lerp(a.sx, b.sx, t) : b.sx, hx: b.hx != null ? lerp(a.hx, b.hx, t) : b.hx, ghost: b.ghost && t > 0.5 ? true : a.ghost && t <= 0.5, showEpi: t > 0.5 ? b.showEpi : a.showEpi, fate: t > 0.5 ? b.fate : a.fate } : b; });
  return { ...B, nodes: mix(A.nodes, B.nodes), edges: mix(A.edges, B.edges), dots: mix(A.dots, B.dots), stubs: mix(A.stubs, B.stubs), coda: t > 0.5 ? B.coda : A.coda };
}
function el(tag, cls) { const e = document.createElement(tag); if (cls) e.className = cls; return e; }
function readState(key) {
  try { const v = new URLSearchParams(location.hash.slice(1)).get(key); return v ? JSON.parse(v) : null; } catch { return null; }
}
function writeState(key, obj) {
  const p = new URLSearchParams(location.hash.slice(1));
  p.set(key, JSON.stringify(obj));
  history.replaceState(null, '', '#' + p.toString());
}
if (typeof customElements !== 'undefined' && !customElements.get('dsv3-kernels')) customElements.define('dsv3-kernels', Dsv3Kernels);
