// Canvas trace viewer, perfetto-flavored: WASD navigation, wheel zoom/pan,
// hover tooltips, click select, M marks, F focuses. Embeddable many times per
// page via the <dsv3-trace> custom element or the TraceViewer class.

import { fmtUs, fmtNum, DSV3, HARDWARE } from './model.js';
import { simulate, LEVELS } from './sim.js';
import { resolveMatmuls, MATMULS, RECIPES, RECIPE_T } from './recipes.js';
import { blockGraph, analyze, RECOMPUTE_PRESETS, MARKABLE, TP_REPLICATED, markKey } from './blockgraph.js';
import { PARAMS } from './params.js';
import { buildCells, evalExpr, cellsEnv } from './cells.js';
import { C, initTheme } from './theme.js';
import { Dsv3Sheet } from './sheet.js';
// theme boot + flip: C()-colors live in rendered attributes, so a theme
// change re-renders every widget (their followers re-sync off the events
// those renders already fire)
if (typeof document !== 'undefined') {
  initTheme();
  addEventListener('dsv3-theme', () => {
    for (const el of document.querySelectorAll(
      'dsv3-layer, dsv3-anatomy-plan, dsv3-param-tally, dsv3-pp-schedule, dsv3-pp-fold, dsv3-beat-deck, dsv3-sheet'))
      (el.render ?? el.draw ?? el.build)?.call(el);
  });
}
// per-page capacity/story helpers (the fit chart's red line and its
// "a span is a factor" ruler are page facts, not model facts)
const HW_SHORT = { h800: 'H800', h100: 'H100', gb200: 'GB200', gb300: 'GB300' };
// 02's distances ruler: the factors its story applies (PP8 · EP64 · 2048 GPUs)
const FACS_02 = [[1, ''], [2, '×2'], [8, '×8 (PP)'], [64, '×64 (EP)'], [2048, '×2048 (GPUs)']];
// facs="2,8:PP,64:EP,2048:GPUs" → the ruler's [factor, label] ticks (1 = the origin tick)
const facsOf = (s) => !s ? FACS_02 : [[1, ''], ...s.split(',').map((t) => {
  const [f, lab] = t.split(':'); return [+f, lab ? `×${f} (${lab})` : `×${f}`];
})];
// a stash chip's multiplicity tag: the Hopper fp8ᵀ dual (both orientations)
// or extra copies an implementation keeps (Megatron's two down-proj linears
// each quantize the norm output) — one chip, doubled bytes, a tag
const stashTag = (A, ids) => ids.some((i) => A.dual.has(i)) ? ' ᵀ×2'
  : ids.some((i) => (A.copies?.[i] ?? 1) > 1) ? ` ×${Math.max(...ids.map((i) => A.copies?.[i] ?? 1))}` : '';
// a byte component's legend label; the gradient buffer's bytes/param follow the config
const compLabel = (c, gradB = 4) => c.prop === 'showGrads' && gradB !== 4 ? `gradients (bf16, ${gradB} B/param)` : c.label;
// in-flight counts print exactly when dyadic (8.5, 2.25) and to 2 decimals otherwise (the chunk-weighted means)
const fmtIF = (v) => Number.isInteger(v * 4) ? String(v) : v.toFixed(2);
// the schedule's display name for labels/readouts
const schedName = (l) => l.sched === 'one' ? '×1mb' : l.sched === 'interleaved'
  ? (l.pp > 1 ? `1F1B·VP${l.vpp}` : '1F1B') : l.pp > 1 ? 'DualPipeV' : '1F1B';
import { schedGeom } from './localmodel.js';
import { BYTE_COMPS, ACT_BUCKETS, actBucketsOf, PP_CHOICES, LOCAL_PAR, CFG_DEFAULTS,
  vstagesOf, ppStage, actLayerBytes, inflightOf, peakStage,
  mmSig, markSig, HAZIZA_CFG } from './localmodel.js';

// spreadsheet-style highlighting, shared by the layer and the anatomy plan:
// mark the [data-op] groups in `hl` and fade the rest (the tabs' visual
// language); hl = null clears, an empty set fades everything
export const applyHighlight = (root, hl) => {
  for (const g of root.querySelectorAll('[data-op]'))
    g.classList.toggle('hl', hl?.has(g.dataset.op) ?? false);
  root.querySelector('svg')?.classList.toggle('hlm', !!hl);
};

// binary byte formatter (the param-bytes lens and the bytes tally)
export const fmtBytes = (b) =>
  !b ? '0'   // empty PP stages: a true zero, not a rounded one
    : b >= 2 ** 40 ? (b / 2 ** 40).toFixed(2) + ' TiB'
      : b >= 2 ** 30 ? (b / 2 ** 30).toFixed(1) + ' GiB'
        : b >= 2 ** 20 ? (b / 2 ** 20).toFixed(1) + ' MiB' : (b / 1024).toFixed(1) + ' KiB';

// fit-chart geometry (svg units): the log₂ axis spans 2^lo…2^hi over bw px
export const BAR_GEO = { w: 800, x0: 110, bw: 650, lo: 28, hi: 44 };   // 110px name gutter (values live at bar ends)

// ---- fit-chart layout & blend ----------------------------------------------
// The chart never animates the MODEL: every render draws a LAYOUT — a pure
// pixel-space description of the picture (rows keyed by stable identity) —
// and every transition is the one rule blendFit(from, to, t):
//   · matched geometry lerps linearly in pixel space (the axis is log₂, so
//     linear pixel motion IS geometric byte motion; px() clamps at the axis
//     floor, so appearing/dying bars grow from / shrink to the floor)
//   · text, data-* attributes and row identity snap to the target
//   · fill colors lerp (the total flips dark↔light grey around a solo)
//   · one-sided rows fade — sub-rows collapse into their parent's row line,
//     whole bars shrink to the floor on the missing side
// The model is consulted once per transition (building the target layout),
// never per frame; the from-side is whatever is on screen when the tween
// starts, so an interrupted tween retargets smoothly instead of stuttering.
const FIT_ROWH = 12;
const fitLerp = (a, b, t) => a + (b - a) * t;
const fitColor = (a, b, t) => {
  if (a === b) return b;
  const c = (h, i) => parseInt(h.slice(i, i + 2), 16);
  return '#' + [1, 3, 5].map((i) =>
    Math.round(fitLerp(c(a, i), c(b, i), t)).toString(16).padStart(2, '0')).join('');
};
const fitEase = (p) => 1 - (1 - p) ** 3;   // ease-out cubic: response motion starts NOW
function blendFit(A, B, p) {
  if (!A || p >= 1) return B;
  const rowsA = new Map(A.rows.map((r) => [r.key, r]));
  const rowsB = new Map(B.rows.map((r) => [r.key, r]));
  // TWO PHASES when the save's ghosts move or appear: the baseline plants
  // itself FIRST (a quick ease-out over the opening share), THEN the bars
  // pour — "here is where you were… now watch it change". Same total
  // duration; no ghost change → single phase (a knob click answers NOW).
  const GS = 0.3;   // the ghosts' share of the tween
  const gChanged = B.rows.some((r) => {
    const g1 = rowsA.get(r.key)?.ghost, g2 = r.ghost;
    return !!g1 !== !!g2 || (g1 && g2 && Math.abs(g1.px - g2.px) > 0.5);
  }) || A.rows.some((r) => r.ghost && !rowsB.has(r.key));
  const tG = fitEase(gChanged ? Math.min(1, p / GS) : p);
  const t = fitEase(gChanged ? Math.max(0, (p - GS) / (1 - GS)) : p);
  const parentY = (r, L) => {
    const m = /^part:(\d+):/.exec(r.key);
    const par = m && L.rows.find((q) => q.key === `seg:${m[1]}`);
    return par ? par.y + FIT_ROWH + 2 : r.y;
  };
  // the missing side of a one-sided row: invisible, sub-rows tucked at the
  // parent's row line (the accordion fold), bars at the axis floor
  const faded = (r, L) => ({ ...r, op: 0, nameOp: 0, y: parentY(r, L),
    segs: r.segs.map((s) => r.type === 'part' ? { ...s, op: 0 } : { ...s, op: 0, x1: s.x0 }),
    ghost: r.ghost && { ...r.ghost, op: 0 },
    val: r.val && { ...r.val, op: 0 } });
  const seg = (a, b) => ({ ...b,
    x0: fitLerp(a.x0, b.x0, t), x1: fitLerp(a.x1, b.x1, t),
    op: fitLerp(a.op, b.op, t), color: fitColor(a.color, b.color, t) });
  const mark = (a, b, keys, t2) => !a && !b ? null : (() => {
    const m1 = a ?? { ...b, op: 0 }, m2 = b ?? { ...a, op: 0 };
    const out = { ...m2, op: fitLerp(m1.op, m2.op, t2) };
    for (const k of keys) out[k] = fitLerp(m1[k], m2[k], t2);
    return out;
  })();
  const row = (a, b) => ({ ...b,
    y: fitLerp(a.y, b.y, t), op: fitLerp(a.op, b.op, t), nameOp: fitLerp(a.nameOp, b.nameOp, t),
    segs: b.segs.map((sb) => seg(a.segs.find((s) => s.key === sb.key) ?? { ...sb, op: 0 }, sb)),
    ghost: mark(a.ghost, b.ghost, ['px'], tG),
    val: mark(a.val, b.val, ['x'], t) });
  const rows = B.rows.map((b) => row(rowsA.get(b.key) ?? faded(b, A), b));
  for (const a of A.rows) if (!rowsB.has(a.key)) rows.push(row(a, faded(a, B)));
  return { ...B, axisY: fitLerp(A.axisY, B.axisY, t), HB: fitLerp(A.HB, B.HB, t), rows };
}
// layout → svg string. Pure: no widget state, no model — width/position
// truths ride the data-* attributes for the visual audit.
function fitSvg(L) {
  const { w, x0, bw, lo: LO, hi: HI } = BAR_GEO;
  const topY = 14;
  const fo = (v) => String(Math.round(v * 1000) / 1000);
  const op = (v) => v < 0.999 ? ` opacity="${fo(v)}"` : '';
  const f1 = (v) => v.toFixed(1);
  const gx = (e) => x0 + (e - LO) / (HI - LO) * bw;
  const aY = L.axisY;
  // unrounded cross-check: value labels are re-emitted ABOVE the scrub
  // overlay (end of this function) so the pointer can reach them — their
  // data-true feeds the raw-bytes hover card (attachTip)
  const VALS = [];
  const B = [`<text class="grplabel" x="2" y="9">${L.hdr}</text>`];
  // the infeasible region is SHADED, not a line; its label sits ON TOP,
  // leaving the bottom axis to the power-of-two labels
  B.push(`<rect x="${f1(L.capPx)}" y="${topY - 2}" width="${f1(x0 + bw - L.capPx)}" ` +
    `height="${f1(aY - topY - 1)}" fill="${C('#0b0b0b')}" opacity="0.07"/>`);
  // unit swatch legend floats right in the header — only when the strip
  // squares it explains are actually mounted (pointless on bars-only views)
  if (L.unit) B.push(`<rect x="${x0 + bw - 96}" y="3" width="5" height="4" fill="${C('#898781')}"/>` +
    `<text class="dims" x="${x0 + bw - 87}" y="9">${L.unit}</text>`);
  for (let e = LO; e <= HI; e += 1)   // the ×2 grid
    B.push(`<line x1="${f1(gx(e))}" y1="${topY - 2}" x2="${f1(gx(e))}" y2="${f1(aY - 3)}" stroke="${C('#e1e0d9')}" stroke-width="1"/>`);
  for (const [e, lab] of [[30, '1 GiB'], [33, '8 GiB'], [36, '64 GiB'], [40, '1 TiB'], [43, '8 TiB']])
    B.push(`<text class="dims" x="${f1(gx(e) + 3)}" y="${f1(aY + 8)}">${lab}</text>`);
  for (const r of L.rows) {
    const y = r.y;
    if (r.type === 'part') {
      // breakdown sub-row: one <g> per row (group opacity carries the part
      // filter dim), name + bar + optional pin ghost + value
      const s = r.segs[0];
      B.push(`<g opacity="${r.op.toFixed(3)}"${r.part != null ? ` data-part="${r.part}" style="cursor:pointer"` : ''}>` +
        `<rect x="0" y="${f1(y - 2)}" width="${x0 - 4}" height="${FIT_ROWH - 2}" fill="transparent"/>` +
        `<text class="dims"${r.cell ? ` data-cell="${r.cell}"` : ''} x="12" y="${f1(y + 5.5)}">· ${r.name}</text>` +
        `<rect data-bar="${s.bar}" data-true="${s.true}" x="${f1(s.x0)}" y="${f1(y)}" width="${f1(Math.max(0.5, s.x1 - s.x0))}" height="5" fill="${s.color}" opacity="${fo(s.op)}"/>` +
        (r.ghost ? `<rect x="${x0}" y="${f1(y)}" width="${f1(Math.max(0.5, r.ghost.px - x0))}" height="5" ` +
          `fill="none" stroke="${r.ghost.color}" stroke-width="1" stroke-dasharray="2 2" opacity="${fo(r.ghost.op)}"/>` : '') +
        `</g>`);
      if (r.val) VALS.push(`<text class="dims" data-role="val:${r.id}" data-true="${r.val.true}" data-pin="${r.val.pin}"${r.cell ? ` data-cell="${r.cell}"` : ''} x="${f1(r.val.x)}" y="${f1(y + 5.5)}"${op(r.val.op * r.op)}>${r.val.text}</text>`);
      continue;
    }
    // gutter: the name alone (whole-row hitbox; click to solo)
    B.push(`<g${r.prop ? ` data-prop="${r.prop}" style="cursor:pointer"` : ''}${op(r.nameOp)}>` +
      (r.prop ? `<rect x="0" y="${f1(y - 2)}" width="${x0 - 4}" height="${FIT_ROWH}" fill="transparent"/>` : '') +
      `<text class="dims" data-role="name:${r.id}" data-true="${r.abs}"${r.cell ? ` data-cell="${r.cell}"` : ''} x="2" y="${f1(y + 7)}" fill="${r.color}" font-weight="600">${r.name}</text></g>`);
    // canonical segs [grey base | colored tips]: degenerate ones (a closed
    // stack) carry no data and skip rendering
    for (const s of r.segs) {
      if (s.bar == null && s.x1 - s.x0 < 0.35) continue;
      B.push(`<rect${s.bar != null ? ` data-bar="${s.bar}" data-true="${s.true}"` : ''} x="${f1(s.x0)}" y="${f1(y)}" ` +
        `width="${f1(Math.max(0.5, s.x1 - s.x0))}" height="8" fill="${s.color}"${op(s.op * r.op)}/>`);
    }
    // the save renders as a dotted GHOST bar (not a tick), so the value
    // label can always ride the live bar's end
    if (r.ghost) B.push(`<rect data-ghost="${r.id}" data-true="${r.ghost.true}" x="${x0}" y="${f1(y)}" width="${f1(Math.max(0.5, r.ghost.px - x0))}" height="8" ` +
      `fill="none" stroke="${r.ghost.color}" stroke-width="1" stroke-dasharray="2 2" opacity="${fo(r.ghost.op)}"/>`);
    // bar end: the ABSOLUTE value (+ the vs-save badge when saved)
    if (r.val) VALS.push(`<text class="dims" data-role="val:${r.id}" data-true="${r.val.true}" data-pin="${r.val.pin}"${r.cell ? ` data-cell="${r.cell}"` : ''} x="${f1(r.val.x)}" y="${f1(y + 7)}"${op(r.val.op)}>${r.val.text}</text>`);
  }
  // map-style DISTANCES legend: on a log axis a span IS a factor, so anchor
  // the important ones — the mesh dims (DP 2048 · EP 64 · PP 16) and a
  // halving — as one true-scale ruler below the axis
  const dy = aY + 18;
  const dpx = (f) => x0 + Math.log2(f) / (HI - LO) * bw;
  B.push(`<text class="dims" x="2" y="${f1(dy + 3)}">a span is a factor:</text>`);
  B.push(`<line x1="${x0}" y1="${f1(dy)}" x2="${f1(dpx(2048))}" y2="${f1(dy)}" stroke="${C('#898781')}" stroke-width="1"/>`);
  for (const [f, lab] of L.facs ?? FACS_02) {
    B.push(`<line data-fac="${f}" x1="${f1(dpx(f))}" y1="${f1(dy - 4)}" x2="${f1(dpx(f))}" y2="${f1(dy + 4)}" stroke="${C('#898781')}" stroke-width="1"/>`);
    if (lab) B.push(`<text class="dims" x="${f1(dpx(f))}" y="${f1(dy + 13)}" text-anchor="middle">${lab}</text>`);
  }
  // the pinned save label shares the legend band, right-aligned (its line is
  // always present now, so pinning still never reflows)
  if (L.lbl) B.push(`<text class="dims" x="${x0 + bw}" y="${f1(dy + 3)}" text-anchor="end">${L.lbl}</text>`);
  // the scrub overlay: cursor affordance AND arming region live exactly on
  // the bars band — not the captions, not below the axis
  B.push(`<rect class="scrub" x="${x0}" y="${topY - 2}" width="${bw}" height="${f1(aY - topY + 1)}" ` +
    `fill="transparent" style="cursor:col-resize"/>`);
  // value labels last: ABOVE the scrub, so their raw-byte hover titles work
  // (a drag can still start anywhere else on the band)
  B.push(...VALS);
  B.push(`<text class="dims" x="${f1(L.capPx)}" y="9" text-anchor="middle">${L.capLbl ?? '80 GiB (H100)'}</text>`);
  return `<svg width="${w}" height="${f1(L.HB)}" viewBox="0 0 ${w} ${f1(L.HB)}">${B.join('')}</svg>`;
}

// parameter-count formatter for the dims parentheticals ('(29M \u00d7256)' / '(7.5B)')
// change-badge magnitude formatting (▲×N / ▼×N), shared with the visual audit
export const facNum = (v) => v >= 100 || Math.abs(v - Math.round(v)) < 0.02 * v ? String(Math.round(v)) : v.toFixed(1);

export const fmtP = (n) => n >= 1e9 ? (n / 1e9).toFixed(1) + 'B'
  : n >= 9.95e6 ? Math.round(n / 1e6) + 'M' : n >= 1e6 ? (n / 1e6).toFixed(1) + 'M'
  : n >= 1e3 ? (n / 1e3).toFixed(1).replace(/\.0$/, '') + 'K' : String(n);
import { downloadTrace, openInPerfetto } from './trace.js';

// shared light-card tooltip style (trace, memory bars, schematic)
const TIP_CARD = 'position: absolute; pointer-events: none; background: var(--c-ffffff); color: var(--c-1c1c1a); padding: 6px 9px;' +
  ' border: 1px solid var(--c-c3c2b7); border-radius: 5px; display: none; box-shadow: 0 2px 10px rgba(11,11,11,0.12);';

// Validated categorical palette (dataviz skill, light surface var(--c-fcfcfb)).
export const CATS = {
  gemm: { c: '#2a78d6', ink: '#fff', label: 'GEMM' },
  attn: { c: '#eb6834', ink: '#fff', label: 'attention' },
  vector: { c: '#1baf7a', ink: '#0b0b0b', label: 'vector/norm' },
  a2a: { c: '#eda100', ink: '#0b0b0b', label: 'all-to-all' },
  fsdp: { c: '#e87ba4', ink: '#0b0b0b', label: 'FSDP coll.' },
  optimizer: { c: '#008300', ink: '#fff', label: 'optimizer' },
  p2p: { c: '#4a3aa7', ink: '#fff', label: 'pipeline p2p' },
  stall: { c: '#d03b3b', ink: '#fff', label: 'stall/GC' },     // status-critical, not a series
  phase: { c: '#e9e8e2', ink: '#52514e', label: 'microbatch' },
};

const GUTTER = 120, RULER = 20, LANE = 17, HEADER = 16;
const CSS = `
.tv { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-0b0b0b);
  border: 1px solid var(--c-e1e0d9); border-radius: 6px; background: var(--c-fcfcfb); overflow: hidden; }
.tv-bar { display: flex; align-items: center; gap: 8px; padding: 5px 8px; border-bottom: 1px solid var(--c-e1e0d9); flex-wrap: wrap; }
.tv-title { font-weight: 600; }
.tv-stats { color: var(--c-52514e); }
.tv-sp { flex: 1; }
.tv button { font: 11px system-ui; padding: 2px 8px; border: 1px solid var(--c-c3c2b7); border-radius: 4px;
  background: var(--c-ffffff); color: var(--c-0b0b0b); cursor: pointer; }
@media (hover: hover) { .tv button:hover { background: var(--c-f3f2ee); } }
.tv-legend { display: flex; gap: 10px; padding: 3px 8px; border-bottom: 1px solid var(--c-e1e0d9);
  color: var(--c-52514e); font-size: 11px; flex-wrap: wrap; }
.tv-legend span { display: inline-flex; align-items: center; gap: 4px; }
.tv-legend i { width: 9px; height: 9px; border-radius: 2px; display: inline-block; }
.tv-wrap { position: relative; }
.tv canvas { display: block; outline: none; }
.tv-tip { ${TIP_CARD} font-size: 11px; max-width: 340px; z-index: 5; line-height: 1.45; }
.tv-tip b { color: var(--c-0b0b0b); }
.tv-foot { padding: 3px 8px; border-top: 1px solid var(--c-e1e0d9); color: var(--c-52514e); font-size: 11px;
  min-height: 16px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.tv-help { position: absolute; top: 6px; right: 6px; background: rgba(11,11,11,.92); color: var(--c-ffffff);
  padding: 8px 12px; border-radius: 6px; font-size: 11px; z-index: 6; display: none; line-height: 1.7; }
`;

let hoveredViewer = null;
if (typeof window !== 'undefined') {
  window.addEventListener('keydown', (e) => {
    if (hoveredViewer && !e.metaKey && !e.ctrlKey) hoveredViewer.onKey(e);
  });
}

export class TraceViewer {
  constructor(container, trace, opts = {}) {
    this.opts = opts;
    container.innerHTML = '';
    const style = document.createElement('style');
    style.textContent = CSS;
    container.append(style);
    this.root = el('div', 'tv');
    container.append(this.root);

    this.bar = el('div', 'tv-bar');
    this.titleEl = el('span', 'tv-title'); this.statsEl = el('span', 'tv-stats');
    this.bar.append(this.titleEl, this.statsEl, el('span', 'tv-sp'));
    for (const [label, fn] of [
      ['⤓ trace.json', () => downloadTrace(this.trace, this.opts.title || 'dsv3-sim')],
      ['open in Perfetto', () => openInPerfetto(this.trace, this.opts.title || 'dsv3-sim')],
      ['?', () => this.helpEl.style.display = this.helpEl.style.display === 'block' ? 'none' : 'block'],
    ]) {
      const b = el('button'); b.textContent = label; b.onclick = fn; this.bar.append(b);
    }
    this.legendEl = el('div', 'tv-legend');
    this.wrap = el('div', 'tv-wrap');
    this.canvas = document.createElement('canvas');
    this.canvas.tabIndex = 0;
    this.tip = el('div', 'tv-tip');
    this.helpEl = el('div', 'tv-help');
    this.helpEl.innerHTML = '<b>navigation</b><br>W/S zoom · A/D pan · wheel scroll · ⌘/ctrl-wheel zoom<br>' +
      'shift-wheel pan · drag pan · click select · ←/→ walk slices<br>F focus selection · M mark · 0 fit · esc clear';
    this.foot = el('div', 'tv-foot');
    this.wrap.append(this.canvas, this.tip, this.helpEl);
    this.root.append(this.bar, this.legendEl, this.wrap, this.foot);

    this.height = opts.height ?? 300;
    this.sel = null; this.mark = null; this.mouse = null;
    this.bindEvents();
    new ResizeObserver(() => this.resize()).observe(this.root);
    this.setTrace(trace);
  }

  setTrace(trace) {
    this.trace = trace;
    this.sel = null; this.mark = null;
    const stats = trace.meta?.stats;
    this.titleEl.textContent = this.opts.title ?? '';
    this.statsEl.textContent = stats
      ? `step ${fmtUs(stats.stepUs)} · MFU ${(stats.mfu * 100).toFixed(1)}% · ${Math.round(stats.tokPerSecPerGpu)} tok/s/GPU`
      : '';
    if (stats?.mem) {
      const m = stats.mem;
      const memEl = el('span');
      memEl.textContent = ` · mem ${m.worst.total.toFixed(0)}/${m.capacityGB} GiB` + (m.fits ? '' : ' — does not fit ✗');
      memEl.style.color = m.fits ? C('#52514e') : C('#d03b3b');
      if (!m.fits) memEl.style.fontWeight = '600';
      this.statsEl.append(memEl);
    }
    this.buildRows();
    this.legendEl.innerHTML = '';
    for (const cat of this.catsPresent) {
      const s = el('span'); const i = el('i'); i.style.background = C(CATS[cat].c);
      s.append(i, document.createTextNode(CATS[cat].label)); this.legendEl.append(s);
    }
    this.resize(); this.fit();
  }

  buildRows() {
    this.rows = []; this.t0 = Infinity; this.t1 = 0;
    const cats = new Set();
    for (const rank of this.trace.ranks) {
      this.rows.push({ kind: 'header', label: rank.label, h: HEADER });
      for (const track of rank.tracks) {
        const lanes = Array.from({ length: track.lanes }, () => []);
        for (const s of track.slices) {
          lanes[Math.min(s.depth ?? 0, track.lanes - 1)].push(s);
          this.t0 = Math.min(this.t0, s.ts); this.t1 = Math.max(this.t1, s.ts + s.dur);
          cats.add(s.cat);
        }
        for (const l of lanes) l.sort((a, b) => a.ts - b.ts);
        this.rows.push({ kind: 'track', label: track.name, lanes, h: track.lanes * LANE + 2 });
      }
    }
    if (!isFinite(this.t0)) { this.t0 = 0; this.t1 = 1; }
    this.contentH = this.rows.reduce((a, r) => a + r.h, 0);
    this.catsPresent = [...Object.keys(CATS)].filter(c => cats.has(c));
    this.yOff = 0;
  }

  resize() {
    const w = this.root.clientWidth;
    if (!w) return;
    const h = Math.min(this.height, this.contentH + RULER + 4);
    const dpr = window.devicePixelRatio || 1;
    this.w = w; this.h = h;
    this.canvas.width = w * dpr; this.canvas.height = h * dpr;
    this.canvas.style.width = w + 'px'; this.canvas.style.height = h + 'px';
    this.ctx = this.canvas.getContext('2d');
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.dirty();
  }

  plotW() { return Math.max(50, this.w - GUTTER); }
  fit() {
    const range = (this.t1 - this.t0) || 1;
    this.tpp = range * 1.02 / this.plotW();
    this.tl = this.t0 - range * 0.01;
    this.dirty();
  }
  xOf(t) { return GUTTER + (t - this.tl) / this.tpp; }
  tOf(x) { return this.tl + (x - GUTTER) * this.tpp; }
  clampView() {
    const range = this.t1 - this.t0 || 1;
    this.tpp = Math.min(Math.max(this.tpp, range / (this.plotW() * 4000)), range * 2 / this.plotW());
    this.tl = Math.min(Math.max(this.tl, this.t0 - range), this.t1 + range * 0.05 - this.plotW() * this.tpp * 0.05);
    this.yOff = Math.min(Math.max(0, this.yOff), Math.max(0, this.contentH - (this.h - RULER)));
  }
  zoomAt(x, factor) {
    const t = this.tOf(x);
    this.tpp *= factor;
    this.clampView();
    this.tl = t - (x - GUTTER) * this.tpp;
    this.dirty();
  }

  dirty() { if (!this._raf) this._raf = requestAnimationFrame(() => { this._raf = null; this.draw(); }); }

  draw() {
    const { ctx, w, h } = this;
    if (!ctx) return;
    this.clampView();
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = C('#fcfcfb'); ctx.fillRect(0, 0, w, h);
    this.drawRuler();
    ctx.save();
    ctx.beginPath(); ctx.rect(0, RULER, w, h - RULER); ctx.clip();
    let y = RULER - this.yOff;
    for (const row of this.rows) {
      if (y + row.h > RULER && y < h) this.drawRow(row, y);
      y += row.h;
    }
    ctx.restore();
    this.drawMark();
    if (this.selRow != null && this.sel) this.drawSelection();
  }

  drawRuler() {
    const { ctx } = this;
    ctx.fillStyle = C('#f9f9f7'); ctx.fillRect(0, 0, this.w, RULER);
    const target = 110 * this.tpp;
    const pow = Math.pow(10, Math.floor(Math.log10(target)));
    const step = [1, 2, 5, 10].map(m => m * pow).find(s => s >= target) ?? pow * 10;
    ctx.font = '10px system-ui'; ctx.fillStyle = C('#898781'); ctx.strokeStyle = C('#e1e0d9');
    const start = Math.floor(this.tl / step) * step;
    for (let t = start; t < this.tOf(this.w); t += step) {
      const x = this.xOf(t);
      if (x < GUTTER) continue;
      ctx.beginPath(); ctx.moveTo(x, RULER - 4); ctx.lineTo(x, this.h); ctx.stroke();
      ctx.fillText(fmtUs(t), x + 3, 12);
    }
    ctx.strokeStyle = C('#c3c2b7');
    ctx.beginPath(); ctx.moveTo(0, RULER + .5); ctx.lineTo(this.w, RULER + .5); ctx.stroke();
  }

  drawRow(row, y) {
    const { ctx } = this;
    if (row.kind === 'header') {
      ctx.fillStyle = C('#f3f2ee'); ctx.fillRect(0, y, this.w, row.h);
      ctx.fillStyle = C('#0b0b0b'); ctx.font = 'bold 10px system-ui';
      ctx.fillText(row.label, 6, y + 12);
      return;
    }
    ctx.fillStyle = C('#898781'); ctx.font = '10px system-ui';
    ctx.fillText(row.label, 14, y + 12);
    const tr = this.tOf(this.w);
    row.lanes.forEach((lane, li) => {
      const ly = y + li * LANE + 1, lh = LANE - 2;
      let i = lowerBound(lane, this.tl);
      if (i > 0 && lane[i - 1].ts + lane[i - 1].dur > this.tl) i--;
      let mx0 = null, mx1 = 0, mcat = null; // merged run of sub-pixel slices
      const flushMerged = () => {
        if (mx0 == null) return;
        ctx.fillStyle = C(CATS[mcat]?.c ?? '#898781');
        ctx.fillRect(mx0, ly, Math.max(mx1 - mx0, 0.6), lh);
        mx0 = null;
      };
      for (; i < lane.length && lane[i].ts < tr; i++) {
        const s = lane[i];
        const x0 = Math.max(this.xOf(s.ts), GUTTER), x1 = Math.min(this.xOf(s.ts + s.dur), this.w);
        const sw = x1 - x0;
        if (sw < 0.8) {
          if (mx0 != null && (x0 - mx1 > 1.5 || s.cat !== mcat)) flushMerged();
          if (mx0 == null) { mx0 = x0; mcat = s.cat; }
          mx1 = Math.max(mx1, x1);
          continue;
        }
        flushMerged();
        const cat = CATS[s.cat] ?? { c: C('#898781'), ink: C('#ffffff') };
        ctx.fillStyle = C(cat.c);
        ctx.fillRect(x0, ly, Math.max(sw - 0.5, 0.6), lh);
        if (sw > 34) {
          ctx.fillStyle = C(cat.ink);
          const chars = Math.floor(sw / 6);
          ctx.fillText(s.name.length > chars ? s.name.slice(0, chars - 1) + '…' : s.name, x0 + 3, ly + 12);
        }
      }
      flushMerged();
    });
  }

  drawMark() {
    if (!this.mark) return;
    const { ctx } = this;
    const x0 = Math.max(this.xOf(this.mark.t0), GUTTER), x1 = this.xOf(this.mark.t1);
    if (x1 < GUTTER || x0 > this.w) return;
    ctx.fillStyle = 'rgba(42,120,214,0.10)'; ctx.fillRect(x0, RULER, x1 - x0, this.h - RULER);
    ctx.strokeStyle = C('#2a78d6');
    for (const x of [x0, x1]) { ctx.beginPath(); ctx.moveTo(x, RULER); ctx.lineTo(x, this.h); ctx.stroke(); }
    const label = fmtUs(this.mark.t1 - this.mark.t0);
    ctx.font = 'bold 10px system-ui';
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = C('#2a78d6'); ctx.fillRect((x0 + x1) / 2 - tw / 2 - 4, RULER + 2, tw + 8, 14);
    ctx.fillStyle = C('#ffffff'); ctx.fillText(label, (x0 + x1) / 2 - tw / 2, RULER + 13);
  }

  drawSelection() {
    const s = this.sel;
    const y = this.rowY(this.selRow) + this.selLane * LANE + 1;
    if (y == null) return;
    this.ctx.strokeStyle = C('#0b0b0b'); this.ctx.lineWidth = 1.5;
    this.ctx.strokeRect(this.xOf(s.ts), y, Math.max(s.dur / this.tpp, 1.5), LANE - 2);
    this.ctx.lineWidth = 1;
  }

  rowY(rowIdx) {
    let y = RULER - this.yOff;
    for (let i = 0; i < rowIdx; i++) y += this.rows[i].h;
    return y;
  }

  hitTest(x, y) {
    if (y < RULER || x < GUTTER) return null;
    let ry = RULER - this.yOff;
    for (let ri = 0; ri < this.rows.length; ri++) {
      const row = this.rows[ri];
      if (y < ry + row.h) {
        if (row.kind !== 'track') return null;
        const lane = Math.min(Math.floor((y - ry - 1) / LANE), row.lanes.length - 1);
        if (lane < 0) return null;
        const t = this.tOf(x);
        const arr = row.lanes[lane];
        let i = lowerBound(arr, t) - 1;
        for (const j of [i, i + 1]) {
          const s = arr[j];
          if (s && t >= s.ts && t <= s.ts + s.dur) return { slice: s, row: ri, lane, arr, idx: j };
        }
        return null;
      }
      ry += row.h;
    }
    return null;
  }

  select(hit) {
    this.sel = hit?.slice ?? null;
    this.selRow = hit?.row; this.selLane = hit?.lane; this.selArr = hit?.arr; this.selIdx = hit?.idx;
    const s = this.sel;
    this.foot.textContent = s
      ? `${s.name} — ${CATS[s.cat]?.label ?? s.cat} · start ${fmtUs(s.ts)} · dur ${fmtUs(s.dur)}` +
      (s.args ? ' · ' + Object.entries(s.args).filter(([, v]) => v != null && v !== '')
        .map(([k, v]) => `${k}=${typeof v === 'number' ? v : v}`).join(' · ') : '')
      : '';
    this.dirty();
  }

  bindEvents() {
    const cv = this.canvas;
    this.wrap.addEventListener('mouseenter', () => { hoveredViewer = this; });
    this.wrap.addEventListener('mouseleave', () => { if (hoveredViewer === this) hoveredViewer = null; this.tip.style.display = 'none'; this.mouse = null; });
    cv.addEventListener('wheel', (e) => {
      e.preventDefault();
      if (e.ctrlKey || e.metaKey) this.zoomAt(e.offsetX, Math.exp(e.deltaY * 0.01));
      else if (e.shiftKey) { this.tl += e.deltaY * this.tpp; this.dirty(); }
      else { this.tl += e.deltaX * this.tpp; this.yOff += e.deltaY; this.dirty(); }
    }, { passive: false });
    let down = null, moved = false;
    cv.addEventListener('mousedown', (e) => { down = { x: e.offsetX, y: e.offsetY, tl: this.tl, yOff: this.yOff }; moved = false; cv.focus({ preventScroll: true }); });
    cv.addEventListener('mousemove', (e) => {
      this.mouse = { x: e.offsetX, y: e.offsetY };
      if (down) {
        if (Math.abs(e.offsetX - down.x) + Math.abs(e.offsetY - down.y) > 3) moved = true;
        this.tl = down.tl - (e.offsetX - down.x) * this.tpp;
        this.yOff = down.yOff - (e.offsetY - down.y);
        this.dirty();
      } else this.hover(e);
    });
    window.addEventListener('mouseup', () => { down = null; });
    cv.addEventListener('click', (e) => { if (!moved) this.select(this.hitTest(e.offsetX, e.offsetY)); });
    cv.addEventListener('dblclick', (e) => {
      const hit = this.hitTest(e.offsetX, e.offsetY);
      if (hit) this.focusSlice(hit.slice);
    });
  }

  hover(e) {
    const hit = this.hitTest(e.offsetX, e.offsetY);
    if (!hit) { this.tip.style.display = 'none'; return; }
    const s = hit.slice;
    const args = s.args ? Object.entries(s.args).filter(([, v]) => v != null && v !== '')
      .map(([k, v]) => `${k}: ${v}`).join('<br>') : '';
    this.tip.innerHTML = `<b>${esc(s.name)}</b><br>${CATS[s.cat]?.label ?? s.cat} · ${fmtUs(s.dur)}` + (args ? '<br>' + args : '');
    this.tip.style.display = 'block';
    const bw = this.wrap.clientWidth;
    this.tip.style.left = Math.min(e.offsetX + 14, bw - this.tip.offsetWidth - 4) + 'px';
    this.tip.style.top = (e.offsetY + 16) + 'px';
  }

  focusSlice(s) {
    this.tpp = Math.max(s.dur / (this.plotW() * 0.4), 1e-6);
    this.tl = s.ts - (this.plotW() * 0.3) * this.tpp;
    this.dirty();
  }

  onKey(e) {
    const cx = this.mouse?.x ?? (GUTTER + this.plotW() / 2);
    const pan = this.plotW() * 0.12 * this.tpp;
    const k = e.key.toLowerCase();
    if (k === 'w') this.zoomAt(cx, 1 / 1.35);
    else if (k === 's') this.zoomAt(cx, 1.35);
    else if (k === 'a') { this.tl -= pan; this.dirty(); }
    else if (k === 'd') { this.tl += pan; this.dirty(); }
    else if (k === 'f' && this.sel) this.focusSlice(this.sel);
    else if (k === '0') this.fit();
    else if (k === 'm' && this.sel) {
      const m = { t0: this.sel.ts, t1: this.sel.ts + this.sel.dur };
      this.mark = this.mark && this.mark.t0 === m.t0 && this.mark.t1 === m.t1 ? null : m;
      this.dirty();
    } else if (k === 'escape') { this.select(null); this.mark = null; this.dirty(); }
    else if ((k === 'arrowleft' || k === 'arrowright') && this.selArr) {
      const idx = this.selIdx + (k === 'arrowleft' ? -1 : 1);
      if (idx >= 0 && idx < this.selArr.length) {
        this.select({ slice: this.selArr[idx], row: this.selRow, lane: this.selLane, arr: this.selArr, idx });
        const s = this.sel;
        if (this.xOf(s.ts) < GUTTER || this.xOf(s.ts + s.dur) > this.w)
          this.tl = s.ts - this.plotW() * 0.1 * this.tpp;
        this.dirty();
      }
    } else return;
    e.preventDefault();
  }
}

function lowerBound(arr, t) {
  let lo = 0, hi = arr.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (arr[mid].ts < t) lo = mid + 1; else hi = mid; }
  return lo;
}
function el(tag, cls) { const e = document.createElement(tag); if (cls) e.className = cls; return e; }
function esc(s) { return String(s).replace(/[<>&]/g, c => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;' }[c])); }

// ---- <dsv3-trace> custom element ---------------------------------------------
// Attributes: level (0-6), title, height, plus any sim config key via
// config='{"microbatches": 12, "hardware": "gb200", ...}'.
export class Dsv3Trace extends HTMLElement {
  static observedAttributes = ['level', 'config', 'title', 'height'];
  connectedCallback() { this.render(); }
  attributeChangedCallback() { if (this.isConnected) this.queueRender(); }
  queueRender() {
    if (this._q) return;
    this._q = true;
    queueMicrotask(() => { this._q = false; this.render(); });
  }
  configOverrides() {
    const o = this.getAttribute('config') ? JSON.parse(this.getAttribute('config')) : {};
    if (this.hasAttribute('level')) o.level = +this.getAttribute('level');
    return o;
  }
  render() {
    try {
      this.result = simulate(this.configOverrides());
    } catch (err) {
      this.textContent = 'sim error: ' + err.message;
      throw err;
    }
    const level = this.result.cfg.level;
    const title = this.getAttribute('title') ?? `level ${level}: ${LEVELS[level].title}`;
    const height = this.hasAttribute('height') ? +this.getAttribute('height') : undefined;
    if (!this.viewer) this.viewer = new TraceViewer(this, this.result.trace, { title, height });
    else { this.viewer.opts.title = title; this.viewer.setTrace(this.result.trace); }
    this.dispatchEvent(new CustomEvent('sim', { detail: this.result }));
  }
}
if (typeof customElements !== 'undefined' && !customElements.get('dsv3-trace')) {
  customElements.define('dsv3-trace', Dsv3Trace);
}

// ---- shared linking -------------------------------------------------------------
// Patch the config of linked widgets by element id (`for="id1 id2"`):
// <dsv3-trace> gets its `config` patched.
// Interactive state persists in the URL hash (shareable, survives refresh):
// #c:<id>=<json> for control strips, l:<id>=<json> for schematics.
function readUrlState(key) {
  try {
    const v = new URLSearchParams(location.hash.slice(1)).get(key);
    return v ? JSON.parse(v) : null;
  } catch { return null; }
}
function writeUrlState(key, obj) {
  const p = new URLSearchParams(location.hash.slice(1));
  p.set(key, JSON.stringify(obj));
  history.replaceState(null, '', '#' + p.toString());
}
function clearUrlState(key) {
  const p = new URLSearchParams(location.hash.slice(1));
  p.delete(key);
  const s = p.toString();
  history.replaceState(null, '', s ? '#' + s : location.pathname + location.search);
}

export function patchTargets(forAttr, patch) {
  for (const id of (forAttr ?? '').split(/[ ,]+/).filter(Boolean)) {
    const t = document.getElementById(id);
    if (!t) continue;
    const cfg = JSON.parse(t.getAttribute('config') ?? '{}');
    t.setAttribute('config', JSON.stringify({ ...cfg, ...patch }));
  }
}

// ---- <dsv3-layer> custom element ------------------------------------------------
// Top-down SVG schematic of one DSv3 transformer block (plus the head).
// Every matmul carries a dtype <select>; the chosen per-matmul precisions are
// pushed to the widgets named in `for="id1 id2"` (memory now, rooflines later).
// dtype colors are the PRECISION family — warm magentas graded by element
// width (e4m3 pink → e5m6 purple → fp32 brick), deliberately clear of the
// byte-component language (weights blue / grads orange / optim green) so a
// picket never impersonates a byte bar. fp8 (Hopper tile-scaled) and mxfp8
// (Blackwell MX) share the pink — same bytes, different provenance.
const DT_STYLE = { bf16: '#52514e', e4m3: '#d6408b', mxfp8: '#d6408b', e5m6: '#7b2fa8', fp32: '#8a3324' };
// e5m6 is a STASH format: the GEMM that reads it runs plain e4m3. Box tags,
// pickets and ribbon runs speak COMPUTE dtype (this mapping); chips speak
// stash format (dtOf) — purple appears only where the E5M6 fact lives.
const COMPUTE_DT = (d) => d === 'e5m6' ? 'e4m3' : d;
// the diagram's visual-language tokens (docs/diagram-grammar.md) — one
// definition, scoped into each widget's stylesheet (the anatomy plan too)
export const tokensCss = (s) => `
${s} .wire { stroke: var(--c-898781); stroke-width: 1.2; fill: none; }
${s} .box { fill: var(--c-ffffff); stroke: var(--c-c3c2b7); }
${s} .op { fill: var(--c-f3f2ee); stroke: var(--c-e1e0d9); }
${s} .comm { fill: var(--c-f3f1fb); stroke: var(--c-6b5bd2); }
${s} .res { fill: var(--c-fcfcfb); stroke: var(--c-c3c2b7); stroke-dasharray: 3 2; }
${s} .grp { fill: none; stroke: var(--c-e1e0d9); }
${s} .name { font: 600 11px system-ui; fill: var(--c-0b0b0b); }
${s} .dims { font: 9px system-ui; fill: var(--c-898781); }
${s} .oplabel { font: 10.5px system-ui; fill: var(--c-52514e); }
${s} .grplabel { font: italic 10px system-ui; fill: var(--c-898781); }
${s} .plus { font: 600 12px system-ui; fill: var(--c-52514e); }
`;
// the local-knob control-strip styles (steppers, grouped rows) — shared by
// the layer's mini-head and <dsv3-pp-schedule>'s replicated pipeline group
const knobCss = (s) => `
${s} .pargrp { display: inline-flex; flex-direction: column; gap: 2px;
  border: 1px solid var(--c-e1e0d9); border-radius: 6px; padding: 3px 8px 5px; align-self: stretch; }
${s} .pargrp.center { justify-content: center; }
${s} .parlab { font: italic 10px system-ui; color: var(--c-898781); }
${s} .parrow { display: flex; align-items: center; gap: 5px; min-height: 20px; }
${s} .stp { display: inline-flex; align-items: stretch; }
${s} .stp button { font: 12px ui-monospace, monospace; width: 20px; padding: 0 0 1px; border: 1px solid var(--c-c3c2b7); background: var(--c-ffffff); color: var(--c-52514e); cursor: pointer; }
@media (hover: hover) { ${s} .stp button:hover:not(:disabled) { background: var(--c-f3f2ee); } }
${s} .stp button:disabled { color: var(--c-dedcd3); cursor: default; }
${s} .stp button:first-child { border-radius: 4px 0 0 4px; }
${s} .stp button:last-child { border-radius: 0 4px 4px 0; }
${s} .stp button + button { border-left: none; }
${s} .stp button.on { background: var(--c-f3f2ee); color: var(--c-0b0b0b); font-weight: 600; cursor: default; }
${s} .stp button { width: auto; min-width: 20px; padding: 0 5px 1px; }
${s} .stp select.v { font: 11px ui-monospace, monospace; min-width: 4ch; padding: 2px 5px;
  border: 1px solid var(--c-c3c2b7); border-left: none; border-right: none; border-radius: 0;
  background: var(--c-ffffff); appearance: none; -webkit-appearance: none; text-align: center;
  text-align-last: center; cursor: pointer; }
${s} select { font: 12px system-ui; padding: 2px 6px; border: 1px solid var(--c-c3c2b7); border-radius: 4px; background: var(--c-ffffff); }
`;

const LAYER_CSS = `
dsv3-layer { display: block; margin: 14px 0 26px; }
.lv { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-0b0b0b);
  border: 1px solid var(--c-e1e0d9); border-radius: 6px; background: var(--c-fcfcfb); padding: 10px 12px; position: relative; }
.lv-tip { ${TIP_CARD} font-size: 11.5px; max-width: 360px; z-index: 7; line-height: 1.5; white-space: pre-line; }
.lv-tip.pinned { border-color: var(--c-eda100); box-shadow: 0 2px 10px rgba(237,161,0,0.3); }
/* cell (formula) tooltips: entries stack downward as the pinned drill deepens */
.lv-cellent + .lv-cellent { border-top: 1px dashed var(--c-e1e0d9); margin-top: 5px; padding-top: 5px; }
.lv-cellfx { font: 11px ui-monospace, monospace; color: var(--c-52514e); margin-top: 1px; }
.lv-tip .cellref { color: var(--c-2a78d6); font-weight: 600; }
.lv-tip.pinned .cellref { cursor: pointer; text-decoration: underline dotted; }
.lv-tip.pinned .celljump { cursor: pointer; }
@media (hover: hover) { .lv-tip.pinned .celljump:hover { text-decoration: underline; } }
.lv-cellhint { color: var(--c-898781); font-size: 10px; margin-top: 5px; }
.lv-head { display: flex; align-items: center; gap: 8px; padding-bottom: 6px; color: var(--c-52514e); flex-wrap: wrap; }
.lv-head select { font: 12px system-ui; padding: 2px 6px; border: 1px solid var(--c-c3c2b7); border-radius: 4px; background: var(--c-ffffff); }
.lv-head .savebox { margin-left: auto; display: inline-flex; gap: 6px; align-items: flex-start;
  padding-left: 12px; border-left: 1px solid var(--c-e1e0d9); align-self: flex-start; }
${knobCss('.lv-head')}
.lv.lv-compact { padding: 6px 12px; }
.lv-compact .lv-head { padding-bottom: 4px; }
/* anatomy-wrapped quant tiers park the recompute policy in the PLAN column
   (left of the card, below the plan — saves a whole head row on laptops).
   Coordinates assume the anatomy grid: 186px column + 28px gap, plan ends
   ~400px into the card. Narrow viewports return it to the flow. */
.lv .lv-side { position: absolute; left: -215px; top: 412px; width: 186px; display: block; padding: 0; }
/* in the margin the segment dissolves into a SPACED STACK of rounded chips
   (the plan column's language: standalone rounded boxes, air between them) —
   no group border, no fused segment edges */
.lv-side .pargrp { display: flex; width: 100%; box-sizing: border-box; border: none; padding: 0; gap: 4px; }
.lv-side .parlab { margin-bottom: 1px; }
.lv-side .parrow { min-height: 0; }
.lv-side .stp { flex-direction: column; align-items: stretch; gap: 5px; width: 100%; }
.lv-side .stp button { text-align: left; padding: 3px 10px 4px; border: 1px solid var(--c-c3c2b7); border-radius: 5px; }
.lv-side .stp button + button { border-left: 1px solid var(--c-c3c2b7); border-top: 1px solid var(--c-c3c2b7); }
.lv-side .stp button:disabled { border-color: var(--c-e1e0d9); }
@media (max-width: 1040px) {
  .lv .lv-side { position: static; width: auto; display: flex; padding-bottom: 4px; }
  .lv-side .pargrp { display: inline-flex; width: auto; }
  .lv-side .stp { flex-direction: row; }
  .lv-side .stp button { border-left: none; border-top: 1px solid var(--c-c3c2b7); border-radius: 0; }
  .lv-side .stp button:first-child { border-left: 1px solid var(--c-c3c2b7); border-radius: 4px 0 0 4px; }
  .lv-side .stp button:last-child { border-radius: 0 4px 4px 0; }
}
.lv-compact .lv-foot2 svg { padding-top: 2px; }
.lv svg { display: block; margin: 0 auto; }
/* no scaling, ever: a diagram wider than its container scrolls horizontally */
.lv-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
/* snapshots are figures: the card shrink-wraps its chart; the gutter legend
   is not clickable there (config-static), only the measuring scrub is */
dsv3-layer[snapshot] .lv { width: fit-content; max-width: 100%; }
dsv3-layer[snapshot] .lv-bar [data-part], dsv3-layer[snapshot] .lv-bar [data-prop] { cursor: default !important; }
/* hypothetical beats: a counterfactual 'to' — the dashed card border is the
   same not-real cue the ghost bars use, plus an explicit italic tag */
dsv3-layer[hypothetical] .lv { border-style: dashed; }
/* snapshot knob panels are READOUTS: values stay ink, ± steppers hide,
   segment selections keep their face */
dsv3-layer[snapshot] .lv-head button:disabled, dsv3-layer[snapshot] .lv-head select:disabled { opacity: 1; color: var(--c-0b0b0b); cursor: default; }
dsv3-layer[snapshot] .lv-head .stp button.on:disabled { background: var(--c-f3f2ee); color: var(--c-0b0b0b); }
/* read-only knob chips (the ctx group): a locked value keeps its ink */
.lv-head .stp button.on:disabled { background: var(--c-f3f2ee); color: var(--c-0b0b0b); opacity: 1; cursor: default; }
dsv3-layer[snapshot] .lv-head .stp button:disabled:not(.on) { color: var(--c-c3c2b7); }
dsv3-layer[snapshot] .lv-head .stp:has(select.v) button { display: none; }
dsv3-layer[snapshot] .lv-head .stp:has(select.v) select.v { border-left: 1px solid var(--c-c3c2b7); border-right: 1px solid var(--c-c3c2b7); border-radius: 4px; }
dsv3-layer[snapshot] .lv-head select:disabled { appearance: none; -webkit-appearance: none; background: var(--c-ffffff); }
.lv-hyptag { font: italic 10.5px system-ui; color: var(--c-898781); padding-bottom: 4px; }
.lv-bar svg { display: block; margin: 2px 0 6px; max-width: 100%; height: auto; }
.lv-bar { position: relative; }
.lv-ruler { display: none; position: absolute; background: rgba(237, 161, 0, 0.12);
  border-left: 1px solid var(--c-0b0b0b); border-right: 1px solid var(--c-0b0b0b); pointer-events: none; }
.lv-ruler-lab { position: absolute; top: -2px; left: 100%; margin-left: 5px; white-space: nowrap;
  font: 11px ui-monospace, monospace; color: var(--c-0b0b0b); background: var(--c-fff8ea); padding: 1px 4px;
  border: 1px solid var(--c-eda100); border-radius: 3px; }
${tokensCss('.lv')}
.lv select.dt { font: 600 10px system-ui; width: 100%; height: 20px; border: 1px solid var(--c-c3c2b7);
  border-radius: 3px; background: var(--c-ffffff); }
.lv button.st { display: block; width: 100%; height: 18px; font: 10px system-ui; border-radius: 3px;
  cursor: pointer; text-align: left; padding: 0 5px; overflow: hidden; white-space: nowrap; }
.lv .st-save { background: var(--c-fff8ea); border: 1px solid var(--c-eda100); color: var(--c-0b0b0b); }
.lv .st-redo { background: var(--c-f3f2ee); border: 1px dashed var(--c-898781); color: var(--c-52514e); }
.lv .st-keep { background: var(--c-ffffff); border: 1px solid var(--c-c3c2b7); color: var(--c-898781); text-decoration: line-through; }
.lv button.st.mode { width: 24px; padding: 0; text-align: center; height: 20px; }
.lv button.st.dtb { width: 52px; padding: 0; text-align: center; height: 20px; font-weight: 600;
  background: var(--c-ffffff); border: 1px solid var(--c-c3c2b7); }
.lv text.tensor { font: 10px system-ui; }
.lv .tsave { fill: var(--c-7a5200); font-weight: 600; }
.lv .tdim { fill: var(--c-898781); font-weight: 400; }
.lv .micro { fill: var(--c-f7f6f1); stroke: var(--c-d8d6cb); }
.lv svg.hlm > :not(.hl):not(defs) { opacity: 0.3; }
.lv g[data-op].hl .dims { fill: var(--c-52514e); font-weight: 600; }
.lv .microlabel { font: italic 10px system-ui; fill: var(--c-52514e); }
.lv .tredo { fill: var(--c-52514e); font-style: italic; }
.lv .tidle { fill: var(--c-a8a69e); }
.lv-note { color: var(--c-898781); font-size: 11px; padding-top: 6px; max-width: 640px; }
.lv-foot2 { display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }
.lv-foot2 .lv-note { flex: 1 1 420px; }
.lv-foot2 svg { flex: none; padding-top: 8px; }
`;
export class Dsv3Layer extends HTMLElement {
  connectedCallback() {
    this.urlKey = 'l:' + (this.id || 'layer');
    this._origRecipe = this.getAttribute('recipe') ?? 'nv-mxfp8';
    this._origRecompute = this.getAttribute('recompute') ?? 'dsv3';
    const st = this.hasAttribute('snapshot') ? null : readUrlState(this.urlKey);
    if (st?.recipe) this.setAttribute('recipe', st.recipe);
    this.matmuls = st?.matmuls ?? resolveMatmuls({ recipe: this.getAttribute('recipe') ?? 'nv-mxfp8' });
    this.marks = st?.marks ?? { ...RECOMPUTE_PRESETS[this.getAttribute('recompute') ?? 'dsv3'] };
    // display scaling: 'combined' totals the block column (× layers × in-flight × tokens)
    this.view = st?.view ?? 'combined';
    this.dispLayers = st?.dispLayers ?? +(this.getAttribute('xlayers') ?? 61);
    this.dispInflight = st?.dispInflight ?? +(this.getAttribute('xinflight') ?? 1);
    this.transposed = st?.transposed ?? this.hasAttribute('transposed');

    this.detail = st?.detail ?? this.hasAttribute('detail');
    this.flatDims = st?.flatDims ?? false;
    // optim/consolidated lenses: which byte components are visible (strips AND
    // numbers follow — the numbers always total exactly what the squares show)
    this.showWeights = st?.showWeights ?? true;
    this.showOptim = st?.showOptim ?? true;
    this.showGrads = st?.showGrads ?? true;
    this.showActs = st?.showActs ?? true;
    // local lens: the fiat-parallelism selectors (EP width or 1 = off, PP
    // degree, PP stage, ZeRO-1 on/off)
    // the page seeds the fiat parallelism through attributes (02 = the
    // LOCAL_PAR defaults: 2048 GPUs · PP8 · EP64 · DualPipeV; 03 sets
    // world/pp/ep/vpp/sched/hw for its cluster); URL state overrides
    const A = (k) => this.getAttribute(k);
    this.ep = st?.ep ?? +(A('ep') ?? 64);
    this.pp = st?.pp ?? +(A('pp') ?? LOCAL_PAR.pp);
    this.zero = st?.zero ?? (st?.zero1 === false ? 0 : 1);   // ZeRO level 0–3 (1 = DSv3)
    this.world = st?.world ?? +(A('world') ?? LOCAL_PAR.world);   // cluster size (GPUs)
    this.sched = st?.sched === 'dpv' ? '1f1b' : st?.sched ?? A('sched') ?? '1f1b';   // admission: '1f1b' (DualPipeV) | 'interleaved' (Megatron) | 'one' (a single microbatch)
    this.hw = st?.hw ?? A('hw') ?? 'h100';                   // the capacity yardstick (HARDWARE key)
    this.a2a = st?.a2a ?? this.hasAttribute('a2a');          // Megatron's 1F1B all-to-all overlap (one extra warmup forward)
    this.tp = st?.tp ?? +(A('tp') ?? 1);                       // tensor parallelism (Megatron: sequence parallel on, expert-TP 1)
    this.gradB = st?.gradB ?? (A('grads') === 'bf16' ? 2 : 4);   // the gradient buffer's bytes/param (02: fp32; Megatron's perf recipes: bf16)
    this.fp8Params = st?.fp8Params ?? this.hasAttribute('fp8params');
    // schedule geometry (schedGeom): under DualPipeV VPP and fold are DERIVED
    // from pp, never knobs (pp > 1 → the V: 2 chunks/rank, reflect); under
    // Megatron's interleaved 1F1B the page/URL sets VP and the layout string
    Object.assign(this, schedGeom({ pp: this.pp, sched: this.sched, fold: st?.fold ?? A('fold') ?? undefined,
      vpp: st?.vpp ?? (A('vpp') ? +A('vpp') : undefined), layout: st?.layout ?? A('layout') ?? undefined, a2a: this.a2a }));
    // default to the PEAK stage — the fully loaded rank is the story; the
    // selector is there to peek at the lighter ones
    this.stage = Math.min(st?.stage ?? peakStage(this.pp, this.ep, this.zero, this.world, this.sched, this.vpp, this.fold, this.layout, this.a2a, this.tp), this.pp - 1);
    // cumulative: every parameter parenthetical multiplies by the selected
    // kind's block count (×3 dense / ×58 MoE); the tabs hide — the kind then
    // comes from the plan selector alone. The local variant is ALWAYS
    // cumulative (no per-block toggle: the fit bar totals the rank, so a
    // per-block diagram would disagree with the chart it explains)
    this.cumulative = this.hasAttribute('local') && this.getAttribute('lens') === 'param-bytes'
      ? true : st?.cumulative ?? this.hasAttribute('cumulative');
    // which block variant to draw: the MLA column is identical; only the FFN
    // column differs (kind="dense" pins the dense-FFN variant, default MoE)
    this.kind = st?.kind ?? (this.getAttribute('kind') === 'dense' ? 'dense' : 'moe');
    this.render();
    if (this.hasAttribute('snapshot')) {
      // a SNAPSHOT is a non-interactive story beat: 'from' renders as the
      // saved baseline (ghosts), 'to' as the live bars with change badges;
      // 'solo' picks the component under discussion (total goes stacked).
      // Static by design: no knobs, no URL state, no chart interactions.
      const from = JSON.parse(this.getAttribute('from') ?? '{}');
      const to = JSON.parse(this.getAttribute('to') ?? '{}');
      this._snapCfg = { from, to };
      // which components the beat shows: solo= (one comp, stacked total,
      // breakdown open) or comps= (a visible subset — the additive-tally
      // beats); both dim the rest to name-only rows
      const P2 = { weights: 'showWeights', grads: 'showGrads', optim: 'showOptim', acts: 'showActs' };
      const soloOf = P2[this.getAttribute('solo')];
      if (soloOf) for (const p2 of this._compProps()) this[p2] = p2 === soloOf;
      const comps = this.getAttribute('comps');
      if (comps) {
        const on = comps.split(/[ ,]+/).map((k) => P2[k]);
        for (const p2 of this._compProps()) this[p2] = on.includes(p2);
      }
      this._applyCfg({ ...CFG_DEFAULTS, ...from });
      this.render();
      if (this.hasAttribute('to')) {   // no 'to' = a single-config figure: no ghosts, no badges
        this._saveBaseline();
        this._applyCfg({ ...CFG_DEFAULTS, ...from, ...to });
        this.render();
      }
    }
    queueMicrotask(() => this.changed(false)); // push initial recipe + marks to linked widgets
  }
  changed(write = true) {
    // explicit map over every markable op (true = save), so it overrides any
    // recompute preset an authored row config pins. Static tiers speak
    // save-everything (the structure view makes no recompute claims).
    const marksEff = (this.getAttribute('controls') ?? 'full') === 'static' ? RECOMPUTE_PRESETS.none : this.marks;
    const savedMap = Object.fromEntries(MARKABLE.map(id => [id, marksEff[id] === true]));
    const detail = { matmuls: { ...this.matmuls }, saved: savedMap };
    this.dispatchEvent(new CustomEvent('recipe', { detail }));
    // recompute:'none' + explicit marks = exactly this.marks (preset merged already)
    patchTargets(this.getAttribute('for'), {
      recipe: null, matmuls: detail.matmuls, recompute: 'none', saved: detail.saved,
      transposedStash: this.transposed,
    });
    if (write && !this.hasAttribute('snapshot')) writeUrlState(this.urlKey, {
      recipe: this.getAttribute('recipe'), matmuls: this.matmuls, marks: this.marks,
      view: this.view, dispLayers: this.dispLayers, dispInflight: this.dispInflight,
      transposed: this.transposed, detail: this.detail, flatDims: this.flatDims,
      fp8Params: this.fp8Params,
      kind: this.kind, cumulative: this.cumulative,
      showWeights: this.showWeights, showOptim: this.showOptim,
      showGrads: this.showGrads, showActs: this.showActs,
      ep: this.ep, stage: this.stage, pp: this.pp, zero: this.zero, world: this.world, sched: this.sched,
      hw: this.hw, vpp: this.vpp, fold: this.fold, layout: this.layout, a2a: this.a2a, gradB: this.gradB, tp: this.tp,
    });
  }
  applyPreset(recipe, recompute, transposed = false) {
    this.setAttribute('recipe', recipe);
    this.setAttribute('recompute', recompute);
    this.matmuls = resolveMatmuls({ recipe });
    this.marks = { ...RECOMPUTE_PRESETS[recompute] };
    this.transposed = transposed;
    clearUrlState(this.urlKey);
    this.render(); this.changed(false);
  }
  toggleMark(ids) {
    ids = ids.map(markKey);   // tied nodes (the SwiGLU-input quantize) edit their partner's mark
    const mutate = () => {   // save-driven marks: {id: true} = save; unlisted = recompute
      const on = this.marks[ids[0]] === true;
      for (const id of ids) { if (on) delete this.marks[id]; else this.marks[id] = true; }
    };
    if (this.hasAttribute('local') && this.getAttribute('lens') === 'param-bytes') {
      const prev = this._snapLocal(); mutate(); this._tweenLocal(prev);
    } else if (this._ctl?.quant) this._tweenQuant(mutate);
    else { mutate(); this.render(); this.changed(); }
  }
  // stash-knob tween for the quant tiers (marks/dtype/full, non-local): the
  // previous analysis + per-matmul dtypes are the lerp endpoints — FLOP bars
  // stretch, tally ribbons pour, saved-tensor chips dissolve and their grids
  // pour. Numbers snap (house rule).
  _tweenQuant(mutate) {
    const prev = { anaPrev: this._anaMemo?.ana, mm: { ...this.matmuls }, marks: { ...this.marks }, transposed: this.transposed };
    mutate();
    this.changed(true);
    this._frames((t) => { this._vtween = { t, prev }; }, () => { this._vtween = undefined; });
  }
  // local-knob mutations shared by the head controls and external drivers
  // (<dsv3-pp-schedule>): callers go through setLocal so every change tweens
  setLocal(mutate) { const prev = this._snapLocal(); mutate(); this._tweenLocal(prev); }
  _setPP(v) {
    const world = this.world ?? LOCAL_PAR.world;
    this.pp = v;
    // DualPipeV wherever a pipeline exists; the Megatron family keeps its VP
    // and re-derives the default layout for the new depth
    Object.assign(this, schedGeom({ pp: v, sched: this.sched, fold: this.fold, vpp: this.vpp, a2a: this.a2a }));
    this.ep = Math.min(this.ep, world / v);
    this.stage = peakStage(v, this.ep, this.zero ?? 1, world, this.sched, this.vpp, this.fold, this.layout, this.a2a, this.tp ?? 1);   // stage indices don't survive a resplit — jump to the new peak
  }
  _setTP(v) {
    this.tp = v;
    this.stage = peakStage(this.pp, this.ep, this.zero ?? 1, this.world ?? LOCAL_PAR.world, this.sched, this.vpp, this.fold, this.layout, this.a2a, v);
  }
  _setVPP(v) {   // Megatron family only (the knob renders there alone)
    Object.assign(this, schedGeom({ pp: this.pp, sched: this.sched, fold: 'wrap', vpp: v, a2a: this.a2a }));
    this.stage = peakStage(this.pp, this.ep, this.zero ?? 1, this.world ?? LOCAL_PAR.world, this.sched, this.vpp, this.fold, this.layout, this.a2a, this.tp ?? 1);
  }
  // ---- local-lens knob tween: EVERY knob change (EP/PP/stage/ZeRO/×N) pours
  // squares between the old and new configuration, per-block-tween style.
  // Numbers snap; a change that flips the kind snaps (different layout).
  // THE frame driver: every animation in this widget is the same
  // deterministic 12-frame ease-out loop (~200 ms, timer-driven so it's
  // steady under headless/virtual time). onFrame(t) mutates the tween state,
  // then the widget re-renders; done() clears it.
  _frames(onFrame, done) {
    // 12 frames (~200 ms) for knob twiddling; hosts that NARRATE a change
    // (the beat deck) set _tweenFrames higher so the pour reads as a story
    const FRAMES = this._tweenFrames ?? 12; let f = 0;
    const gen = this._frameGen = (this._frameGen ?? 0) + 1;
    // the fit chart tweens as a LAYOUT BLEND: from whatever is on screen at
    // this instant (mid-tween retargets included) to the new state's layout
    if (this._fitL) this._ftween = { L0: this._fitL, t: 0 };
    onFrame(0);
    // paint t=0 NOW: between starting a tween and the first timer tick the
    // DOM may hold some other synchronously-rendered state (the deck renders
    // the baseline to build its pin) — one visible beat of it is a flash
    this.render(); this.changed(false);
    // fitEase (ease-out cubic): these tweens RESPOND to a click, and
    // response motion must start immediately (in-out's slow first beat reads
    // as lag) and decelerate into place. Duration handles gravitas.
    const ease = fitEase;
    const step = () => {
      if (this._frameGen !== gen) return;   // superseded by a newer tween
      f++; const p = Math.min(1, f / FRAMES);
      if (this._ftween) this._ftween.t = p;   // RAW progress: the blend eases per phase
      onFrame(ease(p));
      this.render(); this.changed(false);     // linked widgets tween along
      if (p < 1) setTimeout(step, 16);
      else { this._ftween = undefined; done(); this.render(); this.changed(true); }
    };
    setTimeout(step, 16);
  }
  // tween a set of byte-component visibility changes: squares pour in/out
  _compTween(props) {
    this._frames((t) => { this._ctween = { props, t }; }, () => { this._ctween = undefined; });
  }
  toggleComp(prop, on) {
    this[prop] = on;
    this._compTween(new Set([prop]));
  }
  // sub-part filter (inside a solo): click 'experts' etc — same idea one
  // level down. null = all parts.
  togglePart(k) {
    const prev = this.partSel ?? null;
    this.partSel = prev === k ? null : k;
    this._frames((t) => { this._ptween = { prev, t }; }, () => { this._ptween = undefined; });
  }
  // legend clicks SOLO a component (the useful filter: "show me only the
  // weights"); soloing the already-solo component brings everything back
  _compProps() {
    const cons = this.hasAttribute('consolidated') || this.hasAttribute('local');
    return cons ? ['showWeights', 'showGrads', 'showOptim', 'showActs'] : ['showWeights', 'showOptim'];
  }
  soloComp(prop) {
    this.partSel = null;   // the part filter lives inside a solo
    const props = this._compProps();
    const already = this[prop] && props.every((p2) => p2 === prop || !this[p2]);
    const changed = new Set();
    for (const p2 of props) {
      const want = already ? true : p2 === prop;
      if (this[p2] !== want) { this[p2] = want; changed.add(p2); }
    }
    if (changed.size) this._compTween(changed);
  }
  // lock in the CURRENT config as the chart's baseline (ghost bars + factor
  // badges render vs this) — the save button, and snapshot mode's 'from'
  _saveBaseline() {
    this._pinCfg = {
      segs: [...(this._segTotals ?? [])],
      parts: (this._segParts ?? []).map((p2) => [...p2]),
      scalars: { ...(this._scalars ?? {}) },
      state: { ep: this.ep, pp: this.pp, stage: this.stage, world: this.world,
        zero: this.zero, sched: this.sched, vpp: this.vpp, fold: this.fold,
        cumulative: this.cumulative, partSel: this.partSel ?? null,
        showWeights: this.showWeights, showGrads: this.showGrads,
        showOptim: this.showOptim, showActs: this.showActs,
        transposed: this.transposed, marks: { ...this.marks }, matmuls: { ...this.matmuls } },
      label: `EP${this.ep}·PP${this.pp}${(this.tp ?? 1) > 1 ? `·TP${this.tp}` : ''}·rank ${this.stage}·ZeRO-${this.zero ? this.zero : 'off'}·${schedName(this)}·${this.world} GPUs${this.hw && this.hw !== 'h100' ? '·' + HW_SHORT[this.hw] : ''}`,
    };
  }
  // apply an authored config patch (snapshot 'from'/'to', sandbox jumps):
  // plain state keys plus recipe/recompute presets
  _applyCfg(patch) {
    const { recipe, recompute, stage, ...rest } = patch;
    Object.assign(this, rest);
    // geometry is DERIVED from the patch (schedGeom): DualPipeV ignores
    // authored vpp/fold keys; interleaved/wrap configs carry vpp + layout
    Object.assign(this, schedGeom({ pp: this.pp, sched: this.sched, fold: patch.fold, vpp: patch.vpp, layout: patch.layout, a2a: patch.a2a }));
    if (recipe) { this.setAttribute('recipe', recipe); this.matmuls = resolveMatmuls({ recipe }); }
    if (recompute) { this.setAttribute('recompute', recompute); this.marks = { ...RECOMPUTE_PRESETS[recompute] }; }
    this.stage = stage ?? peakStage(this.pp, this.ep, this.zero ?? 1,
      this.world ?? LOCAL_PAR.world, this.sched, this.vpp, this.fold, this.layout, this.a2a, this.tp ?? 1);
  }
  // jump target for snapshots' "open in the full widget" links: land on the
  // snapshot's exact story — its 'from' as the save, its 'to' live, tweened
  _loadScenario(from, to) {
    this._applyCfg({ ...CFG_DEFAULTS, ...from }); this.render(); this._saveBaseline();
    const prev = this._snapLocal();
    this._applyCfg({ ...CFG_DEFAULTS, ...from, ...to });
    this._tweenLocal(prev);
    this.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
  // ---- fit-chart interaction wiring: ONE seam for every view ---------------
  // Capabilities compose per view, so mixed pages (full sim + barsonly panels
  // + snapshot figures) each get exactly the interactions that make sense:
  //   measure — the scrub cursor + drag ruler (READ-ONLY: a span on a log
  //             axis is a factor); every view has it, snapshots included
  //   mutate  — solo/part clicks on the gutter legend; off in snapshots,
  //             which are config-static figures
  _barCaps() {
    return { measure: true, mutate: !this.hasAttribute('snapshot') };
  }
  _wireBars(barSlot) {
    const caps = this._barCaps();
    // scrub cursor: one vertical line you click/drag along the axis — the
    // readout is the value there and its factor vs the 80 GiB capacity
    // (a log axis makes that distance a multiplier). Click ON it to clear.
    const svgEl2 = barSlot.querySelector('svg');
    const toU = (ev) => {   // client → svg units, clamped to the axis
      const r2 = svgEl2.getBoundingClientRect();
      const u = (ev.clientX - r2.left) / r2.width * BAR_GEO.w;
      return Math.max(BAR_GEO.x0, Math.min(BAR_GEO.x0 + BAR_GEO.bw, u));
    };
    const bytesAt = (u) => 2 ** (BAR_GEO.lo + (u - BAR_GEO.x0) / BAR_GEO.bw * (BAR_GEO.hi - BAR_GEO.lo));
    const rul = el('div', 'lv-ruler');
    const rlab = el('div', 'lv-ruler-lab');
    rul.append(rlab);
    barSlot.append(rul);
    const fmtF = (f) => f >= 100 || Math.abs(f - Math.round(f)) < 0.02 * f ? String(Math.round(f)) : f.toFixed(1);
    const drawR = () => {
      const C = this._cursor;
      // Perfetto behavior: nothing shows without an actual drag
      if (!C || Math.abs(C.a - C.b) < 3) { rul.style.display = 'none'; return; }
      // rect math, not offsetLeft: SVG elements have no offsetLeft, which
      // left this at NaNpx (the line never met the cursor)
      const r2 = svgEl2.getBoundingClientRect(), host = barSlot.getBoundingClientRect();
      const k = r2.width / BAR_GEO.w;
      const [u1, u2] = [Math.min(C.a, C.b), Math.max(C.a, C.b)];
      rul.style.display = 'block';
      rul.style.left = `${(r2.left - host.left + u1 * k).toFixed(1)}px`;
      rul.style.width = `${((u2 - u1) * k).toFixed(1)}px`;
      // the ruler spans exactly the bars band (the scrub rect), never the
      // axis labels or the distances legend below it
      const sc = barSlot.querySelector('.scrub').getBoundingClientRect();
      rul.style.top = `${(sc.top - host.top).toFixed(1)}px`;
      rul.style.height = `${sc.height.toFixed(1)}px`;
      // a span on a log axis IS a factor
      const f = 2 ** ((u2 - u1) / BAR_GEO.bw * (BAR_GEO.hi - BAR_GEO.lo));
      rlab.textContent = `×${fmtF(f)} (${fmtBytes(bytesAt(u1))} → ${fmtBytes(bytesAt(u2))})`;
    };
    barSlot.onmousedown = (ev) => {
      if (caps.mutate) {
        const tp = ev.target.closest?.('[data-part]');
        if (tp) { this._cursor = null; drawR(); this.togglePart(+tp.dataset.part); return; }
        const tog = ev.target.closest?.('[data-prop]');
        if (tog) { this._cursor = null; drawR(); this.soloComp(tog.dataset.prop); return; }
      }
      if (!caps.measure) return;
      // the ruler arms only on the scrub overlay (the bars band itself)
      if (!ev.target.classList?.contains('scrub')) return;
      const u0 = toU(ev);
      this._cursor = { a: u0, b: u0 };
      const move = (e2) => { this._cursor.b = toU(e2); drawR(); };
      const up = () => {
        document.removeEventListener('mousemove', move);
        document.removeEventListener('mouseup', up);
        if (this._cursor && Math.abs(this._cursor.a - this._cursor.b) < 3) { this._cursor = null; drawR(); }
      };
      document.addEventListener('mousemove', move);
      document.addEventListener('mouseup', up);
      drawR();
      ev.preventDefault();
    };
    // ANY mousedown outside THIS chart's bars band dismisses its ruler
    // (another chart's scrub included — with several charts per page, only
    // the one being measured keeps a ruler). Deduped across renders.
    if (this._rulDismiss) document.removeEventListener('mousedown', this._rulDismiss);
    this._rulDismiss = (ev) => {
      if (barSlot.contains(ev.target) && ev.target.classList?.contains('scrub')) return;
      if (this._cursor) { this._cursor = null; drawR(); }
    };
    document.addEventListener('mousedown', this._rulDismiss);
    drawR();
  }
  _snapLocal() {
    return { ep: this.ep, pp: this.pp, stage: this.stage,
      zero: this.zero ?? 1, world: this.world ?? LOCAL_PAR.world,
      sched: this.sched ?? '1f1b', vpp: this.vpp ?? 1, fold: this.fold ?? 'reflect', layout: this.layout ?? null,
      hw: this.hw ?? 'h100', a2a: !!this.a2a, gradB: this.gradB ?? 4, mx: Object.values(this.matmuls).includes('mxfp8'), tp: this.tp ?? 1, cum: !!this.cumulative,
      fp8p: !!this.fp8Params && this.matmuls.ffn_gate_up !== 'bf16',
      // the pre-change analysis: stash-affecting knobs (precision, marks,
      // fp8ᵀ) lerp the diagram's chip squares between old and new bytes
      anaPrev: this._anaMemo?.ana };
  }
  _tweenLocal(prev) {
    this.changed(true);
    const kindOf = (S) => ppStage(Math.min(S.stage, S.pp - 1), S.pp, S.vpp, S.fold, S.layout).moe ? 'moe' : 'dense';
    if (kindOf(prev) !== kindOf(this._snapLocal())) { this.render(); return; }
    this._frames((t) => { this._vtween = { t, prev }; }, () => { this._vtween = undefined; });
  }
  render() {
    this.innerHTML = '';
    // one microbatch is a 4096-token sequence; under tensor parallelism every
    // stash divides by TP (sequence parallel shards the residual/MoE path by
    // tokens, attention shards heads, the head shards the vocabulary), so the
    // chips and readouts count this GPU's SHARE — 4096 / TP tokens
    const TOK = this._tok();
    // local lens: the kind follows the selected PP stage (stage 0 holds the
    // 3 dense blocks; every other stage holds only MoE blocks)
    if (this.hasAttribute('local') && this.getAttribute('lens') === 'param-bytes')
      this.kind = ppStage(this.stage ?? 1, this.pp, this.vpp, this.fold, this.layout).moe ? 'moe' : 'dense';
    const style = document.createElement('style'); style.textContent = LAYER_CSS;
    const root = el('div', 'lv');
    // progressive disclosure: controls="static|marks|dtype|full" gates which
    // controls are rendered (the diagram and its derived annotations always draw)
    const cmode = this.getAttribute('controls') ?? 'full';
    // the marks/dtype tiers fight for laptop vspace (the whole widget should
    // fit one screen): tighter card + head + tally paddings, scoped so the
    // LIVE static/full tiers keep their published geometry
    if (cmode === 'marks' || cmode === 'dtype') root.classList.add('lv-compact');
    // scope: how much of the model this instance draws
    // (model = block + head row · block = the block alone · mla/ffn = one column)
    const SCOPE = this.getAttribute('scope') ?? 'model';
    // static = pure structure: save-everything semantics, no quantities
    // (FLOP strips, bytes, grids, dtype tags), no tooltips, minimal caption
    this._ctl = {
      marks: cmode === 'full' || cmode === 'marks',
      dtype: cmode === 'full' || cmode === 'dtype',
      quant: cmode !== 'static',
    };
    // the local variant carries the AC + precision knobs too (all the memory
    // levers in one place) — but quant stays FALSE: the bytes lens has no
    // visual language for FLOPs, so no FLOP strips/tallies/replay notes
    const LOCALKNOBS = this.hasAttribute('local') && this.getAttribute('lens') === 'param-bytes';
    if (LOCALKNOBS) this._ctl = { marks: true, dtype: true, quant: false };
    const head = el('div', 'lv-head');
    // block-variant select: the MLA column is shared; only the FFN column swaps
    const mkKindSel = () => {
      const s = document.createElement('select');
      for (const [v, t] of [['moe', 'MoE block'], ['dense', 'dense block']]) {
        const o = document.createElement('option'); o.value = v; o.textContent = t; o.selected = v === this.kind; s.append(o);
      }
      s.title = 'sparse (MoE) vs dense block — the MLA half is identical; only the FFN column differs';
      s.onchange = () => { this.kind = s.value; this.render(); this.changed(true); };
      return s;
    };
    if (this.hasAttribute('tabs')) head.append('DSv3 block');   // the tabs carry the flip
    else head.append('DSv3 ', mkKindSel());
    if (this._ctl.dtype) head.append(' · precision: ');
    const preset = document.createElement('select');
    // an instance may curate its recipe chips (recipes="bf16,dsv3-fp8"):
    // e.g. the Hopper article drops nv-mxfp8 (that's the Blackwell post's)
    const recipeOpts = (this.getAttribute('recipes')?.split(/[ ,]+/).filter(k => RECIPES[k])) ?? Object.keys(RECIPES);
    for (const name of recipeOpts) {
      const o = document.createElement('option'); o.value = o.textContent = name; preset.append(o);
    }
    // recognize the current matmul dtypes as a recipe (dtype buttons may have
    // moved us off the attribute's preset), else show "custom". The stash-side
    // CHECKBOXES count too: each recipe has a canonical e4m3ᵀ state (RECIPE_T)
    // and the E5M6 choice rides mm.o_proj — flip either and you are custom
    const mmKey = mmSig;   // + the SwiGLU-input stash channel
    const curRecipe = recipeOpts.find(k => mmKey(resolveMatmuls({ recipe: k })) === mmKey(this.matmuls)
      && (RECIPE_T[k] ?? false) === !!this.transposed);
    preset.value = curRecipe ?? 'bf16';
    if (!curRecipe) {
      const o = document.createElement('option'); o.value = o.textContent = 'custom'; o.selected = true; preset.append(o);
    }
    const localTween = (mutate) => {   // stash knobs animate like every other knob
      if (this.hasAttribute('local') && this.getAttribute('lens') === 'param-bytes') {
        const prev = this._snapLocal(); mutate(); this._tweenLocal(prev);
      } else if (this._ctl.quant) this._tweenQuant(mutate);
      else { mutate(); this.render(); this.changed(); }
    };
    preset.onchange = () => {
      if (preset.value === 'custom') return;
      localTween(() => {
        this.setAttribute('recipe', preset.value);
        this.matmuls = resolveMatmuls({ recipe: preset.value });
        this.transposed = RECIPE_T[preset.value] ?? false;   // recipes are full stash-policy bundles
      });
    };
    if (this._ctl.dtype) head.append(preset);
    if (this._ctl.marks) head.append(' · recompute: ');
    const rsel = document.createElement('select');
    for (const name of Object.keys(RECOMPUTE_PRESETS)) {
      const o = document.createElement('option'); o.value = o.textContent = name; rsel.append(o);
    }
    const marksKey = markSig;
    const curPreset = Object.keys(RECOMPUTE_PRESETS).find(k => marksKey(RECOMPUTE_PRESETS[k]) === marksKey(this.marks));
    rsel.value = curPreset ?? 'none';
    if (!curPreset) {
      const o = document.createElement('option'); o.value = o.textContent = 'custom'; o.selected = true; rsel.append(o);
    }
    rsel.onchange = () => {
      if (rsel.value === 'custom') return;
      localTween(() => {
        this.setAttribute('recompute', rsel.value);
        this.marks = { ...RECOMPUTE_PRESETS[rsel.value] };
      });
    };
    if (this._ctl.marks) head.append(rsel);
    head.append(' · view: ');
    const vsel = document.createElement('select');
    for (const [v, t] of [['combined', 'combined'], ['layer', 'per layer']]) {
      const o = document.createElement('option'); o.value = v; o.textContent = t; o.selected = v === this.view; vsel.append(o);
    }
    vsel.onchange = () => { this.view = vsel.value; this.render(); this.changed(true); };
    head.append(vsel);
    const numIn = (label, get, set) => {
      head.append(' ' + label + ' ');
      const i = document.createElement('input');
      i.type = 'number'; i.value = get(); i.style.cssText = 'width:44px;font:12px system-ui;padding:1px 4px;border:1px solid var(--c-c3c2b7);border-radius:3px;';
      i.onchange = () => { set(Math.max(1, +i.value || 1)); this.render(); this.changed(true); };
      head.append(i);
    };
    if (this.view === 'combined') {
      numIn('×layers', () => this.dispLayers, (v) => this.dispLayers = v);
      numIn('×in-flight', () => this.dispInflight, (v) => this.dispInflight = v);
    }
    const reset = document.createElement('button');
    reset.textContent = 'reset';
    reset.style.cssText = 'font:11px system-ui;margin-left:auto;padding:2px 8px;border:1px solid var(--c-c3c2b7);border-radius:4px;background:var(--c-ffffff);cursor:pointer;';
    reset.onclick = () => {
      this.setAttribute('recipe', this._origRecipe);
      this.matmuls = resolveMatmuls({ recipe: this._origRecipe });
      this.marks = { ...RECOMPUTE_PRESETS[this._origRecompute] };
      this.view = 'combined';
      this.dispLayers = +(this.getAttribute('xlayers') ?? 61);
      this.dispInflight = +(this.getAttribute('xinflight') ?? 1);
      this.transposed = this.hasAttribute('transposed');
      this.fp8Params = false;
      this.detail = this.hasAttribute('detail');
      this.flatDims = false;
      this.cumulative = this.hasAttribute('cumulative');
      this.kind = this.getAttribute('kind') === 'dense' ? 'dense' : 'moe';
      // the local lens' own knobs — factory = the degenerate "whole model,
      // one GPU" view: PP1, EP1, ZeRO off (recipe/recompute reset to the
      // instance attrs above: bf16, none)
      this.ep = 1; this.pp = 1; this.world = LOCAL_PAR.world;
      this.zero = 0; this.sched = '1f1b'; this.vpp = 1; this.fold = 'reflect';
      this.stage = 0;
      this.showWeights = this.showGrads = this.showOptim = this.showActs = true;
      this.partSel = null;
      this._pinCfg = null; this._cursor = null;
      clearUrlState(this.urlKey);
      this.render(); this.changed(false);
    };
    const tl = document.createElement('label');
    tl.style.cssText = 'display:inline-flex;align-items:center;gap:4px;margin-left:8px;color:var(--c-52514e);';
    tl.title = 'The wgrad GEMM reads its saved activations TRANSPOSED, and 1×128 tile scales don’t survive that. ' +
      'ON = pay in memory: quantize with transpose at forward, stash BOTH orientations. ' +
      'OFF = pay in compute: one copy, re-quantized (dequant → transpose → requant) during backward — DeepSeek’s own convention. ' +
      'In practice this is an EXPERT-PATH choice: under realistic recompute policies the only fp8 stashes a wgrad still reads are the ' +
      'MoE-FFN inputs (the post-norm token stream and its dispatched copies) — the attention side is either replayed or kept E5M6 ' +
      '(single copy, pow-2 scales requantize the flip losslessly). The model generalizes honestly: any fp8 stash a wgrad reads would dual.';
    const tcb = document.createElement('input');
    tcb.type = 'checkbox'; tcb.checked = this.transposed; tcb.dataset.knob = 'transposed';
    tcb.onchange = () => localTween(() => { this.transposed = tcb.checked; });
    tl.append(tcb, 'e4m3ᵀ dual stash (expert inputs)');
    // the E5M6 sibling: the OTHER stash-format checkbox. It writes mm.o_proj
    // (e5m6 ⇄ bf16), so recipe recognition sees it automatically; the all-fp8
    // recipe pins the attn-out stash to e4m3 and the checkbox greys out.
    const tl2 = document.createElement('label');
    tl2.style.cssText = 'display:inline-flex;align-items:center;gap:4px;margin-left:8px;color:var(--c-52514e);';
    tl2.title = 'DeepSeek’s customized 12-bit stash format, exclusively for the attention output (it feeds both ' +
      'attention backward and the attn-out wgrad — too sensitive for e4m3, and pow-2 scales make the 1×128 → 128×1 ' +
      'orientation flip lossless from ONE copy). OFF = stash it bf16 instead' +
      (this.marks.attn === true ? ` (+${(16384 * 0.5 * TOK * this.dispLayers * this.dispInflight / 2 ** 30).toFixed(1)} GiB at this policy)` : ' (moot right now: this recompute policy replays attention, so attn-out is never stashed)') +
      '. This checkbox chooses the SAVE format only — the GEMM’s compute dtype follows the recipe (its 🔒 tag).';
    const t2cb = document.createElement('input');
    t2cb.type = 'checkbox'; t2cb.checked = this.matmuls.o_proj === 'e5m6'; t2cb.dataset.knob = 'e5m6';
    t2cb.disabled = this.matmuls.o_proj === 'e4m3' || this.matmuls.o_proj === 'mxfp8';
    if (t2cb.disabled) tl2.title = 'this recipe stashes the attention output in ' + this.matmuls.o_proj + ' (both orientations under ᵀ) — the E5M6 trick is the dsv3 recipe’s move';
    t2cb.onchange = () => localTween(() => { this.matmuls = { ...this.matmuls, o_proj: t2cb.checked ? 'e5m6' : 'bf16' }; });
    tl2.append(t2cb, 'E5M6 attn-out stash');
    if (this._ctl.dtype) head.append(tl, tl2);
    // local: the multiplier is the stage's block count, not the whole model's
    const KBLK = this.hasAttribute('local') && this.getAttribute('lens') === 'param-bytes'
      ? (this.kind === 'dense' ? ppStage(this.stage ?? 1, this.pp, this.vpp, this.fold, this.layout).dense
        : ppStage(this.stage ?? 1, this.pp, this.vpp, this.fold, this.layout).moe)
      : this.kind === 'dense' ? (DSV3.denseLayers ?? 3) : DSV3.layers - (DSV3.denseLayers ?? 3);
    const mkCumBtn = () => {
      const b = document.createElement('button');
      b.style.cssText = 'font:11px ui-monospace,monospace;padding:2px 8px;border:1px solid var(--c-c3c2b7);' +
        'border-radius:4px;background:var(--c-ffffff);cursor:pointer;margin-left:8px;min-width:9ch;box-sizing:content-box;';
      b.textContent = this.cumulative ? `\u00d7${KBLK} blocks` : 'per block';
      b.title = 'toggle parameter counts: one block vs cumulative over all blocks of this kind ' +
        '(the tabs hide in cumulative mode \u2014 the multiplier follows the selected block kind)';
      b.onclick = () => {
        if (this.hasAttribute('local') && this.getAttribute('lens') === 'param-bytes') {
          const prev = this._snapLocal();
          this.cumulative = !this.cumulative;
          this._tweenLocal(prev);
          return;
        }
        this.cumulative = !this.cumulative;
        this.changed(true);
        if (this.getAttribute('strips') === 'absolute' && this.getAttribute('lens') === 'param-bytes') {
          const from = this._tween ?? (this.cumulative ? 0 : 1);
          const to = this.cumulative ? 1 : 0;
          this._frames((t) => { this._tween = from + (to - from) * t; }, () => { this._tween = undefined; });
        } else this.render();
      };
      return b;
    };
    const mkDimsBtn = () => {
      const b = document.createElement('button');
      b.style.cssText = 'font:11px ui-monospace,monospace;padding:2px 8px;border:1px solid var(--c-c3c2b7);' +
        'border-radius:4px;background:var(--c-ffffff);cursor:pointer;margin-left:8px;';
      b.textContent = this.flatDims ? '24576' : '128\u00d7192';
      b.title = 'toggle sizes: factored (128\u00d7192) vs multiplied out (24576)';
      b.onclick = () => { this.flatDims = !this.flatDims; this.render(); this.changed(true); };
      return b;
    };
    const dl = document.createElement('label');
    dl.style.cssText = 'display:inline-flex;align-items:center;gap:4px;margin-left:8px;color:var(--c-52514e);';
    dl.title = 'Also draw the kernels the terse view folds away: the latent RMSNorms, RoPE, the router’s ' +
      'sigmoid/top-k, the shared-expert GEMMs, the dispatched top-k weights, and the gating multiply. ' +
      'Display only — they are cheap, carry no marks, and don’t change the model.';
    const dcb = document.createElement('input');
    dcb.type = 'checkbox'; dcb.checked = this.detail;
    dcb.onchange = () => { this.detail = dcb.checked; this.render(); this.changed(true); };
    dl.append(dcb, 'elided kernels');
    head.append(dl, mkDimsBtn(), mkCumBtn(), reset);
    // memoized: tween frames re-render 12× with identical analysis inputs —
    // recomputing the graph walk each frame is what made toggles sluggish
    const marksEff2 = this._ctl.quant || this._ctl.marks ? this.marks : RECOMPUTE_PRESETS.none;
    const anaKey = `${this.kind}|${cmode}|${this._ctl.marks}|${JSON.stringify(this.matmuls)}|` +
      `${JSON.stringify(this.marks)}|${this.transposed}`;
    if (this._anaMemo?.key !== anaKey) {
      this._anaMemo = {
        key: anaKey,
        ana: analyze(blockGraph(this.kind, DSV3, this.matmuls, 4096), marksEff2, this.transposed),
        // dense mode also analyzes the MoE graph, purely for LAYOUT (row
        // alignment across kind flips)
        anaM: this.kind === 'dense'
          ? analyze(blockGraph('moe', DSV3, this.matmuls, 4096), marksEff2, this.transposed)
          : null,
        // local fit charts price mixed-kind ranks exactly: the dense
        // front's layers stash at the DENSE rate (same marks/recipe)
        anaD: this.hasAttribute('local') && this.getAttribute('lens') === 'param-bytes' && this.kind !== 'dense'
          ? analyze(blockGraph('dense', DSV3, this.matmuls, 4096), marksEff2, this.transposed)
          : null,
        // …and the SAVE-EVERYTHING rates at the same recipe/ᵀ: the cell
        // graph's bucket formulas factor the recompute choice out as a 0/1
        // (R•) times these policy-independent rates
        anaMF: this.hasAttribute('local') && this.getAttribute('lens') === 'param-bytes'
          ? analyze(blockGraph('moe', DSV3, this.matmuls, 4096), RECOMPUTE_PRESETS.none, this.transposed)
          : null,
        anaDF: this.hasAttribute('local') && this.getAttribute('lens') === 'param-bytes'
          ? analyze(blockGraph('dense', DSV3, this.matmuls, 4096), RECOMPUTE_PRESETS.none, this.transposed)
          : null,
      };
    }
    const ana = this._anaMemo.ana;
    let barSlot = null;   // the local fit bar renders under the parallelism row
    // the HOUSE knob row (grouped stp segments): one policy/recipe segment
    // plus an always-reserved 'custom' chip, so a hand-edited state lights
    // up without reflowing the row. Shared by the marks/dtype tiers AND the
    // local head (the full sim mirrors the sections' controls exactly).
    // preset segments REMEMBER: 'custom' keeps your last hand-edited
    // state (click it to come back), and clicking the ACTIVE chip toggles
    // back to whatever was selected before it
    this._segMem ??= {};
    const segGrp = (label, name, opts, cur, getState, setPreset, setState, noCustom = false) => {
      const mem = this._segMem[name] ??= {};
      if (cur != null && !opts.includes(cur)) cur = null;   // an uncurated preset reads as custom
      if (cur == null) mem.custom = getState();   // the latest hand-edited composition
      const g = el('span', 'pargrp');
      const l3 = el('div', 'parlab'); l3.textContent = label; g.append(l3);
      const row = el('div', 'parrow');
      const w3 = el('span', 'stp'); w3.dataset.knob = name;
      const pick = (sel, st) => {                 // remember where we came FROM
        mem.prev = { sel: cur ?? 'custom', st: getState() };
        st ? setState(st) : setPreset(sel);
      };
      const back = () => {                        // active chip clicked: swap with the previous pick
        const now = { sel: cur ?? 'custom', st: getState() };
        const pv = mem.prev; mem.prev = now;
        setState(pv.st);
      };
      const arm = (b) => {                        // an active chip with history is a toggle
        if (!mem.prev) return;
        b.onclick = back; b.style.cursor = 'pointer';
        b.title = `click again — back to ${mem.prev.sel}`;
      };
      for (const k of opts) {
        const b = document.createElement('button');
        b.textContent = k; b.type = 'button';
        if (cur === k) { b.classList.add('on'); arm(b); }
        else b.onclick = () => pick(k);
        w3.append(b);
      }
      if (!noCustom) {
        const cb = document.createElement('button');
        cb.textContent = 'custom'; cb.type = 'button';
        if (cur == null) { cb.classList.add('on'); arm(cb); }
        else if (mem.custom) {
          cb.title = 'your last hand-edited state';
          cb.onclick = () => pick('custom', mem.custom);
        } else cb.disabled = true;
        w3.append(cb);
      }
      row.append(w3); g.append(row);
      return g;
    };
    // 'selective' stays a MODEL preset (the trace sim's default; sanity's
    // GB300 anchor pins it) but earns no chip: the story is
    // full / attn-replay / dsv3 / none. noCustom = presets-only (a tier or
    // the local head renders no per-op mark buttons, so a custom policy
    // can't arise there — the AC section is where policies are hand-built).
    // recomputes="none,moe_act,mla_up_proj,…" curates the chips (03 lists Megatron's modules); absent = 02's four
    const polOpts = this.getAttribute('recomputes')?.split(/[ ,]+/).filter((k) => RECOMPUTE_PRESETS[k]) ?? ['full', 'attn-replay', 'dsv3', 'none'];
    const mkPolSeg = (noCustom) => segGrp('recompute policy', 'recompute', polOpts, curPreset,
      () => ({ ...this.marks }),
      (k) => localTween(() => { this.setAttribute('recompute', k); this.marks = { ...RECOMPUTE_PRESETS[k] }; }),
      (st) => localTween(() => { this.marks = { ...st }; }),
      noCustom);
    const mkRecSeg = () => segGrp('precision recipe', 'recipe', recipeOpts, curRecipe,
      () => ({ mm: { ...this.matmuls }, t: !!this.transposed }),   // custom memory carries BOTH channels
      (k) => localTween(() => {
        this.setAttribute('recipe', k);
        this.matmuls = resolveMatmuls({ recipe: k });
        this.transposed = RECIPE_T[k] ?? false;   // recipes are full stash-policy bundles
      }),
      (st) => localTween(() => { this.matmuls = { ...(st.mm ?? st) }; if (st.mm) this.transposed = st.t; }));
    if (cmode === 'marks' || cmode === 'dtype') {
      // these tiers are ALWAYS the detail view (one canonical diagram: every
      // markable op visible — the MLA latent norms carry marks too), so the
      // elided-kernels toggle is gone. The section analyzes the PEAK rank,
      // which is all-MoE: kind pins to moe and the dense/MoE tabs go — the
      // enclosure stays, wearing a static label + the region toggle instead.
      this.detail = true;
      this.kind = 'moe';
      this._noKind = true;
      const hh = el('div', 'lv-head');
      // BOTH quant tiers get the recompute segment, in the same slot. The
      // dtype tier's is PRESETS-ONLY (no per-op mark buttons there).
      // Anatomy-wrapped instances park it in the PLAN column instead of a
      // head row (the AC widget's whole second row disappears — laptops).
      const polSeg = mkPolSeg(!this._ctl.marks);
      let sideWrap = null;
      if (this.closest('dsv3-anatomy')) {
        sideWrap = el('div', 'lv-head lv-side');
        sideWrap.append(polSeg);
      } else hh.append(polSeg);
      if (this._ctl.dtype) hh.append(mkRecSeg());
      if (this._ctl.dtype) hh.append(tl, tl2);
      // ctx: the section's FIXED parallelism as a READOUT row that mirrors
      // the full sim's knob layout exactly (cluster · pipeline · SPMD mesh ·
      // ZeRO, same groups, same places) — locked, so it reads as context,
      // not levers. It takes the whole width; the policy row sits below.
      let hr = null;
      if (this.getAttribute('ctx')) {
        const C = JSON.parse(this.getAttribute('ctx'));   // {world, pp, ep, zero[, sched, vpp, hw]}
        const ILV = C.sched === 'interleaved';
        hr = el('div', 'lv-head');
        const grp3 = (label) => { const g = el('span', 'pargrp'); const l5 = el('div', 'parlab'); l5.textContent = label; g.append(l5); return g; };
        const row3 = (...kids) => { const d = el('div', 'parrow'); d.append(...kids); return d; };
        const txt3 = (t3) => { const sp = el('span'); sp.style.cssText = 'color:var(--c-52514e);font-size:11px;'; sp.textContent = t3; return sp; };
        const seg3 = (name, opts, onIdx) => {
          const w5 = el('span', 'stp'); w5.dataset.knob = name;
          opts.forEach((t3, i) => {
            const b = document.createElement('button');
            b.type = 'button'; b.textContent = t3; b.disabled = true;
            if (i === onIdx) b.classList.add('on');
            w5.append(b);
          });
          return w5;
        };
        const chip3 = (name, v) => seg3(name, [String(v)], 0);
        const DPn2 = C.world / C.pp;
        const gC = grp3('cluster');
        gC.append(row3(txt3('GPUs'), chip3('gpus', C.world), ...(C.hw ? [txt3('·'), chip3('hw', HW_SHORT[C.hw])] : [])));
        const gP = grp3('pipeline');
        // ONE row (PP · rank · sched inline): the readout is context, not
        // levers — laptop vspace beats mirroring the sim's two-row group.
        // Under Megatron's interleaved schedule the rank is a plain number
        // (every rank differs) and VP joins the row.
        gP.append(ILV
          ? row3(txt3('PP'), chip3('pp', C.pp), txt3('VP'), chip3('vpp', C.vpp ?? 1), txt3('rank'),
            chip3('rank', `r${C.stage ?? 0}${C.peak === false ? '' : ' · peak'}`),
            txt3('sched'), seg3('sched', [C.a2a ? 'interleaved 1F1B · a2a overlap' : 'interleaved 1F1B', '×1 mb'], 0))
          : row3(txt3('PP'), chip3('pp', C.pp), txt3('rank'),
            seg3('rank', ['r0 · emb+head', `r1–${C.pp - 1} · peak`], 1),
            txt3('sched'), seg3('sched', ['DualPipeV', '×1 mb'], 0)));
        const gM = grp3('SPMD mesh');
        gM.append(C.tp > 1
          ? row3(txt3('non-expert: TP'), chip3('tp', C.tp), txt3(`× DP ${DPn2 / C.tp}`), txt3('· expert: EP'), chip3('ep', C.ep), txt3(`× EDP ${DPn2 / C.ep}`))
          : row3(txt3(`non-expert: DP ${DPn2}`), txt3('· expert: EP'), chip3('ep', C.ep), txt3(`× EDP ${DPn2 / C.ep}`)));
        const gZ = grp3('ZeRO'); gZ.classList.add('center');
        gZ.append(row3(seg3('zero', ['off', '1', '2', '3'], C.zero)));
        hr.append(gC, gP, gM, gZ);
      }
      reset.style.cssText = 'font:11px ui-monospace,monospace;padding:2px 8px;border:1px solid var(--c-c3c2b7);' +
        'border-radius:4px;background:var(--c-ffffff);cursor:pointer;margin-left:auto;';
      // the marks tier gets NO reset: the preset chips ARE the reset (clicking
      // dsv3 is it). The dtype tier keeps one — it also clears fp8ᵀ.
      if (this._ctl.dtype) hh.append(reset);   // no sizes toggle here: shapes are 01's story, not this section's
      if (hr) root.append(hr);
      if (hh.childElementCount) root.append(hh);   // marks tier with a sided policy: no second row at all
      if (sideWrap) root.append(sideWrap);
    } else if (cmode !== 'static') root.append(head);
    if (cmode === 'static') {
      const mini = el('div', 'lv-head');
      // article instances disclose a SUBSET of the knob groups:
      // knobs="cluster,pipeline,mesh,zero,save,prec,blocks" (absent = all)
      const knAttr = this.getAttribute('knobs') ?? (this.hasAttribute('snapshot') ? '' : null);
      const KN = (k) => knAttr == null || knAttr.split(/[ ,]+/).includes(k);
      // local gets TWO control rows: parallelism (with the fit bar right
      // under it — the headline effect) and a second row for the misc bits
      const mini2 = this.hasAttribute('local') ? el('div', 'lv-head') : mini;
      // param-bytes always shows sizes multiplied out — no toggle to offer
      const sizeCtl = this.getAttribute('lens') === 'param-bytes' ? [] : ['sizes:', mkDimsBtn()];
      // the optim variant is pinned per block; consolidated allows ×N (long!);
      // local is pinned cumulative (the stage total — what the fit bar prices)
      const cumCtl = (this.hasAttribute('optim') && !this.hasAttribute('consolidated')) || LOCALKNOBS
        ? [] : [mkCumBtn()];
      // no kind select when MLA-only (kind-independent) or when the tabs carry the flip
      if (KN('blocks')) {
        if (SCOPE === 'mla' || this.hasAttribute('tabs')) mini2.append(...sizeCtl, ...cumCtl);
        else mini2.append('block: ', mkKindSel(), ...(sizeCtl.length ? [' · '] : []), ...sizeCtl, ...cumCtl);
      }
      if (this.hasAttribute('local')) {
        // the fiat parallelism: cluster size, PP {1,8}, the rank picker,
        // EP width, and the ZeRO level are the knobs. The kind follows the
        // selected rank.
        const pp = this.pp ?? LOCAL_PAR.pp;
        // EP/PP step by powers of two: a segmented − value + control
        const POW2 = [1, 2, 4, 8, 16, 32, 64];
        const mkStep = (get, set, fmt, max = 64, opts = POW2, min = 1) => {
          const wrap = el('span', 'stp');
          // the value chip is itself a dropdown: click the number to jump
          const val = document.createElement('select'); val.className = 'v';
          for (const o of opts.filter((o) => o <= max && o >= min)) val.append(new Option(fmt(o), o));
          val.value = String(get());
          val.onchange = () => { const prev = this._snapLocal(); set(+val.value); this._tweenLocal(prev); };
          // step by OPTION INDEX, not ×2: PP's choices are {1, 8} (for the
          // power-of-two knobs the two are the same thing)
          const btn = (txt, dir, max) => {
            const b = document.createElement('button');
            b.textContent = txt; b.type = 'button';
            const ok = opts.filter((o) => o <= max && o >= min);
            const j = ok.indexOf(get()) + dir;
            b.disabled = j < 0 || j >= ok.length;
            if (!b.disabled) b.onclick = () => {
              const prev = this._snapLocal();
              set(ok[j]);
              this._tweenLocal(prev);
            };
            return b;
          };
          wrap.append(btn('−', -1, max), val, btn('+', 1, max));
          return wrap;
        };
        // EP is a subgroup of DP, so EP ≤ DP = world/PP (raising PP or
        // shrinking the cluster clamps EP)
        const world = this.world ?? LOCAL_PAR.world;
        const epMax = Math.min(64, world / (this.pp ?? LOCAL_PAR.pp));
        // grouped by MESH STRUCTURE: the cluster, the pipeline split (with its
        // stage/schedule — independent of the SPMD mesh), and the SPMD mesh
        // itself as two rows (non-expert = plain DP; expert = EP × derived
        // EDP). ZeRO gets its own spanning group: it applies universally.
        const grp2 = (label) => {
          const g = el('span', 'pargrp');
          const l2 = el('div', 'parlab'); l2.textContent = label;
          g.append(l2);
          return g;
        };
        const row2 = (...kids) => { const d = el('div', 'parrow'); d.append(...kids); return d; };
        const txt2 = (t3) => { const sp = el('span'); sp.style.cssText = 'color:var(--c-52514e);font-size:11px;'; sp.textContent = t3; return sp; };
        const knob = (n, e2) => { e2.dataset.knob = n; return e2; };
        const gCluster = grp2('cluster');
        gCluster.append(row2(txt2('GPUs'), knob('gpus', mkStep(() => world,
          (v) => { this.world = v; this.ep = Math.min(this.ep, v / this.pp); },
          String, 16384, [128, 256, 512, 1024, 2048, 4096, 8192, 16384], 128))));
        const seg2 = (opts, get, set) => {
          const w2 = el('span', 'stp');
          for (const [k, lab] of opts) {
            const b = document.createElement('button');
            b.textContent = lab; b.type = 'button';
            if (get() === k) b.classList.add('on');
            else b.onclick = () => { const prev = this._snapLocal(); set(k); this._tweenLocal(prev); };
            w2.append(b);
          }
          return w2;
        };
        const gPipe = grp2('pipeline');
        // the RANK picker collapsed to a two-way segment: under the slot
        // split every interior rank (1…pp−1) holds the same 8 MoE layers —
        // only rank 0 differs (emb + head + the dense front). 'rank', not
        // 'stage': DualPipeV's stages are the 2·pp chunks.
        const pkR = peakStage(pp, this.ep, this.zero ?? 1, world, this.sched, this.vpp ?? 1, this.fold, this.layout);
        // the Megatron family (wrap fold): PP over {1,2,4,8}, a VP stepper,
        // and a plain rank stepper — under a layout every rank differs
        const MEG = this.fold === 'wrap';
        const rankSeg = pp === 1 ? null : MEG
          ? knob('rank', mkStep(() => this.stage, (v) => { this.stage = v; },
            (v) => `r${v}${v === pkR ? ' · peak' : ''}`, pp - 1, Array.from({ length: pp }, (_, i) => i), 0))
          : knob('rank', seg2(
            [[0, `r0 · emb+head${pkR === 0 ? ' · peak' : ''}`], [1, `r1–${pp - 1}${pkR !== 0 ? ' · peak' : ''}`]],
            () => this.stage === 0 ? 0 : 1,
            (k) => { this.stage = k === 0 ? 0 : pkR || 1; }));
        gPipe.append(
          row2(txt2('PP'), knob('pp', mkStep(() => this.pp, (v) => this._setPP(v), String, 64, MEG ? [1, 2, 4, 8, 16] : PP_CHOICES)),
            ...(MEG && pp > 1 ? [txt2('VP'), knob('vpp', mkStep(() => this.vpp, (v) => this._setVPP(v), String, 64 / pp, [1, 2, 4, 8, 16]))] : []),
            ...(rankSeg ? [txt2('rank'), rankSeg] : [])));
        const gMesh = grp2('SPMD mesh');
        const txtR = (t3) => {   // right-aligned row labels, so the mesh rows line up
          const sp = txt2(t3);
          sp.style.cssText += 'display:inline-block;width:64px;text-align:right;';
          return sp;
        };
        const TPS = this.getAttribute('tps')?.split(/[ ,]+/).map(Number).filter((v) => v >= 1);
        gMesh.append(
          row2(txtR('non-expert:'), ...(TPS?.length > 1
            ? [txt2('TP'), knob('tp', mkStep(() => this.tp ?? 1, (v) => this._setTP(v), String, Math.max(...TPS), TPS)), txt2(`× DP ${world / pp / (this.tp ?? 1)}`)]
            : [txt2(`DP ${world / pp / (this.tp ?? 1)}`)])),
          row2(txtR('expert:'), txt2('EP'), knob('ep', mkStep(() => this.ep, (v) => { this.ep = v; }, String, epMax)),
            txt2(`× EDP ${world / pp / this.ep}`)));
        // ZeRO-(off|1|2|3): a segmented level picker (1 shards optimizer,
        // 2 + gradients, 3 + weights — each over its replication group)
        const zw = el('span', 'stp'); zw.dataset.knob = 'zero';
        for (const [i, lv] of [[0, 'off'], [1, '1'], [2, '2'], [3, '3']]) {
          const b = document.createElement('button');
          b.textContent = lv; b.type = 'button';
          if ((this.zero ?? 1) === i) b.classList.add('on');
          b.onclick = () => {
            if ((this.zero ?? 1) === i) return;
            const prev = this._snapLocal(); this.zero = i; this._tweenLocal(prev);
          };
          zw.append(b);
        }
        // admission is the one schedule knob left: DualPipeV steady state
        // (uniform PP+½ in flight, emb+head on rank 0) vs a single
        // microbatch. VPP/fold are derived — the schedule IS DualPipeV.
        const sw2 = knob('sched', seg2(MEG ? [['interleaved', 'interleaved 1F1B'], ['one', '×1 mb']] : [['1f1b', 'DualPipeV'], ['one', '×1 mb']],
          () => this.sched ?? '1f1b', (k) => { this.sched = k; }));
        gPipe.append(row2(txt2('sched'), sw2));
        // hws="gb200,gb300": the capacity yardstick becomes a knob (the
        // Blackwell post's GB200-vs-GB300 question); absent = fixed
        const hws = this.getAttribute('hws')?.split(/[ ,]+/).filter((k) => HARDWARE[k]);
        if (hws?.length > 1) gCluster.append(row2(txt2('GPU'), knob('hw', seg2(hws.map((k) => [k, `${HW_SHORT[k]} · ${HARDWARE[k].memGB} GiB`]),
          () => this.hw ?? 'h100', (k) => { this.hw = k; }))));
        const gZ = grp2('ZeRO');
        gZ.classList.add('center');   // spans the mesh rows, like PP: it applies universally
        gZ.append(row2(zw));
        for (const [k2, g3] of [['cluster', gCluster], ['pipeline', gPipe], ['mesh', gMesh], ['zero', gZ]])
          if (KN(k2)) mini.append(g3);
      }
      if (this.getAttribute('lens') === 'param-bytes') {
        // the strip unit rescales with the ×N toggle — label it so the jump
        // reads as a unit change, not a glitch (▫ = nonzero but sub-square)
        const KM2 = this.kind === 'dense' ? (DSV3.denseLayers ?? 3) : DSV3.layers - (DSV3.denseLayers ?? 3);
        // fixed unit whenever the strips GROW instead of renormalizing
        // (strips=absolute and the local variant)
        const absP = this.getAttribute('strips') === 'absolute' || this.hasAttribute('local');
        const unit = PARAMS.largestOp.moe * (this.cumulative && !absP ? KM2 : 1) / 32 * 2;
        const leg = el('span');
        leg.style.cssText = 'color:var(--c-52514e);margin-left:10px;font-size:11px;white-space:nowrap;';
        // the swatch is a real 5×4 rect — the same size as the strip squares
        // (a text ▪ renders at whatever the font says)
        // inline-block + zero margin: the .lv svg{display:block;margin:0 auto}
        // rule for the main diagram would otherwise stack the swatch on its own line
        const sw = (c) => `<svg width="5" height="4" style="display:inline-block;margin:0;vertical-align:baseline"><rect width="5" height="4" fill="${c}"/></svg>`;
        const cons2 = this.hasAttribute('consolidated') || this.hasAttribute('local');
        if (this.hasAttribute('marginlegend')) { /* the anatomy margin hosts the legend */ }
        else if (this.hasAttribute('optim') || cons2) {
          // the legend entries ARE the visibility toggles: the squares pour
          // in/out (per-block-tween style, boxes reflow with the filled rows)
          // and the numbers snap to exactly what's shown
          const cb = (label, color, prop) => {
            const lab = document.createElement('label');
            lab.style.cssText = 'display:inline-flex;align-items:center;gap:3px;margin-right:10px;cursor:pointer;';
            const c = document.createElement('input');
            c.type = 'checkbox'; c.checked = this[prop];
            c.onchange = () => this.toggleComp(prop, c.checked);
            const t = el('span');
            t.style.cssText = 'display:inline-flex;align-items:center;gap:3px;';   // swatch centers regardless of baseline
            t.innerHTML = `${sw(color)} <span>${label}</span>`;
            lab.append(c, t);
            return lab;
          };
          const comps2 = cons2 ? BYTE_COMPS : [BYTE_COMPS[0], BYTE_COMPS[2]];
          leg.append(...comps2.map((c) => cb(compLabel(c, this.gradB), C(c.color), c.prop)));
          if (cons2) leg.append(cb(this.hasAttribute('local')
            ? `saved activations (bf16, ×${TOK} tok × ${fmtIF(inflightOf(this.sched ?? '1f1b', this.stage ?? 1, this.pp ?? LOCAL_PAR.pp, this.vpp, this.fold, this.layout, this.kind))} in flight)`
            : `saved activations (bf16, ×${TOK} tokens)`, C('#eda100'), 'showActs'));
          const u = el('span');
          u.style.cssText = 'display:inline-flex;align-items:center;gap:3px;';   // swatch centers regardless of baseline
          u.innerHTML = `· ${sw(C('#898781'))} <span>= ${fmtBytes(unit)}</span>`;
          leg.append(u);
        } else {
          leg.style.cssText += 'display:inline-flex;align-items:center;gap:3px;';
          leg.innerHTML = `${sw(C('#2a78d6'))} <span>= ${fmtBytes(unit)}</span>`;
        }
        if (KN('legend')) mini2.append(leg);
      }
      if (this.hasAttribute('local')) {
        // pin a baseline config: the log bars then carry ticks at the pinned
        // values and ×N/÷N factors — "I ×256'ed this and it /256'ed that"
        const mkBtn = (txt, title, fn) => {
          const b = document.createElement('button');
          b.style.cssText = 'font:11px ui-monospace,monospace;padding:2px 8px;border:1px solid var(--c-c3c2b7);' +
            'border-radius:4px;background:var(--c-ffffff);cursor:pointer;';
          b.textContent = txt; b.title = title; b.onclick = fn;
          return b;
        };
        // SAVE semantics, top right and visually apart: save locks in the
        // current config (deltas display vs the save; re-save to lock in a
        // change), reset returns TO the save, reset-all is factory
        const saveBox = el('span', 'savebox');
        saveBox.append(mkBtn('save', 'lock in this config: the chart shows deltas vs the save (re-save to lock in a change)', () => {
          this._saveBaseline();
          this.render();
        }));
        // always present (disabled until a save exists) so saving never reflows
        const rst = mkBtn('reset', 'return to the saved config', () => {
          const prev = this._snapLocal();
          Object.assign(this, this._pinCfg.state,
            { marks: { ...this._pinCfg.state.marks }, matmuls: { ...this._pinCfg.state.matmuls } });
          this._tweenLocal(prev);
        });
        if (!this._pinCfg?.state) { rst.disabled = true; rst.title = 'save a config first'; rst.style.color = C('#c3c2b7'); rst.style.cursor = 'default'; }
        saveBox.append(rst, reset);   // factory reset (built above; also clears the save)
        reset.textContent = 'reset all';
        reset.style.cssText = 'font:11px ui-monospace,monospace;padding:2px 8px;border:1px solid var(--c-c3c2b7);' +
          'border-radius:4px;background:var(--c-ffffff);cursor:pointer;';   // match the save cluster's face
        if (KN('save')) mini.append(saveBox);
        // the AC + precision knobs, wearing the SAME house segments as the
        // AC and low-precision sections (recipe chips with recognition +
        // both stash-format checkboxes) — the full sim mirrors the sections
        if (KN('prec')) {
          // AMAIA-style fp8-resident PARAMETERS (distinct from tl, which duals
          // the activation stashes): weights = the fp8 copies themselves,
          // both orientations, at 1×128-scale cost
          const tl3 = document.createElement('label');
          tl3.style.cssText = 'display:inline-flex;align-items:center;gap:4px;margin-left:8px;color:var(--c-52514e);';
          const p8 = document.createElement('input');
          p8.type = 'checkbox'; p8.checked = !!this.fp8Params; p8.dataset.knob = 'fp8params';
          p8.disabled = this.matmuls.ffn_gate_up === 'bf16';
          tl3.title = p8.disabled
            ? 'pick an fp8 recipe first — bf16 compute keeps bf16 working weights'
            : 'fp8-resident parameters: the working weights ARE the fp8 copies, kept in BOTH orientations (Hopper layout law) ' +
              'with 1×128-tile scales: 2 × (1 + 1/32) = 2.0625 B/param on the block params. Embedding/lm head stay bf16; ' +
              'the fp32 master lives in the optimizer bar; norms/router (≈0.1% of block params) are booked fp8 too — noise. ' +
              'OFF = bf16 working weights (2 B/param), quantized on the fly.';
          p8.onchange = () => this.setLocal(() => { this.fp8Params = p8.checked; });
          // fp8-RESIDENT parameters: Hopper keeps e4m3 + its transpose; Blackwell's
          // MXFP8 keeps row- and column-wise copies (TE's post-all-gather
          // processing) — the same 2 × (1 + 1/32) B/param either way
          tl3.append(p8, Object.values(this.matmuls).includes('mxfp8') ? 'mxfp8 params (row + col)' : 'e4m3+ᵀ params');
          // recompute is presets-only here (no per-op mark buttons on the
          // local diagram); the recipe segment keeps its custom chip — the
          // stash checkboxes can compose states no recipe names
          mini2.append(mkRecSeg(), mkPolSeg(true), tl, tl2, tl3);
        }
      }
      // snapshot knobs are READOUTS — unless the host opts into 'live'
      // (the beat deck: fiddling is a marked DETOUR, rewound on step)
      if (this.hasAttribute('snapshot') && !this.hasAttribute('live'))
        for (const c2 of [...mini.querySelectorAll('button, select'), ...mini2.querySelectorAll('button, select')])
          c2.disabled = true;
      root.append(mini);
      if (this.hasAttribute('local')) { barSlot = el('div', 'lv-bar'); root.append(barSlot); }
      if (mini2 !== mini && mini2.childNodes.length) root.append(mini2);
    }
    // dense mode also analyzes the MoE graph, purely for LAYOUT: the dense
    // column reserves whitespace where the routing rows sit, so flipping
    // kinds keeps every surviving element in the same place
    const anaM = this._anaMemo.anaM;
    // never scale the diagram: it renders at natural size inside its own
    // scroll container (tooltips stay outside it, so they aren't clipped)
    const scroller = el('div', 'lv-scroll');
    const diagSvg = this.buildSvg(ana, anaM);
    if (diagSvg) scroller.append(diagSvg);
    if (barSlot && this._barHtml) {
      barSlot.innerHTML = this._barHtml;
      this._barHtml = null;
      this._wireBars(barSlot);
      if (this.hasAttribute('snapshot') && this.hasAttribute('hypothetical') && !this.getAttribute('knobs')) {
        const ht = el('div', 'lv-hyptag');
        ht.textContent = this.getAttribute('hypothetical') || 'hypothetical — not what DSv3 did';
        barSlot.before(ht);
      }
      const sb = this.hasAttribute('snapshot') && this.getAttribute('sandbox');
      if (sb) {
        const a = document.createElement('a');
        a.textContent = 'play with this scenario in the full widget ↗';
        a.href = '#'; a.style.cssText = 'font-size:11.5px;';
        // resolve the target lazily: anatomy-wrapped layers don't exist yet
        // when snapshots upgrade (bare layers upgrade first)
        a.onclick = (ev) => {
          ev.preventDefault();
          document.getElementById(sb)?._loadScenario?.(this._snapCfg.from, this._snapCfg.to);
        };
        const nd = el('div', 'lv-note'); nd.append(a);
        barSlot.after(nd);
      }
    }
    if (!this.hasAttribute('barsonly') && !this.hasAttribute('snapshot')) root.append(scroller);
    const note = el('div', 'lv-note');
    const M2 = this.view === 'combined' ? this.dispLayers * this.dispInflight * TOK : 1;
    const parts = [
      !this._ctl.quant ? '' :
      (this.view === 'combined'
        ? `stashed for backward: ${(ana.savedBytes * M2 / 2 ** 30).toFixed(1)} GiB total = ${(ana.savedBytes / 1024).toFixed(0)} KiB/token\u00b7layer \u00d7 ${this.dispLayers} layers \u00d7 ${this.dispInflight} in-flight \u00d7 ${TOK} tokens (set layers/in-flight to your PP stage to tally with the memory bars) \u00b7 `
        : `stashed for backward: ${(ana.savedBytes / 1024).toFixed(0)} KiB/token\u00b7layer \u00b7 `) +
      `backward replays +${(ana.replayFrac * 100).toFixed(0)}% of fwd FLOPs` +
      (ana.replayComm.length ? ` + a2a ${ana.replayComm.join('+')}` : '') + '.',
      this._ctl.marks
        ? 'The \ud83d\udcbe/\u21bb button on each op chooses save-output vs recompute-in-backward; the wire below shows the derived result \u2014'
        : this._ctl.quant
          ? 'Each wire label is an output, tagged with the recompute policy\u2019s derived result \u2014'
          : 'Each wire label is an output, tagged with whether backward reads it \u2014',
      this._ctl.quant
        ? '\u2193 \u2191 \u21c5 saved for backward, read by the op below / above / both; \u21d3 \u21d1 \u21d5 read by a RECOMPUTE (replay) instead \u2014 both families appear when a tensor serves both roles ' +
          '(\u25aa = 4 KiB/token; violet boxes = communication), ' +
          '\u21bb recomputed, \u00b7 not needed, \ud83d\udd12 always saved; ' +
          'right arrows are aux backward artifacts (rstd, lse), \u2190 saved unless their op replays.'
        : '\u2193 \u2191 \u21c5 read by the op below / above / both, \u00b7 not needed (violet boxes = communication); ' +
          'grey sizes are per-token element counts \u2014 bytes need a dtype, which comes later; ' +
          'right arrows are aux backward artifacts (rstd, lse).',
      this._ctl.marks ? 'Marking an op \u21bb forces the outputs it reads to stay saved.' : '',
      this.detail ? 'Small italic boxes are kernels the terse view folds away \u2014 cheap vector/permute ops with no marks of their own (negligible FLOPs and bytes).' : '',
      (this.kind === 'dense'
        ? ''   // no shared expert in a dense block
        : this.detail
          ? (this._ctl.quant ? 'The shared expert follows the grouped ffn boxes\u2019 mark and dtype; its FLOPs are counted in their strips. ' : '')
          : (this._ctl.quant
            ? `Shared expert${SCOPE === 'model' ? ' + dense MLPs' : ''} follow${SCOPE === 'model' ? '' : 's'} the ffn choices; `
            : `The shared expert${SCOPE === 'model' ? ' and dense MLPs share' : ' shares'} the ffn boxes; `))
          + `RoPE carries its own mark${this._ctl.quant ? ' (every preset replays it; saving it stashes the same bytes post-rotation)' : ''}.`,
      !this._ctl.quant ? '' :
      `The picket run inside each op is its compute TIME at ${HW_SHORT[JSON.parse(this.getAttribute('ctx') ?? '{}').hw ?? this.hw ?? 'h100']} peak \u2014 one picket \u2248 ${Math.round(10e6 * 4096 / HARDWARE[JSON.parse(this.getAttribute('ctx') ?? '{}').hw ?? this.hw ?? 'h100'].flops.bf16 * 1e6)} \u00b5s per 4096-token microbatch. ` +
      'A picket packs 10 MFLOP/token at bf16\u2019s 989 TFLOP/s; e4m3/mxfp8 run 2\u00d7 (1979), so theirs pack 20; ' +
      'the fp32 router runs on CUDA cores at 67 TFLOP/s (TF32 would truncate the mantissa the pin exists to keep) \u2014 \u224815\u00d7 bf16 time per FLOP. ' +
      'Dtype colors here and on the saved-tensor tags: pink e4m3, purple e5m6, dark bf16, brick fp32. ' +
      'The lm head uses the same unit \u2014 per-token vocab work, independent of depth. Norms/SwiGLU ' +
      'get a hollow dashed fig-leaf (bandwidth-bound, compute precision unspecified).',
      this._ctl.dtype ? 'One click on a dtype button toggles bf16 \u21c4 e4m3 (attn-out and the router are pinned \ud83d\udd12; the E5M6 checkbox picks the attn-out save format).' : '',
      this._ctl.quant
        ? 'The tally at right totals fwd + bwd (2\u00d7 fwd \u2014 dgrad + wgrad; sdpa likewise) + replay'
          + (this._ctl.marks ? ' \u2014 marking ops \u21bb grows its replay row.' : '.')
        : '',
      this._ctl.dtype
        ? 'The e4m3\u1d40 toggle: wgrads read saved activations TRANSPOSED and 1\u00d7128 tile scales don\u2019t survive that \u2014 ' +
          'ON stashes both orientations (\u1d40\u00d72 tags); OFF re-quantizes during backward (DeepSeek\u2019s convention, priced on the bwd ribbon).'
        : '',
    ];
    note.textContent = parts.filter(Boolean).join(' ');
    // nocaption: the page explains the diagram in its own prose
    if (!this.hasAttribute('nocaption')) {
      const foot = el('div', 'lv-foot2');
      // marks/dtype tiers drop the prose caption (the page's own text explains
      // the diagram) — the foot is just the full-width tally + stash readout
      if (cmode !== 'marks' && cmode !== 'dtype') foot.append(note);
      if (cmode !== 'static') foot.append(this._tallySvg);
      root.append(foot);
    }
    // tooltips everywhere except the structure-only tier — the local sim
    // gets the full set (⚠/⇄ badges, box FLOP/param facts, raw-bytes lens)
    if (this._ctl.quant || LOCALKNOBS) this.attachTip(root);
    this.append(style, root);
    this.applyHl();
  }
  // spreadsheet-style highlighting: mark the boxes whose parameters a
  // clicked tally row sums (dsv3-param-tally drives this)
  // ids = null clears; ids = [] fades EVERYTHING (a selected sum with no
  // cells in this diagram — e.g. the embedding row greys the block out)
  highlightOps(ids) { this._hl = ids ? new Set(ids) : null; this.applyHl(); }
  applyHl() { applyHighlight(this, this._hl); }
  // this GPU's token share of a microbatch (see render): 4096 / TP
  _tok() {
    return 4096 / (this.hasAttribute('local') && this.getAttribute('lens') === 'param-bytes'
      ? (this.tp ?? 1) : (JSON.parse(this.getAttribute('ctx') ?? '{}').tp ?? 1));
  }
  buildSvg(ana, anaM = null) {
    const TOK = this._tok();
    const P = [];
    // Two columns (MLA | MoE), head row underneath. The dataflow spine runs
    // down the LEFT of each column; output tensors are annotated on the spine
    // (▣ saved + block grid, ▪ = 4 KiB/token · ↻ recomputed · · not needed).
    // Aux backward artifacts (rstd, lse) exit each box to the RIGHT — always saved.
    // only="mla" / only="ffn" draws a single column (for composed anatomy
    // pages that show each component once); default draws the full block
    const SCOPE = this.getAttribute('scope') ?? 'model';
    const ONLY = SCOPE === 'mla' || SCOPE === 'ffn' ? SCOPE : null;   // single-column scopes
    const LENS = this.getAttribute('lens');
    const PBYTES = LENS === 'param-bytes';   // parameter MEMORY at bf16 (2 B/param)
    const PONLY = LENS === 'params' || PBYTES;   // parameter focus: intermediates/dims/aux hidden
    // byte components stacked under each op, colored to match the memory-bars
    // segments, ONE global unit — the ratios ARE the picture. optim = weights +
    // optimizer states; consolidated = + fp32 gradients + a saved-activations
    // band. optim is pinned per block; consolidated allows ×N (with
    // strips="absolute" the squares grow — a very long diagram). Legend checkboxes
    // toggle components: the squares pour in/out per-block-tween style and the
    // numbers always total exactly the squares shown.
    // local: what ONE GPU holds under the fiat parallelism (LOCAL_PAR + the
    // EP/stage selectors). Expert weights divide by EP; optimizer states are
    // ZeRO-1-sharded over each parameter's replication group (dense params
    // /DP, expert params /expert-DP = world/pp/EP); the kind and block
    // multiplier follow the selected PP stage. Implies consolidated.
    const LOCAL = PBYTES && this.hasAttribute('local');
    const CONS = LOCAL || (PBYTES && this.hasAttribute('consolidated'));
    const OPTIM = CONS || (PBYTES && this.hasAttribute('optim'));
    // nostrips (PROTOTYPE): drop the in-box parameter squares entirely — at
    // PP1/EP1 they are enormous and the message is just "this is big" (the
    // log chart carries it); countable squares earn their keep only once
    // things fit. Amber activation chips stay (they ARE the AC feedback).
    const NOSTRIPS = this.hasAttribute('nostrips');
    const EPn = this.ep ?? 64, PPn = this.pp ?? LOCAL_PAR.pp, STG = this.stage ?? 1;
    const ZL = this.zero ?? 1;                               // ZeRO level (0 off · 1 optim · 2 +grads · 3 +weights)
    const WORLD = this.world ?? LOCAL_PAR.world;
    const TPn = this.tp ?? 1;
    const DPn = WORLD / PPn / TPn;                           // non-expert data parallelism (TP shards those params)
    const EDP = WORLD / PPn / EPn;                           // expert-DP (EP=1: = DP — no expert parallelism); TP widens it: expert-TP is 1
    // a parameter class's per-GPU share under TP: sharded ('d' block ops, 'v' vocab) 1/TP; replicated ('r': norms, router,
    // MLA down-projections) and experts ('e': EP's business) whole
    const tpfOf = (cls, tp) => cls === 'd' || cls === 'v' ? 1 / tp : 1;
    const SCHED = this.sched ?? '1f1b';
    const VPPn = this.vpp ?? 1, FOLD = this.fold ?? 'reflect', LAYOUT = this.layout ?? null, HWk = this.hw ?? 'h100';
    const stg = ppStage(STG, PPn, VPPn, FOLD, LAYOUT);
    // knob tween (local): squares pour between the OLD and NEW configuration —
    // each component's effective factor (bytes/param × EP share × stage ×N)
    // lerps with this._vtween.t; numbers snap to the new config
    const A2A = !!this.a2a, GRADB = this.gradB ?? 4, MX = Object.values(this.matmuls).includes('mxfp8');
    const IFN = inflightOf(SCHED, STG, PPn, VPPn, FOLD, LAYOUT, this.kind, { a2a: A2A });     // microbatches in flight on this stage (this kind's layers)
    const Snow = { ep: EPn, pp: PPn, stage: STG, zero: ZL, world: WORLD, sched: SCHED, vpp: VPPn, fold: FOLD, layout: LAYOUT, hw: HWk,
      a2a: A2A, gradB: GRADB, mx: MX, tp: TPn, cum: !!this.cumulative,
      fp8p: !!this.fp8Params && this.matmuls.ffn_gate_up !== 'bf16' };
    const dLoc = (S) => {
      const g = ppStage(Math.min(S.stage, S.pp - 1), S.pp, S.vpp, S.fold, S.layout);
      const kmul = this.kind === 'dense' ? g.dense : g.moe;
      const tp = S.tp ?? 1, dp = (S.world ?? LOCAL_PAR.world) / S.pp / tp, edp = dp * tp / S.ep;
      // the gradient buffer's bytes/param is a config fact (fp32 in 02, bf16 in Megatron's perf recipes)
      const bppOf = (c) => c.prop === 'showGrads' ? (S.gradB ?? 4) : c.bpp;
      return { mult: S.cum ? kmul : 1, eFrac: 1 / S.ep,
        acts: (S.cum ? kmul : 1) * inflightOf(S.sched ?? '1f1b', S.stage, S.pp, S.vpp, S.fold, S.layout, this.kind, { a2a: !!S.a2a }) / tp,
        bpp: (c, cls) => tpfOf(cls, tp) * ((S.zero ?? 1) >= c.zthresh ? bppOf(c) / (cls === 'e' ? edp : dp) : bppOf(c)) };
    };
    const W8 = (S, c, cls) => c.prop === 'showWeights' && cls !== 'v' && S.fp8p ? 2.0625 / 2 : 1;
    const fEff = (c, cls, S) => {
      const d = dLoc(S);
      return d.bpp(c, cls) * d.mult * (cls === 'e' ? d.eFrac : 1) * W8(S, c, cls);
    };
    const fT = (c, cls) => {
      const fN = fEff(c, cls, Snow), V = this._vtween;
      return V ? fEff(c, cls, V.prev) + (fN - fEff(c, cls, V.prev)) * V.t : fN;
    };
    // activations multiplier (stage blocks × microbatches in flight), lerped
    const actsT = (() => {
      const aN = dLoc(Snow).acts, V = this._vtween;
      return V ? dLoc(V.prev).acts + (aN - dLoc(V.prev).acts) * V.t : aN;
    })();
    // CHANGE factors (vs a pinned baseline) wear an alert style: direction is
    // the arrow (magnitude always ≥ 1 — cleaner scan than ×1/N), grew = red
    // ▲ (memory alarm; red is unused by the components), shrank = bold ▼
    const facBadge = (cur, base) => {
      if (!base || !cur) return '';
      const r = cur / base;
      if (Math.abs(Math.log2(r)) < 0.05) return '';
      return r > 1
        ? `<tspan fill="${C('#d03b3b')}" font-weight="600"> ▲×${facNum(r)}</tspan>`
        : `<tspan fill="${C('#0b0b0b')}" font-weight="600"> ▼×${facNum(1 / r)}</tspan>`;
    };
    const facTxt = (cls) => {
      const pin2 = this._pinCfg;
      if (!LOCAL || !pin2?.scalars) return '';
      return facBadge(this._scalars[cls], pin2.scalars[cls]);
    };
    const COMPS = !OPTIM ? [BYTE_COMPS[0]]
      : CONS ? BYTE_COMPS : [BYTE_COMPS[0], BYTE_COMPS[2]];
    // tween multiplier for a component: 1 shown, 0 hidden, in between mid-pour
    const cmult = (prop) => this._ctween?.props?.has(prop)
      ? (this[prop] ? this._ctween.t : 1 - this._ctween.t)
      : (this[prop] ? 1 : 0);
    // sub-part filter visibility (experts / non-expert / emb+lm head), lerped
    const psel = (state2, k) => state2 == null || state2 === k ? 1 : 0;
    const pvis = (k) => this._ptween
      ? psel(this._ptween.prev, k) + (psel(this.partSel ?? null, k) - psel(this._ptween.prev, k)) * this._ptween.t
      : psel(this.partSel ?? null, k);
    // an op's part: expert-class ops are the experts part; vocab never
    // appears in block scope (the plan carries it); the rest are non-expert
    const partOfCls = (cls) => cls === 'e' ? 0 : 1;
    // visible bytes per parameter (numbers snap). Two classes under local:
    // 'd' (dense/replicated: optimizer ZeRO-1-sharded /DP) and 'e' (expert:
    // sharded over the smaller expert-DP group)
    const bppOf = (c, cls) => (LOCAL ? tpfOf(cls, TPn) : 1) * (!LOCAL || ZL < c.zthresh ? c.bpp
      : c.bpp / (cls === 'e' ? EDP : DPn));
    const BPPT = (cls = 'd') => COMPS.reduce((t, c) => t + (this[c.prop] ? bppOf(c, cls) : 0), 0);
    const clsOf = (id) => LOCAL && this.kind === 'moe' && (id === 'ffn_gate_up' || id === 'ffn_down') ? 'e'
      : LOCAL && TP_REPLICATED.includes(id) ? 'r' : 'd';
    // param-bytes always shows sizes multiplied out (factored ×256 byte chains
    // pull no weight there; the sizes toggle is hidden in that lens)
    const FLAT = this.flatDims || PBYTES;
    // static/params tiers can never fill the in-box strip band (FLOP strips
    // need a dtype tier, param strips need the bytes lens) — compact boxes
    // instead of reserving space for strips that can't appear
    const BQ = this._ctl.quant || PBYTES;
    // quant tiers reserve TWO picket rows per box (fwd + recompute) so the
    // recompute row pouring in/out never reflows
    const BH = this._ctl.quant ? 45 : BQ ? 38 : 32;      // bold matmul box height
    const HBH = this._ctl.quant ? 67 : BQ ? 60 : 32;     // narrow half-column box (buttons + strip sit below the text)
    // quant tiers carry byte-quantity labels (e.g. attention's lse) that need
    // more room between the columns; the static tier keeps its published width
    const W = 290, C1 = 60,
      // quant+detail tightened to 540: the widened MLA ends at ~480, and the
      // laptop column is ~1000px — every spare hpx counts
      C2 = (ONLY === 'ffn' ? 60 : !this._ctl.quant ? 512 : this.detail ? 540 : 524)
        + (ONLY !== 'ffn' && this._ctl.quant && this.hasAttribute('tabs') ? 20 : 0);
    const SX1 = C1 + 22, SX2 = C2 + 22, RAIL1 = C1 - 26;
    const WIDTH = ONLY === 'mla' ? C1 + W + 250
      : C2 + W + (this.detail ? (this._ctl.quant ? 232 : 224) : 180); // right margin fits aux labels (+ shared column in detail; quant byte tags are wider)
    // dims display: factored (128\u00d7192) or multiplied out (24576)
    const flatten = (s) => {
      if (!FLAT || !s) return s;
      return String(s).split('\u2192').map(part => {
        const t = part.trim().replace(/\u00d7/g, '*');
        if (!/^[\d\s+*()]+$/.test(t)) return part.trim();
        try { return String(Function('"use strict";return (' + t + ')')()); } catch { return part.trim(); }
      }).join(' \u2192 ');
    };
    const PCNT = {
      qkv_down: PARAMS.qkvDown,
      q_up: DSV3.qRank * DSV3.heads * (DSV3.qkNope + DSV3.qkRope),
      kv_up: DSV3.kvRank * DSV3.heads * (DSV3.qkNope + DSV3.vHead),
      o_proj: DSV3.heads * DSV3.vHead * DSV3.hidden,
      router: PARAMS.routerWeight,
      ...(this.kind === 'dense' ? {
        ffn_gate_up: DSV3.hidden * 2 * DSV3.denseInter,
        ffn_down: DSV3.denseInter * DSV3.hidden,
      } : {
        // active view: only the fired experts count (top-k; the shared expert
        // has its own boxes)
        // local: this rank hosts 256/EP of the routed experts
        ffn_gate_up: [DSV3.hidden * 2 * DSV3.moeInter,
          LOCAL ? DSV3.routedExperts / EPn : this.activeView ? DSV3.topk : DSV3.routedExperts],
        ffn_down: [DSV3.moeInter * DSV3.hidden,
          LOCAL ? DSV3.routedExperts / EPn : this.activeView ? DSV3.topk : DSV3.routedExperts],
      }),
      lm_head: DSV3.hidden * DSV3.vocab,
    };
    const exactParam = (id) => {
      if (id === 'router_bias') return PARAMS.routerBias;
      if (id === 'norm1' || id === 'norm2') return DSV3.hidden;
      if (id === 'q_norm') return DSV3.qRank;
      if (id === 'kv_norm') return DSV3.kvRank;
      const p = PCNT[id];
      return Array.isArray(p) ? p[0] * p[1] : p;
    };
    // cumulative: block params carry ×K (the selected kind's block count);
    // the sizes toggle collapses the whole product. The lm head is not a
    // block parameter and never multiplies.
    // local: the multiplier is the selected PP stage's block count of the
    // shown kind; cumulative toggles between one block and the stage total
    const KMUL = LOCAL ? (this.kind === 'dense' ? stg.dense : stg.moe)
      : this.kind === 'dense' ? (DSV3.denseLayers ?? 3) : DSV3.layers - (DSV3.denseLayers ?? 3);
    const CUM = !!this.cumulative;
    // per-class scale scalars: every number of a sharding class moves by the
    // SAME factor under a knob change, so a pinned baseline can annotate each
    // box with an exact ×N/÷N (visible components only — numbers match squares)
    if (LOCAL) this._scalars = {
      d: (CUM ? KMUL : 1) * BPPT('d'),
      r: (CUM ? KMUL : 1) * BPPT('r'),
      e: (CUM ? KMUL : 1) * BPPT('e') / EPn,
      a: (CUM ? KMUL : 1) * IFN,
    };
    // cumulative is always shown multiplied out — factored ×256 ×58 chains
    // are noise; the sizes toggle keeps governing dims and per-block factoring
    // param-bytes lens: the VISIBLE bytes per parameter (bf16 weights = 2 B,
    // + 8 B optimizer when shown), formatted as binary bytes — the number on a
    // box always totals exactly the squares drawn in it
    const fmtPB = (nParams, cls) => fmtBytes(nParams * BPPT(cls));
    // param-bytes labels ride a data-raw tspan: hovering the rounded number
    // shows the unrounded byte count (attachTip) — the cross-check affordance
    const fmtPV = (n, cls = 'd') => PBYTES
      ? `<tspan data-raw="${(n * BPPT(cls)).toFixed(2)}">${fmtPB(n, cls)}</tspan>` : fmtP(n);
    const pk = (n, noK = false, cls = 'r') => {
      if (PBYTES && !BPPT()) return '';   // nothing visible, nothing to number
      if (LOCAL && this.partSel != null && this.partSel !== 1) return '';   // norms/micro ops are non-expert
      const v = (CUM && !noK ? fmtPV(n * KMUL, cls) : fmtPV(n, cls)) + facTxt(cls);
      return PONLY ? ` ${v}` : ` (${v})`;   // params lenses: no parens — params are the only numbers left
    };
    const pstr = (id) => {
      const p = PCNT[id];
      if (!p || (PBYTES && !BPPT())) return '';
      if (LOCAL && this.partSel != null && partOfCls(clsOf(id)) !== this.partSel) return '';
      const tot = (Array.isArray(p) ? p[0] * p[1] : p);
      const wrap = (str) => PONLY ? ` ${str}` : ` (${str})`;
      const fx = facTxt(clsOf(id));
      if (CUM && id !== 'lm_head') return wrap(fmtPV(tot * KMUL, clsOf(id)) + fx);
      if (Array.isArray(p)) return FLAT ? wrap(fmtPV(tot, clsOf(id)) + fx) : wrap(`${fmtPV(p[0], clsOf(id))} \u00d7${p[1]}`);
      return wrap(fmtPV(p, clsOf(id)) + fx);
    };
    const dt = (id) => this.matmuls[id];
    const marks = this._ctl.quant || this._ctl.marks ? this.marks : RECOMPUTE_PRESETS.none;   // static: save everything
    const stateA = (A, mks, id) => {   // chip state against a given analysis (tween endpoints)
      const n = A.byId[id];
      if (!n) return null;
      // the checkpoint-anchor lock only means something when a replay exists
      // to terminate at it: with recompute none, x0 is just a saved tensor
      if (n.always) return A.replayed.size ? 'pin' : (A.neededSaved.has(id) ? 'save' : 'idle');
      if (mks[markKey(id)] !== true) return 'redo';   // tied nodes read their partner's mark
      return A.neededSaved.has(id) ? 'save' : 'idle';
    };
    const state = (id) => stateA(ana, marks, id);
    // one-click precision toggle (bf16 ⇄ the fp8 flavor — the article is anchored on
    // bf16, so fp32 compute is not a lever here), hidden below the dtype tier.
    // The ROUTER is not a lever at all: production runs it fp32 (a tiny,
    // numerically sensitive GEMM) — its tag is a pinned readout of the recipe,
    // shown in EVERY quant tier (the AC section too) so the label never flips
    // between sections.
    const dtBtn = (id, x, y) => (id === 'router' ? !(this._ctl.quant || this._ctl.dtype) : !this._ctl.dtype) ? '' :
      // pinned tags are 4px wider for the 🔒 — the frame grows LEFT so the
      // right edge stays put inside the box
      `<foreignObject x="${id === 'router' || id === 'o_proj' ? x - 6 : x}" y="${y}" width="${id === 'router' || id === 'o_proj' ? 60 : 52}" height="20">` +
      (id === 'router'
        ? `<button xmlns="http://www.w3.org/1999/xhtml" class="st dtb" data-dt="${id}" disabled style="color:${C(DT_STYLE[dt(id)])};cursor:default;opacity:0.8;width:56px" ` +
          `title="🔒 pinned: the router runs fp32 in production (tiny GEMM, numerically sensitive) — it follows the recipe, not a per-op lever">${dt(id)} 🔒</button>`
        : id === 'o_proj'
          ? `<button xmlns="http://www.w3.org/1999/xhtml" class="st dtb" data-dt="${id}" disabled style="color:${C(DT_STYLE[COMPUTE_DT(dt(id))])};cursor:default;opacity:0.8;width:56px" ` +
            `title="🔒 pinned: the GEMM's COMPUTE dtype, which follows the recipe — its input's SAVE format is the 'E5M6 attn-out stash' checkbox above (the one GEMM whose stash and compute formats differ)">${COMPUTE_DT(dt(id))} 🔒</button>`
          : `<button xmlns="http://www.w3.org/1999/xhtml" class="st dtb" data-dt="${id}" style="color:${C(DT_STYLE[dt(id)])}" ` +
            `title="toggle precision: bf16 ⇄ ${FP8K}">${dt(id)}</button>`) + '</foreignObject>';
    // the block-output add has NO free mark: its output IS the next block's
    // x0 — the checkpoint anchor, always saved (and charged there). It wears
    // a locked 🔒 so every op still shows its state.
    const lockBtn = (x, y) => !this._ctl.marks ? '' :
      `<foreignObject x="${x}" y="${y}" width="26" height="20">` +
      `<button xmlns="http://www.w3.org/1999/xhtml" class="st mode st-save" disabled style="cursor:default;opacity:0.8" ` +
      `title="locked: this add's output IS the next block's x0 \u2014 the checkpoint anchor, always saved (charged to the next block)">\ud83d\udd12</button></foreignObject>`;
    const modeBtn = (ids, x, y) => {
      if (!this._ctl.marks) return '';       // hidden below the marks tier
      ids = ids.map(markKey);                // tied nodes (the SwiGLU-input quantize) mirror their partner's mark
      const st = state(ids[0]);
      if (st === 'pin') return '';
      const redo = this.marks[ids[0]] !== true;
      // a recompute nobody reads gets a warning (the mark is honored — literal
      // semantics — but a demand-driven planner would have skipped it)
      const warn = redo && ana.pointless?.has(ids[0])
        ? `<g data-tip="${escAttr('\u26a0 pointless recompute: nothing in backward reads this op\u2019s output — the replay burns time (and may pin its inputs in the stash) while saving no memory. torch_remat does what you said; a demand-driven planner would skip this mark.')}">` +
          `<text x="${x - 14}" y="${y + 15}" font-size="11">\u26a0\ufe0f</text></g>`
        : redo && NEUTRAL.has(ids[0])
          ? `<g data-tip="${escAttr('\u21c4 byte-NEUTRAL recompute: the replay is free (~0 FLOPs) and the stash just moves to an equal-sized tensor on the other side — net memory unchanged. On its own this mark buys nothing; it pays off as part of a longer replay chain (mark its inputs \u21bb too and the stash walks up toward x0).')}">` +
            `<text x="${x - 14}" y="${y + 15}" font-size="11" fill="${C('#52514e')}" font-weight="600">\u21c4</text></g>`
          : '';
      // the mark is a BOOLEAN — recompute in backward, yes/no. The old 💾
      // face overclaimed: an unmarked op's output is stashed only where
      // backward actually reads it (the chip beside the op shows the
      // outcome), so the off state is a struck ↻, not a save icon.
      return warn + `<foreignObject x="${x}" y="${y}" width="26" height="20">` +
        `<button xmlns="http://www.w3.org/1999/xhtml" class="st mode st-${redo ? 'redo' : 'keep'}" ` +
        `data-mark="${ids.join(',')}" title="recompute in backward? ↻ = yes: replay this op from its inputs (its output is never stashed). ` +
        `Struck ↻ = no: forward runs once — the output is stashed ONLY if some backward op actually reads it (the chip shows the outcome).">↻</button></foreignObject>`;
    };
    const blockGrid = (bytes, x, y, minOne = true, hollow = false, phantomBytes = 0, solidFrac = hollow ? 0 : 1) => {
      // byte squares, single-line ALWAYS (one stash = one run; wrapping is
      // banned — a wrapped grid crossed the wire routes). hollow = the
      // COUNTERFACTUAL stash: what saving this recomputed tensor would cost.
      // phantomBytes = the bf16-equivalent footprint: dashed squares extend
      // the run to what the stash WOULD cost wide — the outer edge is
      // dtype-independent (elems × 2 B), so dtype flips pour the solid fill
      // inside a fixed dashed silhouette (the ghost language: dashed = the
      // baseline you are beating).
      // solidFrac (save⇄recompute tween): the leading share of a hollow grid
      // drawn FILLED — squares convert one by one instead of two overlapping
      // grids crossfading (no ghost-flash)
      const n = Math.max(minOne ? 1 : 0, Math.round(bytes / 1024 / 4));
      const nPh = Math.max(n, Math.round(phantomBytes / 1024 / 4));
      if (!n && !nPh) return { svg: '', rows: 0, pitch: 11 };
      const nSolid = hollow ? Math.round(solidFrac * n) : n;
      let s = '';
      for (let i = 0; i < n; i++)
        s += i >= nSolid
          ? `<rect x="${x + i * 6 + 0.4}" y="${y + 0.4}" width="4.2" height="4.2" fill="none" stroke="${C('#eda100')}" stroke-width="0.8"/>`
          : `<rect x="${x + i * 6}" y="${y}" width="5" height="5" fill="${C('#eda100')}"/>`;
      for (let i = n; i < nPh; i++)
        s += `<rect x="${x + i * 6 + 0.4}" y="${y + 0.4}" width="4.2" height="4.2" fill="none" stroke="${C('#d19023')}" stroke-width="0.8" stroke-dasharray="1.6 1.4"/>`;
      return { svg: s, rows: 1, pitch: 11 };
    };
    const fmtB = (bytes) => bytes >= 1024 ? (bytes / 1024).toFixed(1) + ' KiB' : bytes + ' B';
    // combined view: totals over the block column — layers × in-flight microbatches × 4096 tokens
    const M = this.view === 'combined' ? this.dispLayers * this.dispInflight * TOK : 1;
    const fmtMem = (bytes) => {
      if (M === 1) return fmtB(bytes);
      const b = bytes * M;
      return b >= 2 ** 30 ? (b / 2 ** 30).toFixed(1) + ' GiB' : b >= 2 ** 20 ? (b / 2 ** 20).toFixed(0) + ' MiB' : (b / 1024).toFixed(0) + ' KiB';
    };
    // FLOP cost strip inside each op box, MFU-style: TIME at peak
    // (fp8 flavors counted half — 2× peak; fp32 at the CUDA-core rate). Scaled so the
    // largest op in the transformer block fills exactly one row of 30 blocks;
    // the lm head takes however many rows it needs at the same scale.
    // Colored by the op's precision; vector ops get a muted fig-leaf block.
    // e5m6 names a STASH format (the attn-out linear) — its GEMM runs fp8
    // time relative to the bf16 rate, calibrated to the H100 roofline: the
    // fp8 flavors run at 2× tensor peak (half width); fp32 runs on CUDA
    // cores (989/67 ≈ 14.8× bf16 time per FLOP — TF32 would defeat the
    // router pin's purpose, so it gets the true-fp32 rate)
    // the pickets' TIME base is the section's hardware (ctx.hw / hw attr; 02 = H100):
    // one picket = 10 MFLOP/token × 4096 tokens at that GPU's bf16 peak
    const HWP = HARDWARE[JSON.parse(this.getAttribute('ctx') ?? '{}').hw ?? this.hw ?? 'h100'];
    const HWPn = HW_SHORT[JSON.parse(this.getAttribute('ctx') ?? '{}').hw ?? this.hw ?? 'h100'];
    const PICKET_US = Math.round(10e6 * 4096 / HWP.flops.bf16 * 1e6);
    const RF32 = HWP.flops.bf16 / HWP.flops.fp32;
    const flopEq = (flopsTok, d) => flopsTok * (d === 'e4m3' || d === 'mxfp8' || d === 'e5m6' ? 0.5 : d === 'fp32' ? RF32 : 1);
    const opDt = (id) => {
      const n = ana.byId[id];
      if (!n) return 'vector';
      if (n.opKind === 'matmul' || n.opKind === 'attn') return COMPUTE_DT(dt(id === 'gate_up' ? 'ffn_gate_up' : id));
      return 'vector';
    };
    const escAttr = (s) => esc(s).replace(/"/g, '&quot;').replace(/\n/g, '&#10;');
    const boxTip = (id, dimsNote, paramId = id) => {
      const n = ana.byId[id];
      // the FLOP number only — its derivation lives in the cell sheet /
      // formula tooltips now (hand-written expression strings could diverge)
      const f = n?.flopsTok ? `${fmtNum(n.flopsTok)} FLOP/token` : '';
      const pc = exactParam(paramId);
      return ` data-tip="${escAttr([f, dimsNote, pc == null ? '' : `parameters: ${pc.toLocaleString('en-US')}`].filter(Boolean).join('\n'))}"`;
    };
    // dtype the sim ascribes to a stashed tensor (the dtype of the matmul whose
    // backward reads it — a real degree of freedom, so we surface it)
    // which fp8 FLAVOR this instance speaks (bytes are identical; the label
    // carries provenance): mxfp8 only if the current recipe actually uses it
    const FP8K = Object.values(this.matmuls).includes('mxfp8') ? 'mxfp8' : 'e4m3';   // e4m3 = the tile-scaled Hopper flavor's precise name
    const dtOf = (n) => {
      const b = n.outBytes / n.elems;
      return b >= 3.5 ? 'fp32' : b >= 1.7 ? 'bf16' : b >= 1.2 ? 'e5m6' : FP8K;
    };
    // 32/row: a power of two, so parallelism shards divide the strips cleanly
    // (EP64 on the ×58 MoE gate/up strip = 32·58/64 = 29 whole squares/rank),
    // and the byte unit lands exact: largestOp.moe/32 · 2 B = 448 MiB/square
    const FLOP_ROW = 32;
    const CHIP_ROW = 16;   // amber wire-chip squares wrap sooner (chips sit between columns)
    // strips="absolute": one FIXED unit (the per-block largest op fills a
    // row); cumulative strips GROW into space reserved for them, so the
    // toggle neither rescales nor reflows. Costs vertical space; dense/MoE
    // flips may reflow in this profile.
    const ABS = PBYTES && this.getAttribute('strips') === 'absolute';
    const PB_BASE = PARAMS.largestOp.moe / FLOP_ROW;
    const PB_UNIT = PB_BASE * (CUM && !ABS && !LOCAL ? KMUL : 1);   // local keeps the fixed 448 MiB unit
    // absolute profile: the STRIP grows, not the unit. The ×N toggle tweens
    // this._tween 0→1: squares pour in and the boxes grow with the filled
    // rows (compact at per-block, tall at cumulative).
    const CUMT = this._tween ?? (CUM ? 1 : 0);   // 0 = per block, 1 = ×N (mid-tween in between)
    // local hides the tabs in BOTH views (the kind follows the stage)
    const HIDT = LOCAL ? 1 : CUMT;
    const T_ = ABS ? CUMT : 0;
    const stripMul = ABS ? 1 + (KMUL - 1) * T_ : 1;
    // per-component cells for one op, at the current tween state: a toggled
    // component's squares pour in/out (count scales with t) and everything
    // downstream reflows smoothly — the per-block-tween style, not a fade.
    // A visible run that rounds to 0 shows one hollow trace square.
    // squares in local speak FULL (EP-independent) params: the EP share, the
    // stage ×N, and the ZeRO divisor all live in the tweened factor fT
    const sqParam = (id) => {
      const p = PCNT[id];
      if (LOCAL && clsOf(id) === 'e' && Array.isArray(p)) return p[0] * DSV3.routedExperts;
      return exactParam(id);
    };
    // per-comp cells FLOOR (sub-square remainders just round down — rounding
    // error is understood); ONE hollow trace appears only when the op would
    // otherwise show nothing at all, in the largest remainder's color
    const compCells = (nParams, cls) => COMPS.map((c) => {
      const m = cmult(c.prop);
      if (!m) return { c, f: 0, n: 0 };
      const pm = LOCAL ? pvis(partOfCls(cls)) : 1;   // sub-part filter (inside a solo)
      const eff = (LOCAL ? nParams * fT(c, cls) : nParams * bppOf(c, cls) * stripMul) * pm;
      const f = eff / 2 / PB_UNIT * m;
      return { c, f, n: Math.floor(f) };
    });
    const stripCells = (nParams, cls) => {
      const cells = compCells(nParams, cls);
      const n = cells.reduce((t, r) => t + r.n, 0);
      return n || (cells.some((r) => r.f > 0.02) ? 1 : 0);
    };
    const stripExtra = (nParams, cls, row = FLOP_ROW) => {   // box growth beyond the built-in strip row
      if (NOSTRIPS || !nParams || !(ABS || OPTIM)) return 0;
      return (Math.max(1, Math.ceil(stripCells(nParams, cls) / row)) - 1) * 6;
    };
    const paramBlocks = (x, y, nParams, cls, row = FLOP_ROW) => {
      if (NOSTRIPS || !PBYTES || !nParams) return;
      const cells = compCells(nParams, cls);
      let g = '', i = 0;
      for (const { c, n } of cells)
        for (let k = 0; k < n; k++, i++)
          g += `<rect x="${x + (i % row) * 6}" y="${y + Math.floor(i / row) * 6}" width="5" height="4" fill="${C(c.color)}"/>`;
      if (!i) {
        const top = cells.reduce((b2, r) => r.f > b2.f ? r : b2, { f: 0 });
        if (top.f > 0.02)
          g += `<rect x="${x + 0.4}" y="${y + 0.4}" width="4.2" height="3.2" fill="none" stroke="${C(top.c.color)}" stroke-width="0.8"/>`;
      }
      P.push(g);
    };
    // FLOP cost as a BAR: length = TIME at H100 peak (squares count bytes;
    // length measures time — the pipeline strip's language). ONE fixed
    // unit: a picket ≈ 41 µs per 4096-token microbatch (= 10 MFLOP/token
    // at the bf16 rate; e4m3 packs 20, CUDA-core fp32 only 0.68), so dtype
    // flips visibly stretch/shrink the runs instead of renormalizing the
    // scale. (bwd ≈ 280 pickets fills most of the tally runway)
    const TB_X = 62, TB_AVAIL = 852;   // tally ribbons: label gutter ('recompute' needs ~55px) + runway (sum keeps the svg at 1080)
    // NEUTRAL marks (⇄): a ↻ whose flip to keep would leave the stash total
    // EXACTLY unchanged — the recompute is free (an add, a rotation) and the
    // stash just moves to an equal-sized tensor on the other side. Detected
    // by counterfactual: re-analyze with that one mark flipped. Disjoint
    // from ⚠ (pointless takes precedence).
    const NEUTRAL = new Set();
    if (this._ctl.marks) {
      const cfNodes = blockGraph(this.kind, DSV3, this.matmuls, 4096);
      for (const nid of MARKABLE) {
        if (this.marks[nid] === true || ana.pointless?.has(nid) || !ana.byId[nid]) continue;
        const cf = analyze(cfNodes, { ...this.marks, [nid]: true }, this.transposed);
        if (Math.abs(cf.savedBytes - ana.savedBytes) < 1e-6) NEUTRAL.add(nid);
      }
    }
    // stash-knob tween endpoints (quant tiers): the previous analysis carries
    // membership + bytes, prev.mm the previous per-matmul dtypes. Widths and
    // colors lerp; numbers snap.
    const VQ = this._vtween?.prev?.anaPrev ? this._vtween : null;
    const anaP = VQ?.prev?.anaPrev;
    const lerpQ = (a, b) => VQ ? a + (b - a) * VQ.t : b;
    const dtPm = (id) => VQ?.prev?.mm?.[id];                 // undefined = unchanged
    const eqT = (flopsTok, dt2, dtp) => !dtp || dtp === dt2 ? flopEq(flopsTok, dt2)
      : lerpQ(flopEq(flopsTok, dtp), flopEq(flopsTok, dt2));
    const barColor = (dt2, dtp) => !VQ || !dtp || dtp === dt2 ? (C(DT_STYLE[dt2]) ?? C('#c3c2b7'))
      : fitColor(C(DT_STYLE[dtp]) ?? C('#c3c2b7'), C(DT_STYLE[dt2]) ?? C('#c3c2b7'), VQ.t);
    // compute = PICKETS: a tally of time quanta (2×5 rects at 3px pitch,
    // dtype-colored) — the 1D shape for the 1D quantity; squares stay bytes.
    // Sub-picket ops wear the hollow trace; the last picket is a partial-
    // width sliver (area exact). Same unit in the tally ribbons.
    const FUNIT = 10e6;
    const barRows = (flopsTok, dt2, maxW, dtp) => {
      if (!flopsTok || !this._ctl.quant || dt2 === 'vector') return 1;
      return Math.max(1, Math.ceil(Math.ceil(eqT(flopsTok, dt2, dtp) / FUNIT) / Math.floor(maxW / 3)));
    };
    const barExtra = (flopsTok, dt2, maxW, dtp) => (barRows(flopsTok, dt2, maxW, dtp) - 1) * 14;
    // DOUBLED rows: the fwd pickets (dtype color) and, below them, the
    // RECOMPUTE pickets (their own color) — the op's replay cost, pouring to
    // zero when it is saved. Space is reserved either way (no reflow).
    // recompute pickets keep the op's FORWARD dtype color, just LIGHTER
    // (fill-opacity): with dtypes owning the hue axis, a foreign recompute
    // hue would read as a fifth precision. Position (the second row / its
    // own ribbon) + lightness carry the replay meaning.
    const REDO_OP = ' fill-opacity="0.55"';
    const flopBar = (x, y, flopsTok, dt2, maxW = W - 16, dtp, id) => {
      if (!flopsTok || !this._ctl.quant) return;
      dt2 = COMPUTE_DT(dt2); dtp = dtp && COMPUTE_DT(dtp);   // pickets speak COMPUTE dtype (e5m6 is a stash format; pricing is identical)
      // recompute share, tweened: membership pours in/out with the marks
      const rT = !id ? 0 : lerpQ(
        anaP ? (anaP.replayed.has(id) ? flopEq(flopsTok, dtp ?? dt2) : 0) : (ana.replayed.has(id) ? flopEq(flopsTok, dt2) : 0),
        ana.replayed.has(id) ? flopEq(flopsTok, dt2) : 0) / FUNIT;
      if (dt2 === 'vector') {   // unpriced fig leaf: bandwidth-bound, and we model no epilogue fusions
        P.push(`<rect x="${x}" y="${y}" width="5" height="4" fill="none" stroke="${C('#c3c2b7')}" stroke-width="0.8" stroke-dasharray="1.5 1"/>`);
        if (rT > 0.001) P.push(`<g opacity="${Math.min(1, rT * 40).toFixed(2)}"><rect x="${x + 8}" y="${y}" width="5" height="4" fill="none" stroke="${C('#c3c2b7')}" stroke-opacity="0.7" stroke-width="0.8" stroke-dasharray="1.5 1"/></g>`);
        return;
      }
      const color = barColor(dt2, dtp);
      const n = eqT(flopsTok, dt2, dtp) / FUNIT, per = Math.floor(maxW / 3);
      if (n < 1) {   // sub-picket: the hollow trace
        P.push(`<rect x="${x + 0.3}" y="${y + 0.3}" width="1.4" height="4.4" fill="none" stroke="${color}" stroke-width="0.7"/>`);
        if (rT > 0.001) P.push(`<rect x="${x + 0.3}" y="${y + 7.3}" width="1.4" height="4.4" fill="none" stroke="${color}" stroke-width="0.7" opacity="${(Math.min(1, rT * 40) * 0.65).toFixed(2)}"/>`);
        return;
      }
      const rows = Math.ceil(n / per);
      for (let u = 0; u < Math.ceil(n); u++)
        P.push(`<rect x="${x + (u % per) * 3}" y="${y + Math.floor(u / per) * 7}" width="${(2 * Math.min(1, n - u)).toFixed(2)}" height="5" fill="${color}"/>`);
      for (let u = 0; u < Math.ceil(rT); u++)
        P.push(`<rect x="${x + (u % per) * 3}" y="${y + (rows + Math.floor(u / per)) * 7}" width="${(2 * Math.min(1, rT - u)).toFixed(2)}" height="5" fill="${color}"${REDO_OP}/>`);
    };
    // who reads this saved tensor in backward: consumer below (↓), the
    // producer's own backward above (↑), or both (⇅). If EVERY reader is a
    // REPLAY (the tensor is a recompute ANCHOR, not a direct backward
    // input), the arrow goes double-struck: ⇓ ⇑ ⇕ — the paper's own framing
    // ('cache the inputs of the SwiGLU operator and recompute its output')
    const needDir = (ids, only) => {
      const by = ids.flatMap(i => [...(ana.neededBy.get(i) ?? [])])
        .filter(c => !only || only.includes(c));
      const isR = (c) => c === 'replay anchor' || ana.replayed.has(c);
      const dir = (set2, glyphs) => {
        if (!set2.size) return '';
        const up = [...set2].some(c => ids.includes(c));
        const down = [...set2].some(c => !ids.includes(c));
        return glyphs[up && down ? 2 : up ? 1 : 0];
      };
      // a tensor can serve BOTH roles — show both families side by side
      return (dir(new Set(by.filter(c => !isR(c))), ['↓', '↑', '⇅'])
        + dir(new Set(by.filter(isR)), ['⇓', '⇑', '⇕'])) || '↓';
    };
    // the arrows explained, per chip: WHO keeps this tensor alive
    const needTip = (ids, only, brief = false) => {
      const by = ids.flatMap(i => [...(ana.neededBy.get(i) ?? [])])
        .filter(c => !only || only.includes(c));
      const isR = (c) => c === 'replay anchor' || ana.replayed.has(c);
      const lab = (c) => c === 'replay anchor' ? 'the replay chain (every replay ends at a saved anchor)' : (ana.byId[c]?.label ?? c);
      const bwd = [...new Set(by.filter(c => !isR(c)).map(lab))];
      const rep = [...new Set(by.filter(isR).map(lab))];
      return escAttr('kept alive by '
        + [bwd.length ? `the BACKWARD of: ${bwd.join(', ')}${brief ? '' : ' (single arrows \u2193\u2191\u21c5)'}` : '',
          rep.length ? `the REPLAY of: ${rep.join(', ')}${brief ? '' : ' (double arrows \u21d3\u21d1\u21d5)'}` : '']
          .filter(Boolean).join(', and ')
        + (brief ? '.' : '.\nDirection: \u2193\u21d3 read by an op below \u00b7 \u2191\u21d1 by this op\u2019s own backward \u00b7 \u21c5\u21d5 both.'));
    };
    // ov (optional): display-split override for a chip that shows part of one
    // graph node — { name, tdims, frac } (bytes and grid scale by frac)
    const tensorChip = (ids, x, y, ov) => {
      const id = ids[0], st = state(id), n = ana.byId[id];
      const bytes = ids.reduce((t, i) => t + ana.byId[i].outBytes * ana.mul(i), 0) * (ov?.frac ?? 1);
      const dualTag = stashTag(ana, ids);
      const name0 = ov?.name ?? n.tensor;
      if (PONLY) {
        // consolidated/local: the saved activations live ON THE WIRES, like
        // the AC diagram — amber chips at rank scale (name + bytes, squares
        // at the global unit), carrying the quant tiers' FULL stash language
        // (the union minus the flop pickets, which conflict for space):
        // dtype tags with ᵀ×2 duals, state tooltips, bf16 phantom tails,
        // and hollow ↻ counterfactual grids on recomputed tensors.
        const m = CONS ? cmult('showActs') : 0;
        if (!m) return 12;
        const name = esc(name0.replace(' (checkpoint anchor)', ''));
        const chipF = LOCAL ? actsT : ABS ? stripMul : CUM ? KMUL : 1;
        const CROW = ov?.short ? 8 : CHIP_ROW;
        // dtype tags are the LOCAL sim's union (the consolidated figures are
        // the earlier bf16 pedagogy — tags there just cross the enclosure)
        const dtag = !LOCAL ? () => '' : (A) => {
          const n2 = A.byId[id] ?? n;
          return ` <tspan fill="${C(DT_STYLE[dtOf(n2)])}">${dtOf(n2)}${stashTag(A, ids)}</tspan>`;
        };
        // tips ride the NAME tspan only (the byte value keeps its raw-B
        // hover free of conflicts); data-chip makes the chip a jump target
        const nameTip = (txt) => `<tspan data-tip="${escAttr(txt)}">`;
        if (st === 'redo') {   // recomputed: named + the would-be size + the counterfactual grid
          const cfB = bytes * TOK * (LOCAL ? (CUM ? KMUL : 1) * IFN : CUM ? KMUL : 1);
          const cf = ov?.flat ? 0 : Math.round(Math.round(bytes * TOK * chipF / (PB_UNIT * 2)) * m);
          const cfLbl = ov?.flat ? '' : ` <tspan class="tdim">(<tspan data-raw="${cfB.toFixed(2)}">${fmtBytes(cfB)}</tspan>)</tspan>`;   // flat: never a stash, no counterfactual
          const rTip = nameTip('↻ recomputed in backward, not stashed — the (size) and hollow squares price what saving it WOULD cost.');
          // narrow fork columns: the would-be size takes the second line
          let g = ov?.short
            ? `<text class="tensor tredo" x="${x}" y="${y + 8}">${rTip}↻ ${name}</tspan></text>` +
              `<text class="tensor tredo" x="${x}" y="${y + 21}">${cfLbl}${dtag(ana)}</text>`
            : `<text class="tensor tredo" x="${x}" y="${y + 8}">${rTip}↻ ${name}</tspan>${cfLbl}${dtag(ana)}</text>`;
          for (let i = 0; i < cf; i++)
            g += `<rect x="${x + (i % CROW) * 6 + 0.4}" y="${y + 12 + Math.floor(i / CROW) * 6 + 0.4}" width="4.2" height="3.2" fill="none" stroke="${C('#eda100')}" stroke-width="0.8"/>`;
          P.push(`<g data-chip="${ov?.chip ?? id}" opacity="${m.toFixed(3)}">${g}</g>`);
          return 12;
        }
        if (st !== 'save' && st !== 'pin') {   // idle: named + dtype (the wire's precision), never stashed
          P.push(`<g data-chip="${ov?.chip ?? id}" opacity="${m.toFixed(3)}"><text class="tensor tidle" x="${x}" y="${y + 8}">` +
            `${nameTip('· not needed: no backward op or replay reads this tensor — saved or not, it is never stashed.')}· ${name}</tspan>${dtag(ana)}</text></g>`);
          return 12;
        }
        // cumulative: every block's stash is resident — chips follow the ×N
        // convention (labels snap, squares grow with the tween like the strips)
        const b4096 = bytes * TOK * (LOCAL ? (CUM ? KMUL : 1) * IFN : CUM ? KMUL : 1);
        // stash-knob tween: the squares pour between the OLD and NEW bytes
        const VA = this._vtween?.prev?.anaPrev;
        const bytesT = VA
          ? ids.reduce((t2, i2) => {
            const nb = ana.byId[i2].outBytes * ana.mul(i2);
            const pb2 = (VA.byId[i2]?.outBytes ?? nb / ana.mul(i2)) * ana.mul(i2);
            return t2 + pb2 + (nb - pb2) * this._vtween.t;
          }, 0) * (ov?.frac ?? 1)
          : bytes;
        const full = Math.round(bytesT * TOK * chipF / (PB_UNIT * 2));
        const nsq = Math.round(full * m), hollow = !nsq && m >= 0.5 && chipF > 0;
        // the bf16 phantom tail: dashed squares out to the 2 B/elem edge —
        // dtype-independent, so recipe flips pour the solid fill inside a
        // fixed dashed silhouette (nothing to brag about when not beating it)
        const bfB = ids.reduce((t2, i2) => t2 + (ana.byId[i2]?.elems ?? 0) * 2, 0) * (ov?.frac ?? 1);
        const nPh = Math.round(Math.round(bfB * TOK * chipF / (PB_UNIT * 2)) * m);
        const lock = st === 'pin' ? ' 🔒' : '';
        // the saved-for-backward tip: BRIEF, and on the name only (the byte
        // value keeps its raw-B hover)
        const svTip = `<tspan data-tip="${needTip(ids, ov?.readers, true)
          + (st === 'pin' ? escAttr('\n🔒 always saved: the checkpoint anchor.') : '')}">`;
        // narrow fork columns (ov.short) get two lines — one line would run
        // into the neighbouring column's spine at ×58 byte widths
        // squares wrap well before the strip width (chips sit between
        // columns); the wire gaps grow with the rows (chipSpaceA prices them)
        const [sqX, sqY] = ov?.short ? [x + 100, y + 17] : [x, y + 12];   // past the dtype-tagged bytes line
        const bLbl = `<tspan data-raw="${b4096.toFixed(2)}">${fmtBytes(b4096)}</tspan>`;
        let g = ov?.short
          ? `<text class="tensor tsave" x="${x}" y="${y + 8}">${svTip}${needDir(ids, ov?.readers)} ${name}${lock}</tspan></text>` +
            `<text class="tensor tsave" x="${x}" y="${y + 21}">${bLbl}${dtag(ana)}${facTxt('a')}</text>`
          : `<text class="tensor tsave" x="${x}" y="${y + 8}">${svTip}${needDir(ids, ov?.readers)} ${name}</tspan> · ${bLbl}${dtag(ana)}${facTxt('a')}${lock}</text>`;
        if (hollow) g += `<rect x="${sqX + 0.4}" y="${sqY + 0.4}" width="4.2" height="3.2" fill="none" stroke="${C('#eda100')}" stroke-width="0.8"/>`;
        else for (let i = 0; i < Math.max(nsq, nPh); i++)
          g += i < nsq
            ? `<rect x="${sqX + (i % CROW) * 6}" y="${sqY + Math.floor(i / CROW) * 6}" width="5" height="4" fill="${C('#eda100')}"/>`
            : `<rect x="${sqX + (i % CROW) * 6 + 0.4}" y="${sqY + Math.floor(i / CROW) * 6 + 0.4}" width="4.2" height="3.2" fill="none" stroke="${C('#d19023')}" stroke-width="0.8" stroke-dasharray="1.6 1.4"/>`;
        P.push(`<g data-chip="${ov?.chip ?? id}" opacity="${m.toFixed(3)}">${g}</g>`);
        return 12;
      }

      let h = 12;
      if (!this._ctl.quant) {
        // structure only: name + backward-need direction, no bytes/dtype/grid
        const name = esc(name0.replace(' (checkpoint anchor)', ''));  // recompute vocabulary
        // unitless per-token size (element counts, like the op dims — dtype unspecified here)
        const sz = flatten(ov?.tdims ?? ids.map(i => ana.byId[i].tdims).join(' + '));
        P.push(st === 'idle'
          ? `<text class="tensor tidle" x="${x}" y="${y + 8}">· ${name}</text>`
          : `<text class="tensor tsave" x="${x}" y="${y + 8}">${needDir(ids, ov?.readers)} ${name} <tspan class="tdim">· ${sz}</tspan></text>`);
        return h;
      }
      // stash-knob tween: when the STATE flips (saved ↔ recomputed/idle) the
      // two text forms dissolve through each other, and the grid squares pour
      // between the OLD and NEW bytes (unsaved = 0) — nothing pops
      const bytesA = (A) => ids.reduce((t2, i2) => t2 + (A.byId[i2]?.outBytes ?? 0) * A.mul(i2), 0) * (ov?.frac ?? 1);
      const stP2 = VQ && anaP ? stateA(anaP, VQ.prev.marks ?? marks, id) : null;
      const chipTxt = (A, s2) => {
        const tip = s2 === 'save' || s2 === 'pin'
          ? ` data-tip="${needTip(ids, ov?.readers)}${s2 === 'pin' ? escAttr('\n\ud83d\udd12 always saved: the checkpoint anchor.') : ''}"`
          : s2 === 'redo'
            ? ` data-tip="${escAttr('\u21bb recomputed in backward, not stashed' + (this._ctl.marks ? ' \u2014 the hollow squares price what saving it WOULD cost.' : '.'))}"`
            : ` data-tip="${escAttr('\u00b7 not needed: no backward op or replay reads this tensor \u2014 saved or not, it is never stashed.')}"`;
        if (s2 === 'save' || s2 === 'pin') {
          const n2 = A.byId[id] ?? n;
          const dtag = stashTag(A, ids);
          // narrow fork/shared columns (ov.short) go TWO lines — one line
          // runs into the neighbouring column (the CONS chips' pattern)
          if (ov?.short) return `<text class="tensor tsave"${tip} x="${x}" y="${y + 8}">${needDir(ids, ov?.readers)} ${esc(name0)}${s2 === 'pin' ? ' 🔒' : ''}</text>` +
            `<text class="tensor tsave" x="${x}" y="${y + 21}">${fmtMem(bytesA(A))} <tspan fill="${C(DT_STYLE[dtOf(n2)])}">${dtOf(n2)}${dtag}</tspan></text>`;
          return `<text class="tensor tsave"${tip} x="${x}" y="${y + 8}">${needDir(ids, ov?.readers)} ${esc(name0)} · ${fmtMem(bytesA(A))} ` +
            `<tspan fill="${C(DT_STYLE[dtOf(n2)])}">${dtOf(n2)}${dtag}</tspan>${s2 === 'pin' ? ' 🔒' : ''}</text>`;
        }
        // dtype-bearing tiers state EVERY intermediate's precision, saved or
        // not (the paper is explicit even about unsaved wires — the combine
        // is bf16 both directions); the tag follows the same reader-dtype
        // convention the saved chips use
        const wdt = this.getAttribute('controls') === 'dtype' ? (() => {   // the dtype TIER only: the full tier's narrower kv column can't fit the longer labels
          const n3 = A.byId[id] ?? n;
          return ` <tspan fill="${C(DT_STYLE[dtOf(n3)])}">${dtOf(n3)}</tspan>`;
        })() : '';
        // ov.short: narrow fork columns drop the suffix (the ↻ glyph carries it)
        if (s2 === 'redo') return `<text class="tensor tredo"${tip} x="${x}" y="${y + 8}">↻ ${esc(name0)}${ov?.short ? '' : ' — recomputed'}${wdt}</text>`;
        return `<text class="tensor tidle"${tip} x="${x}" y="${y + 8}">· ${esc(name0)}${wdt}</text>`;
      };
      const SAVED = (s2) => s2 === 'save' || s2 === 'pin';
      // the grid: FILLED squares for a real stash, HOLLOW for a recomputed
      // tensor (the counterfactual: what saving it would cost) — flipping a
      // mark crossfades filled/hollow instead of vanishing the bytes
      // grid position follows the chip's FORM: two-line saved short chips
      // park it beside the value line; single-line forms (redo, non-short)
      // put it flush-left under the text — a hollow row at the +88 offset
      // floated into the neighbouring column
      const gpos = (s2) => ov?.short && SAVED(s2) ? [x + 88, y + 13] : ov?.short ? [x, y + 13] : [x, y + 12];
      // the bf16-equivalent footprint (phantom): what this stash would cost
      // at 2 B/elem — drawn only where the actual stash beats it (a dual fp8ᵀ
      // stash at 2.06 B/elem has nothing to brag about). dtype tiers only:
      // the marks tier's story is which tensors exist, not their width
      const bf16B = ids.reduce((t2, i2) => t2 + (ana.byId[i2]?.elems ?? 0) * 2, 0) * (ov?.frac ?? 1);
      // pass the edge UNCONDITIONALLY (blockGrid draws dashed only past the
      // solid fill): mid-tween the phantom must complement the LERPED fill
      // frame by frame — an endpoint-gated phantom pops in at the end
      const phFor = () => this._ctl.dtype ? bf16B : 0;
      // the hollow ↻ counterfactual grid belongs to the AC story — the pure
      // dtype tier (recompute pinned upstairs) drops it
      const redoGrid = (this._ctl.marks || !this._ctl.dtype) && !ov?.flat;
      const gridFor = (A, s2) => {
        const [gx, gy] = gpos(s2);
        return SAVED(s2) ? blockGrid(bytesA(A), gx, gy, true, false, phFor()).svg
          : s2 === 'redo' && redoGrid ? blockGrid(bytesA(A), gx, gy, true, true).svg : '';
      };
      if (stP2 != null && stP2 !== st) {
        // labels crossfade; the GRID never does — squares convert in place
        // (hollow⇄solid one by one) or pour to/from zero, so no ghost grid
        // flashes out while a filled one fades in
        P.push(`<g opacity="${(1 - VQ.t).toFixed(3)}">${chipTxt(anaP, stP2)}</g>` +
          `<g opacity="${VQ.t.toFixed(3)}">${chipTxt(ana, st)}</g>`);
        const has = (s2) => SAVED(s2) || (s2 === 'redo' && redoGrid);
        const g0 = has(stP2), g1 = has(st);
        if (g0 && g1 && ov?.short && gpos(stP2)[0] !== gpos(st)[0])
          // short chips park the grid in different pockets per state — a
          // moving grid can only crossfade
          P.push(`<g opacity="${(1 - VQ.t).toFixed(3)}">${gridFor(anaP, stP2)}</g>` +
            `<g opacity="${VQ.t.toFixed(3)}">${gridFor(ana, st)}</g>`);
        else if (g0 && g1) {
          const bT = lerpQ(bytesA(anaP), bytesA(ana));
          const [gx, gy] = gpos(st);
          const conv = SAVED(stP2) !== SAVED(st);   // save⇄pin never re-converts
          P.push(blockGrid(bT, gx, gy, true, conv, SAVED(st) ? phFor() : 0,
            conv ? (SAVED(st) ? VQ.t : 1 - VQ.t) : 1).svg);
        } else if (g0 !== g1) {
          // one side has no grid (the pure dtype tier drops the ↻ hollow):
          // the squares pour to/from zero instead of fading as a block
          const A2 = g1 ? ana : anaP, s3 = g1 ? st : stP2;
          const bT = lerpQ(g1 ? 0 : bytesA(anaP), g1 ? bytesA(A2) : 0);
          const [gx, gy] = gpos(s3);
          P.push(blockGrid(bT, gx, gy, false, s3 === 'redo', SAVED(s3) ? phFor() : 0).svg);
        }
      } else {
        P.push(chipTxt(ana, st));
        if (SAVED(st) || (st === 'redo' && redoGrid)) {   // same state: bytes still pour (dtype flips)
          const b0 = anaP ? bytesA(anaP) : bytes;
          const bT = lerpQ(b0, bytes);
          const [gx, gy] = gpos(st);
          // minOne holds through the tween when BOTH endpoints are nonzero —
          // a sub-square chip (kv latent: ¼ square) must not blink out
          // mid-pour; only a genuine pour to/from zero may reach zero squares.
          // The phantom edge is dtype-independent, so it never tweens: the
          // solid fill pours inside the fixed dashed silhouette
          P.push(blockGrid(bT, gx, gy, !VQ || (b0 > 0 && bytes > 0), st === 'redo',
            st === 'redo' ? 0 : phFor()).svg);
        }
      }
      if (SAVED(st) || st === 'redo') h = ov?.short ? 25 : 12 + 11 + 2;
      return h;
    };
    const wire = (cx, y1, y2) =>
      P.push(`<line class="wire" x1="${cx}" y1="${y1}" x2="${cx}" y2="${y2}" marker-end="url(#arr)"/>`);
    // reserve chip space for the WORST case (saved, bf16) so toggling
    // save/recompute or precision never reflows the layout
    const chipSpaceA = (anaX, ids) => {
      if (!this._ctl.quant) {
        const mA = CONS ? cmult('showActs') : 0;
        if (!mA) return 18;                            // one text line
        // amber chip squares below the text: reserve their rows (worst case
        // over the knob tween: the larger of the old and new configuration),
        // easing with the acts checkbox tween so the gap never pops
        const chipF = LOCAL
          ? Math.max(dLoc(Snow).acts, this._vtween ? dLoc(this._vtween.prev).acts : 0)
          : CUM ? KMUL : 1;
        const b = ids.reduce((t, i) => {
          const n2 = anaX.byId[i];   // worst case: the ᵀ dual OR the bf16 phantom edge
          return t + Math.max(n2.outBytes * anaX.mul(i), n2.elems * 2);
        }, 0) * TOK * chipF;
        const rows = Math.max(1, Math.ceil(Math.round(b / (PB_UNIT * 2)) / CHIP_ROW));
        return Math.round(18 + (rows * 6 - 2) * mA);
      }
      return 12 + 11 + 2;   // single-line byte squares: one fixed band
    };
    const chipSpace = (ids) => chipSpaceA(ana, ids);
    // the MoE column's wire gaps, measured on the parallel MoE analysis —
    // the dense column advances by these to stay row-aligned across the flip
    const gapM = (ids) => Math.max(22, chipSpaceA(anaM, ids) + 10);
    const wireOut = (ids, sx, y, ov) => {
      tensorChip(ids, sx + 14, y + 4, ov);
      // ov.flat: a wire that is never stashed (the gate/up GEMM's bf16
      // output — the quantized copy below it is the stash): one text line,
      // no grid, no worst-case reservation
      const gap = Math.max(22, (ov?.flat ? 18 : chipSpace(ids)) + 10);
      wire(sx, y, y + gap);
      return y + gap;
    };
    // aux backward artifact, exiting the box to the right
    const auxOut = (id, x, yMid) => {
      if (PONLY && !LOCAL) return;
      const n = ana.byId[id];
      if (!n.aux) return;
      const replayed = ana.replayed.has(id); // a replay regenerates its aux
      // attention sits inside the MLA group: its lse label starts past the
      // group border so the border never cuts through the text
      const xt = (id === 'attn' && MLAGW && !(this._ctl.quant && this.detail) && !LOCAL) ? Math.max(x + W + 14, C1 - 10 + MLAGW + 8) : x + W + 14;
      const auxSc = n.aux ? n.aux.bytes * TOK * (CUM ? KMUL : 1) * IFN : 0;
      const txt = (rep) => rep
        ? (LOCAL
          ? `<text class="tensor tredo" x="${xt}" y="${yMid + 3}">↻ ${esc(n.aux.name)} <tspan class="tdim">(<tspan data-raw="${auxSc.toFixed(2)}">${fmtBytes(auxSc)}</tspan>)</tspan></text>`
          : `<text class="tensor tredo" x="${xt}" y="${yMid + 3}">↻ ${esc(n.aux.name)}</text>`)
        : LOCAL
          ? `<text class="tensor tsave" x="${xt}" y="${yMid + 3}">← ${esc(n.aux.name)} · ` +
            `<tspan data-raw="${auxSc.toFixed(2)}">${fmtBytes(auxSc)}</tspan> ` +
            `<tspan fill="${DT_STYLE.fp32}">fp32</tspan></text>`
          : !this._ctl.quant
            ? `<text class="tensor tsave" x="${xt}" y="${yMid + 3}">← ${esc(n.aux.name)}</text>`
            : `<text class="tensor tsave" x="${xt}" y="${yMid + 3}">← ${esc(n.aux.name)} · ${fmtMem(n.aux.bytes)} ` +
              `<tspan fill="${DT_STYLE.fp32}">fp32</tspan></text>`;
      const repP = anaP?.replayed.has(id);   // mark-flip tween: the two forms dissolve
      P.push(`<g data-chip="${id}:aux"><line class="wire" x1="${x + W}" y1="${yMid}" x2="${xt - 4}" y2="${yMid}" marker-end="url(#arr)"/>` +
        (VQ && repP != null && repP !== replayed
          ? `<g opacity="${(1 - VQ.t).toFixed(3)}">${txt(repP)}</g><g opacity="${VQ.t.toFixed(3)}">${txt(replayed)}</g>`
          : txt(replayed)) + '</g>');
    };
    // display-only elided kernel (detail view): cheap, no marks, not in the graph
    const DET = this.detail;
    let MLAGW = 0;   // MLA group width — attention's lse label starts past its right edge
    const micro = (label, x, y, w = W, tip, pc = '', opId = null) => {
      const pcTip = opId == null || tip?.includes('parameters:') ? null : exactParam(opId);
      const tip2 = [tip, pcTip == null ? '' : `parameters: ${pcTip.toLocaleString('en-US')}`].filter(Boolean).join('\n');
      const body = `<rect class="micro" x="${x}" y="${y}" width="${w}" height="18" rx="9"/>` +
        `<text class="microlabel" x="${x + 9}" y="${y + 13}">${label}${pc ? `<tspan class="dims"> ${pc}</tspan>` : ''}</text>`;
      P.push(tip2 || opId ? `<g${opId ? ` data-op="${opId}"` : ''}${tip2 ? ` data-tip="${escAttr(tip2)}"` : ''}>${body}</g>` : body);
      // real graph nodes drawn as micros (the MLA latent RMSNorms) carry the
      // save/recompute button like any other markable op
      if (opId && ana.byId[opId] && !ana.byId[opId].always) P.push(modeBtn([opId], x + w - 28, y - 1));
      return y + 18;
    };
    const plus = (cx, y) => P.push(`<circle cx="${cx}" cy="${y}" r="9" class="box"/>` +
      `<text class="plus" x="${cx}" y="${y + 4}" text-anchor="middle">+</text>`);
    // the dashed add box beside a + junction (residual adds, routed+shared add)
    const addBox = (x, yMid, label, tip, w = 126) => P.push(`<g data-tip="${escAttr(tip)}">` +
      `<rect class="res" x="${x}" y="${yMid - 11}" width="${w}" height="22" rx="4"/>` +
      `<text class="oplabel" x="${x + 8}" y="${yMid + 4}">${label}</text></g>`);
    const grp = (x, y0, y1, label, w = W + 20) => P.push(
      `<rect class="grp" x="${x - 10}" y="${y0}" width="${w}" height="${y1 - y0}" rx="6"/>` +
      `<text class="grplabel" x="${x - 2}" y="${y0 + 11}">${label}</text>`);
    // REGION toggle (AC tiers): set every mark in a column at once — the
    // tristate ↻ none / ↻ all / mixed segment, where 'mixed' both displays an
    // in-between state and REMEMBERS the last one (the custom-chip pattern)
    const MLA_RIDS = ['qkv_down', 'q_norm', 'kv_norm', 'q_up', 'kv_up', 'rope_q', 'rope_kv', 'attn', 'o_proj'];
    const regionToggle = (rids, memKey, fx, fy, fw) => {
      if (!this._ctl.marks || !this._noKind) return '';
      const st3 = rids.every(i => this.marks[i] === true) ? 'save'
        : rids.every(i => this.marks[i] !== true) ? 'redo' : 'mixed';
      if (st3 === 'mixed')
        (this._segMem ??= {})[memKey] = Object.fromEntries(rids.map(i => [i, this.marks[i] === true]));
      const hasMix = !!this._segMem?.[memKey];
      const pv = this._segMem?.[memKey + ':prev'];   // toggle-back: the pick before this one
      const rb = (act, label, onStyle, on, dis, title) =>
        `<button xmlns="http://www.w3.org/1999/xhtml" class="st" data-regionact="${act}" data-mem="${memKey}" ` +
        `data-rids="${rids.join(',')}" data-on="${on ? 1 : 0}"${dis ? ' disabled' : ''} ` +
        `style="width:auto;padding:0 7px;height:18px;` +
        `${on ? onStyle + 'font-weight:600;' + (pv ? 'cursor:pointer;' : 'cursor:default;') : 'background:var(--c-ffffff);border:1px solid var(--c-c3c2b7);color:var(--c-52514e);'}` +
        `${dis ? 'opacity:0.45;cursor:default;' : ''}" title="${on && pv ? `click again \u2014 back to ${pv.sel}` : title}">${label}</button>`;
      return `<foreignObject x="${fx}" y="${fy}" width="${fw}" height="22">` +
        `<div xmlns="http://www.w3.org/1999/xhtml" style="display:flex;gap:4px;justify-content:flex-end;">` +
        rb('save', '\u21bb none', 'background:var(--c-ffffff);border:1px solid var(--c-898781);color:var(--c-0b0b0b);', st3 === 'save', false,
          'recompute nothing in this region \u2014 outputs stay available (stashed only where backward reads them)') +
        rb('redo', '\u21bb all', 'background:var(--c-f3f2ee);border:1px dashed var(--c-898781);color:var(--c-52514e);', st3 === 'redo', false,
          'recompute this ENTIRE region in backward') +
        rb('mixed', 'mixed', 'background:var(--c-f3f2ee);border:1px solid var(--c-898781);color:var(--c-0b0b0b);', st3 === 'mixed', !hasMix && st3 !== 'mixed',
          'your last mixed set of marks for this region') +
        '</div></foreignObject>';
    };
    const mmBox = (ids, x, y, markIds, label, dims) => {
      const spec = MATMULS.find(m => m.id === ids[0]);
      const extra = stripExtra(sqParam(ids[0]), clsOf(ids[0]))
        + barExtra(ana.byId[(markIds ?? ids)[0]]?.flopsTok, dt(ids[0]), W - 16, dtPm(ids[0]));
      P.push(`<g data-op="${ids[0]}"${boxTip((markIds ?? ids)[0], dims ? undefined : spec.dimsNote, ids[0])}>` +
        `<rect class="box" x="${x}" y="${y}" width="${W}" height="${BH + extra}" rx="4"/>` +
        (PBYTES && !this._ctl.dtype && exactParam(ids[0]) != null ? `<text class="dims" x="${x + W - 8}" y="${y + 13}" text-anchor="end">bf16</text>` : '') +
        `<text class="name" x="${x + 8}" y="${y + 13}">${label ?? spec.label}</text>` +
        `<text class="dims" x="${x + 8}" y="${y + 26}">${PONLY ? pstr(ids[0]).trim() : flatten(dims ?? spec.dims) + pstr(ids[0])}</text></g>`);
      P.push(modeBtn(markIds ?? ids, x + W - 86, y + 6));
      P.push(dtBtn(ids[0], x + W - 58, y + 6));
      auxOut((markIds ?? ids)[0], x, y + 19);
      flopBar(x + 8, y + 30, ana.byId[(markIds ?? ids)[0]]?.flopsTok, dt(ids[0]), W - 16, dtPm(ids[0]), (markIds ?? ids)[0]);
      paramBlocks(x + 8, y + 30, sqParam(ids[0]), clsOf(ids[0]));
      return y + BH + extra;
    };
    // the SwiGLU-input quantize pill's tooltip (dtype tiers only draw the pill)
    const QUANT_TIP = 'quantize the gate/up GEMM’s bf16 output for the stash — FUSED into the SwiGLU kernel in production (one pass reads gate,up in bf16 and writes both the swiglu output and this copy). ' +
      'This copy IS the gate/up stash: an fp8 GEMM’s own output is high precision and nothing in backward reads it (a matmul’s backward needs its INPUT; the SwiGLU’s backward reads this). ' +
      'Its format is free-floating — no GEMM ever consumes it, so no GEMM forces the precision; the paper CHOOSES fp8 (§3.3.3: ‘cache the inputs of the SwiGLU operator in FP8’). bf16 = the identity (the GEMM output kept as-is). ' +
      'Its ↻ mark is the gate/up GEMM’s: recomputing the GEMM re-quantizes.';
    const opNode = (id, label, x, y, cls = 'op', pc = '') => {
      const h2 = cls === 'comm' || !BQ ? 22 : this._ctl.quant ? 34 : 27;   // the extra rows hold the fig-leaf + recompute stubs
      // the quantize pill carries the stash-format button (data-dt swiglu_in):
      // the SwiGLU INPUT's save precision is free-floating (the elementwise
      // backward is the only reader — no GEMM forces it), so it gets its own
      // lever outside the GEMM boxes; its ↻ is the gate/up GEMM's (tied mark)
      const quantBtn = id === 'quant' && this._ctl.dtype
        ? `<foreignObject x="${x + W - 88}" y="${y + 1}" width="54" height="20">` +
          `<button xmlns="http://www.w3.org/1999/xhtml" class="st dtb" data-dt="swiglu_in" style="color:${C(DT_STYLE[this.matmuls.swiglu_in ?? 'bf16'])};width:52px" ` +
          `title="the gate/up STASH format (the SwiGLU input) — free-floating: no GEMM reads it in forward or backward (SwiGLU backward is elementwise), ` +
          `so no GEMM forces its precision. The paper CHOOSES fp8 (§3.3.3: 'cache the inputs of the SwiGLU operator in FP8'); bf16 = no quantize. Toggle bf16 ⇄ ${FP8K}.">${this.matmuls.swiglu_in ?? 'bf16'}</button></foreignObject>`
        : '';
      P.push(`<g data-op="${id}"${id === 'quant' ? ` data-tip="${escAttr(QUANT_TIP)}"` : boxTip(id)}>` +
        `<rect class="${cls}" x="${x}" y="${y}" width="${W}" height="${h2}" rx="6"/>` +
        `<text class="oplabel" x="${x + 10}" y="${y + 15}">${label}${pc ? `<tspan class="dims"> ${pc}</tspan>` : ''}</text></g>` +
        quantBtn + modeBtn([id], x + W - 30, y + 1));
      auxOut(id, x, y + Math.round(h2 / 2));
      if (cls !== 'comm') { flopBar(x + 10, y + 19, ana.byId[id]?.flopsTok, 'vector', W - 16, undefined, id); paramBlocks(x + 10, y + 19, sqParam(id)); }
      return y + h2;
    };

    // ---- column 1: MLA (skipped in only="ffn" mode) ----
    let y = 14, x1Y = 14, col1End = 44;
    if (ONLY !== 'ffn') {
    P.push(`<text class="oplabel" x="${SX1 + 14}" y="${y}">x — residual stream (7168)</text>`);
    tensorChip(['x0'], SX1 + 170, y - 8);
    const tap1 = y + 6;
    wire(SX1, y + 3, y + 18); y += 18;
    y = opNode('norm1', 'RMSNorm', C1, y, 'op', pk(DSV3.hidden).trim());
    let g1;
    y = wireOut(['norm1'], SX1, y); g1 = y + 3; y += 27;
    let bypX = 0;                                // k_rope rail x (set in the MLA fork block)
    {
      // the AC tiers widen the fork (+50) so saved-chip lines never wrap;
      // static/params/local keep the published 150 (01 is live)
      const KVO = (this._ctl.quant || (this.hasAttribute('local') && this.getAttribute('lens') === 'param-bytes')) && DET ? 200 : 150;
      const RX = C1 + KVO + 22;
      // the down-projection is two separate GEMMs in production stacks
      // (wq_a | wkv_a in every production stack), so it is split at every tier:
      // fork norm1-out first
      P.push(`<circle cx="${SX1}" cy="${y - 10}" r="2.5" fill="${C('#898781')}"/>` +
        `<path class="wire" d="M ${SX1} ${y - 10} L ${RX} ${y - 10} L ${RX} ${y}" marker-end="url(#arr)"/>`);
      const qFrac = DSV3.qRank / (DSV3.qRank + DSV3.kvRank + DSV3.qkRope);
      const HALF_ROW = 21;   // 140px half boxes: strip rows wrap inside the box
      const dhalf = (x, name, dims, tip, frac, withBtns, pc = '', nP = 0) => {
        const ex = stripExtra(nP, 'd', HALF_ROW)   // strip rows can outgrow the box (×N)
          + barExtra(ana.byId.qkv_down.flopsTok * frac, dt('qkv_down'), 128, dtPm('qkv_down'));
        P.push(`<g data-op="qkv_down" data-tip="${escAttr(tip)}">` +
          `<rect class="box" x="${x}" y="${y}" width="140" height="${HBH + ex}" rx="4"/>` +
          (PBYTES && !this._ctl.dtype ? `<text class="dims" x="${x + 134}" y="${y + 13}" text-anchor="end">bf16</text>` : '') +
          `<text class="name" x="${x + 6}" y="${y + 13}">${name}</text>` +
          `<text class="dims" x="${x + 6}" y="${y + 25}">${PONLY ? pc.trim() : flatten(dims) + pc}</text></g>` +
          (withBtns ? modeBtn(['qkv_down'], x + 140 - 86, y + 29) + dtBtn('qkv_down', x + 140 - 58, y + 29) : ''));
        flopBar(x + 6, y + 52, ana.byId.qkv_down.flopsTok * frac, dt('qkv_down'), 128, dtPm('qkv_down'), 'qkv_down');
        paramBlocks(x + 6, y + 52, nP, 'd', HALF_ROW);
        return ex;
      };
      const pQ = pk(DSV3.hidden * DSV3.qRank), pKV = pk(DSV3.hidden * (DSV3.kvRank + DSV3.qkRope));
      const exQ = dhalf(C1, 'q down-proj', '7168 → 1536',
        `2 · 7168 · 1536 FLOP/token — wq_a; a separate GEMM from kv down-proj in production stacks\nparameters: ${(DSV3.hidden * DSV3.qRank).toLocaleString('en-US')}`, qFrac, true, pQ, DSV3.hidden * DSV3.qRank);
      const exKV = dhalf(C1 + KVO, 'kv down-proj', '7168 → 512 + 64',
        `2 · 7168 · (512 + 64) FLOP/token — wkv_a; shares q down-proj’s mark and dtype (one graph node — the buttons are MIRRORS)\nparameters: ${(DSV3.hidden * (DSV3.kvRank + DSV3.qkRope)).toLocaleString('en-US')}`, 1 - qFrac, true, pKV, DSV3.hidden * (DSV3.kvRank + DSV3.qkRope));
      y += HBH + Math.max(exQ, exKV);   // the pair advances together
      // display-split of the one latents stash. What backward keeps is the
      // POST-norm latent (the up-proj's input), so in detail the chips sit
      // below the RMSNorm row. The kv down-proj box has TWO outputs: k_rope
      // (64) leaves from its bottom-right corner immediately and rides an
      // outer rail (clear of chip text) down to the kv-side RoPE.
      const latTot = DSV3.qRank + DSV3.kvRank;   // the k_rope dims are never stashed
      bypX = C1 + 208 + KVO;                     // k_rope rail, clear of all chip text
      let bypTop = 0;
      if (DET) {
        const kx = C1 + KVO + 122;
        bypTop = y + 14;
        P.push(`<path class="wire" d="M ${kx} ${y} L ${kx} ${bypTop} L ${bypX} ${bypTop}"/>`);
        // quant tiers: idle-chip form (name only, +dtype in the dtype tier —
        // never stashed, so no size claim); structure tiers keep the dims
        if (!PONLY) P.push(`<text class="tensor tidle" x="${kx + 6}" y="${bypTop - 4}">· k_rope${
          this._ctl.quant ? '' : ` · ${DSV3.qkRope}`}${
          this.getAttribute('controls') === 'dtype' ? ` <tspan fill="${DT_STYLE.bf16}">bf16</tspan>` : ''}</text>`);   // the rail feeds the bf16 attention core
        else if (CONS) P.push(`<text class="tensor tidle" x="${kx + 6}" y="${bypTop - 4}">· k_rope</text>`);   // named, idle: never saved (RoPE bwd is a rotation)
        // pre-norm latent chips: real graph state (saved at no-AC — the latent
        // norms' backward input; the replay anchor under recompute presets)
        tensorChip(['qkv_down'], SX1 + 14, y + 24,
          { name: 'q latent', tdims: String(DSV3.qRank), frac: DSV3.qRank / latTot, short: !this._ctl.quant, readers: ['q_norm'] });
        tensorChip(['qkv_down'], RX + 14, y + 24,
          { name: 'kv latent', tdims: String(DSV3.kvRank), frac: DSV3.kvRank / latTot, short: !this._ctl.quant, readers: ['kv_norm'] });
        wire(SX1, y, y + 48);
        P.push(`<path class="wire" d="M ${RX} ${y} L ${RX} ${y + 48}" marker-end="url(#arr)"/>`);
        y += 48;
        // MLA-internal RMSNorms (q_a_layernorm; kv_a_layernorm norms the 512 only)
        const normTip = 'input-form backward: reads its INPUT (the pre-norm latent) + rstd — never its own output. ' +
          'That input is the bf16 latent chip above (kept wide — a disclosed inference; the paper doesn\u2019t state the anchor format). ' +
          'The normed OUTPUT is the up-proj\u2019s cached fp8 input — stashed only if this norm is saved AND the up-proj isn\u2019t replaying.';
        micro('RMSNorm', C1, y, 140, normTip, pk(DSV3.qRank).trim(), 'q_norm');
        micro('RMSNorm', C1 + KVO, y, 140, normTip, pk(DSV3.kvRank).trim(), 'kv_norm');
        y += 18;
        // their rstd: exits the bottom, elbows right (\u2191 = read by the op
        // above, the norm's own backward); a replayed norm regenerates it
        if (!PONLY) for (const [nid, bx] of [['q_norm', C1], ['kv_norm', C1 + KVO]]) {
          const rep = ana.replayed.has(nid);
          P.push(`<path class="wire" d="M ${bx + 112} ${y} L ${bx + 112} ${y + 7} L ${bx + 124} ${y + 7}" marker-end="url(#arr)"/>` +
            `<text class="tensor ${rep ? 'tredo' : 'tsave'}" x="${bx + 128}" y="${y + 10}">${rep ? '\u21bb' : '\u2191'} rstd</text>`);
        }
        y += 14;
      }
      if (DET) {
        // the normed latents are their own graph nodes (q_norm / kv_norm)
        tensorChip(['q_norm'], SX1 + 14, y + 4, { short: !this._ctl.quant });
        tensorChip(['kv_norm'], RX + 14, y + 4, { short: !this._ctl.quant });
      } else {
        tensorChip(['qkv_down'], SX1 + 14, y + 4,
          { name: 'q latent', tdims: String(DSV3.qRank), frac: DSV3.qRank / latTot, short: !this._ctl.quant, readers: ['q_norm'] });
        tensorChip(['qkv_down'], RX + 14, y + 4,
          { name: 'kv latent', tdims: String(DSV3.kvRank), frac: DSV3.kvRank / latTot, short: !this._ctl.quant, readers: ['kv_norm'] });
      }
      // consolidated: the narrow fork chips are two lines tall (name / bytes)
      const latGap = Math.max(CONS ? 36 : 26, chipSpace(['qkv_down']) + 8);
      const wireTop = DET ? y - 14 : y;          // span the rstd band too — no spine gap
      wire(SX1, wireTop, y + latGap);
      P.push(`<path class="wire" d="M ${RX} ${wireTop} L ${RX} ${y + latGap}" marker-end="url(#arr)"/>`);
      y += latGap;
      const halfBox = (id, x) => {
        const m = MATMULS.find(mm2 => mm2.id === id);
        const ex = stripExtra(sqParam(id), 'd', 21)   // strip rows can outgrow the box (×N; 140px box row)
          + barExtra(ana.byId[id]?.flopsTok, dt(id), 128, dtPm(id));
        P.push(`<g data-op="${id}"${boxTip(id, m.dimsNote)}>` +
          `<rect class="box" x="${x}" y="${y}" width="140" height="${HBH + ex}" rx="4"/>` +
          (PBYTES && !this._ctl.dtype ? `<text class="dims" x="${x + 134}" y="${y + 13}" text-anchor="end">bf16</text>` : '') +
          `<text class="name" x="${x + 6}" y="${y + 13}">${m.label}</text>` +
          `<text class="dims" x="${x + 6}" y="${y + 25}">${PONLY ? pstr(id).trim() : flatten(m.dims) + pstr(id)}</text></g>` +
          modeBtn([id], x + 140 - 86, y + 29) + dtBtn(id, x + 140 - 58, y + 29));
        flopBar(x + 6, y + 52, ana.byId[id]?.flopsTok, dt(id), 128, dtPm(id), id);
        paramBlocks(x + 6, y + 52, sqParam(id), 'd', 21);
        return ex;
      };
      y += HBH + Math.max(halfBox('q_up', C1), halfBox('kv_up', C1 + KVO));   // the pair advances together
      if (DET) {
        // the up-proj outputs get names before RoPE; then two separate RoPE
        // kernels (Megatron: apply_mla_rope_for_q / _for_kv) — the kv one is
        // rope plus a little extra (split, broadcast, assemble K and V) — feed
        // q and k,v directly into attention. The k_rope rail lands here.
        if (this._ctl.quant) {
          // the PRE-RoPE outputs are real chips in the quant tiers: with RoPE
          // marked \u21bb and an up-proj marked \ud83d\udcbe, THIS is where the stash
          // lives — the ledger must be visible wherever it lands. (Static/params
          // tiers keep the plain idle labels — 01 is published.)
          tensorChip(['q_up'], SX1 + 14, y + 2);
          tensorChip(['kv_up'], RX + 14, y + 2);
          wire(SX1, y, y + 28);
          P.push(`<path class="wire" d="M ${RX} ${y} L ${RX} ${y + 28}" marker-end="url(#arr)"/>`);
          y += 28;
        } else {
          if (!PONLY) P.push(`<text class="tensor tidle" x="${SX1 + 14}" y="${y + 12}">q_heads · ${flatten('128×192')}</text>` +
            `<text class="tensor tidle" x="${RX + 14}" y="${y + 12}">kv_heads · ${flatten('128×(128+128)')}</text>`);
          wire(SX1, y, y + 16);
          P.push(`<path class="wire" d="M ${RX} ${y} L ${RX} ${y + 16}" marker-end="url(#arr)"/>`);
          y += 16;
        }
        micro('RoPE', C1, y, 140,
          'fused_apply_mla_rope_for_q — rotate the 64 rope dims of every q head (fp32), make Q contiguous. '
          + 'A real (zero-byte) mark: save the rotated q — the SAME bytes as pre-RoPE — or re-run the rotation in backward (bandwidth-bound, unmetered here)', '', 'rope_q');
        micro('RoPE + build K,V', C1 + KVO, y, 140,
          'fused_apply_mla_rope_for_kv — split kv_heads into k_nope and V, rotate k_rope, broadcast it across the 128 heads, concat K = [k_nope | k_rope], make K and V contiguous. '
          + 'Same zero-byte mark as RoPE (q): rotated vs pre-RoPE k,v are the same size', '', 'rope_kv');
        P.push(`<path class="wire" d="M ${bypX} ${bypTop} L ${bypX} ${y + 9} L ${C1 + KVO + 141} ${y + 9}" marker-end="url(#arr)"/>`);
        y += 18;
      }
      tensorChip(['rope_q'], SX1 + 14, y + 4);
      tensorChip(['rope_kv'], RX + 14, y + 4);
      const csp = Math.max(chipSpace(['rope_q']), chipSpace(['rope_kv']));
      const elb = y + csp + 6;   // the return elbow rides BELOW the chip band
      const gap = Math.max(24, csp + 16);
      wire(SX1, y, y + gap);
      P.push(`<path class="wire" d="M ${RX} ${y} L ${RX} ${elb} L ${SX1 + 3} ${elb}"/>` +
        `<circle cx="${SX1}" cy="${elb}" r="2.5" fill="${C('#898781')}"/>`);
      y += gap;
    }
    MLAGW = DET ? bypX - C1 + 22 : 336;   // before the attn row: its lse label clears this edge
    y = mmBox(['attn'], C1, y);
    y = wireOut(['attn'], SX1, y);
    y = mmBox(['o_proj'], C1, y);
    // fixed-parallelism (ctx) instances already SUM the layers in their
    // readout (× 8 layers × 8.5 in flight) — a ×61 multiplier on the
    // enclosure would double-claim, so the label goes per-block there
    grp(C1, g1, y + 5, CUM ? `MLA · ${fmtPV(PARAMS.mla * KMUL)}${facTxt('d')}`
      : this.getAttribute('ctx') ? `MLA · ${fmtPV(PARAMS.mla)}${facTxt('d')}`
        : `MLA ×${DSV3.layers} · ${fmtPV(PARAMS.mla)}${facTxt('d')}`, MLAGW);
    P.push(regionToggle(MLA_RIDS, 'mlaMixed', C1 - 10 + MLAGW - 196, g1 + 1, 190));
    y = wireOut(['o_proj'], SX1, y + 5);
    if (ONLY === 'mla') {
      // component view: the residual add lives in the block wiring, not here
      P.push(`<text class="tensor tidle" x="${SX1 + 8}" y="${y + 4}">→ ⊕ residual add (block wiring)</text>`);
      x1Y = y;
      col1End = y + 24;
    } else {
    y += 13;
    plus(SX1, y);
    P.push(`<path class="wire" d="M ${SX1} ${tap1} L ${RAIL1} ${tap1} L ${RAIL1} ${y} L ${SX1 - 11} ${y}" marker-end="url(#arr)"/>`);
    // the residual add is an op like any other: dashed box beside the junction
    addBox(SX1 + 16, y, 'residual add', "residual add — no FLOPs; its output x1 is what the second RMSNorm's backward reads");
    P.push(modeBtn(['x1'], SX1 + 16 + 126 - 30, y - 10));
    tensorChip(['x1'], SX1 + 16, y + 15);
    x1Y = y;
    col1End = y + 46;
    }
    }  // end MLA column

    // the norm2 return rail: right of all of column 1's aux labels, and clear
    // of the tabs' enclosure border (at C2 - 14) when tabs are drawn
    const midX = this.hasAttribute('tabs') ? C2 - 34 : C2 - 16;

    // ---- column 2: the FFN half (MoE machinery, or one wide dense FFN);
    // skipped in only="mla" mode ----
    const nExp = DSV3.topk + DSV3.sharedExperts;   // grouped boxes carry topk/nExp, shared 1/nExp
    // tabs: dense/MoE flip tabs (with the per-block tally) above the FFN column
    const TABS = this.hasAttribute('tabs') && ONLY !== 'mla';
    // kind-pinned tiers keep the ENCLOSURE but drop the tab flaps
    const FLAPS = TABS && !this._noKind;
    const FFN_RIDS = ['router', 'dispatch', 'gate_up', 'swiglu', 'ffn_down', 'combine', 'moe_add'];
    let z = ONLY === 'ffn' ? 36 : 16;
    let encTop = 0, encBot = 0;   // enclosure extent: just the kind-dependent region
    if (ONLY !== 'mla') {
    if (TABS) P.push('__ENC__');  // placeholder: the enclosure, sized after the column is drawn
    // tab shapes at y=ty (26 tall): the ACTIVE tab fuses into the enclosure
    // below it (its fill covers the shared edge, its border stops at it); the
    // inactive tab is a detached grey flap on the enclosure's top edge
    const drawTabs = (ty) => {
      const tab = (x, w, kind, label, sub) => {
        const on = this.kind === kind, r = 6, y0 = ty, y1 = ty + 26;
        const shape = `M ${x} ${y1} L ${x} ${y0 + r} Q ${x} ${y0} ${x + r} ${y0} ` +
          `L ${x + w - r} ${y0} Q ${x + w} ${y0} ${x + w} ${y0 + r} L ${x + w} ${y1}`;
        return `<g data-kind="${kind}" style="cursor:${on ? 'default' : 'pointer'}">` +
          (on
            ? `<path d="${shape} Z" fill="${C('#fcfcfb')}" stroke="none"/>` +
              `<path d="${shape}" fill="none" stroke="${C('#c3c2b7')}"/>`
            : `<path d="${shape} Z" fill="${C('#eeede7')}" stroke="${C('#d8d6cb')}"/>`) +
          `<text x="${x + 10}" y="${y0 + 17}" style="font:600 11px system-ui" fill="${on ? C('#0b0b0b') : C('#898781')}">${label}` +
          `<tspan style="font:10px system-ui" fill="${on ? C('#898781') : C('#a8a69e')}"> ${sub}</tspan></text></g>`;
      };
      if (HIDT < 1) {   // cumulative mode: the plan selector alone carries the kind (tabs fade with the tween)
        P.push(`<g opacity="${(1 - HIDT).toFixed(3)}">` +
          tab(C2 + 42, 148, 'dense', 'dense FFN', `×${DSV3.denseLayers ?? 3} · ${fmtPV(PARAMS.denseFfnBlk)}`) +
          tab(C2 + 198, 168, 'moe', 'MoE FFN', `×${DSV3.layers - (DSV3.denseLayers ?? 3)} · ${fmtPV(this.activeView ? PARAMS.activeMoeFfnBlk : PARAMS.moeFfnBlk)}`) +
          `</g>`);
      }
    };
    const norm2Top = z;
    if (ONLY === 'ffn') {
      // component view: input arrives from the block wiring (post-attention x1);
      // the residual fork and add live there, not here
      P.push(`<text class="oplabel" x="${SX2 + 14}" y="${TABS ? 46 : 12}">x1 (7168) — from the block wiring</text>`);
      wire(SX2, TABS ? 40 : 6, z);
    }
    z = opNode('norm2', 'RMSNorm', C2, z, 'op', pk(DSV3.hidden).trim());
    if (this.kind === 'dense') {
      // dense block: same spine, a single wide FFN — no router, no a2a, no
      // shared column. The column advances through the MoE rows' positions
      // (whitespace where the routing machinery sits, gaps measured on the
      // parallel MoE analysis) so flipping kinds keeps elements in place.
      tensorChip(['norm2'], SX2 + 14, z + 4);
      const spineFrom = z;   // one continuous spine through the whitespace below
      // norm2 gap (same formulas as the MoE branch, + room for the tab row),
      // then whitespace where the routing rows sit: router box (+ top-k micro
      // in detail) + its chip, a2a dispatch + its chip
      const g0 = (DET ? Math.max(38, chipSpace(['norm2']) + 20) : Math.max(22, chipSpace(['norm2']) + 10))
        + (TABS ? (FLAPS ? 36 : 16) : 0);
      if (TABS) { if (FLAPS) drawTabs(z + g0 - 46); encTop = z + g0 - 20; }
      z += g0 + BH + (DET ? 18 : 0) + gapM(['router']) + 22 + gapM(['dispatch']);
      const gTop = z + 3; z += 21;
      wire(SX2, spineFrom, gTop - 3);   // arrow stops above the group, like the MoE rows
      z = mmBox(['ffn_gate_up'], C2, z, ['gate_up'], 'ffn gate/up', `7168 → 2×${DSV3.denseInter}`);
      // dtype tiers draw the SwiGLU-input quantize as its own pill (the GEMM
      // emits bf16; the quantized copy is the stash); the structure/AC tiers
      // hang the stash chip straight off the GEMM (quantize = identity there)
      if (this._ctl.dtype) {
        z = wireOut(['gate_up'], SX2, z, { flat: true });
        z = opNode('quant', 'quantize', C2, z);
      }
      z = wireOut(['quant'], SX2, z);
      z = opNode('swiglu', 'SwiGLU', C2, z);
      z = wireOut(['swiglu'], SX2, z);
      z = mmBox(['ffn_down'], C2, z, undefined, 'ffn down', `${DSV3.denseInter} → 7168`);
      grp(C2, gTop, z + 5, 'dense FFN — every token');
      tensorChip(['ffn_down'], SX2 + 14, z + 9);
      const zc = z + 5;
      // enclosure bottom mirrors the MoE footprint (combine + its chip, and
      // the routed+shared add in detail), so the box doesn't move across flips
      encBot = zc + gapM(['ffn_down']) + 22 + gapM(['combine']) + (DET ? 48 : 4);
      if (ONLY === 'ffn') {
        z = zc + Math.max(22, chipSpace(['ffn_down']) + 10);
      } else {
        // whitespace where the a2a combine sits; the add clamps to col1End,
        // the same row the MoE residual add lands on
        z = Math.max(zc + gapM(['ffn_down']) + 22 + gapM(['combine']) + 13, col1End - 4);
        wire(SX2, zc, z - 11);
        plus(SX2, z);
        addBox(SX2 + 26, z, 'residual add', 'residual add — x1 + the ffn output');
        P.push(lockBtn(SX2 + 26 + 126 - 30, z - 10));
      }
    } else {
    let shBot = 0, shTop = 0;
    const SHX = C2 + 320, shMid = SHX + 22;        // shared-expert mini column; spine down its LEFT, like every column
    const shBox = (name, dims, tip, yy, pc = '', markId = null, dtId = null) => {
      const n = name.includes('gate/up') ? 2 * DSV3.hidden * DSV3.moeInter : DSV3.hidden * DSV3.moeInter;
      P.push(`<g data-op="shared" data-tip="${escAttr(`${tip}\nparameters: ${n.toLocaleString('en-US')}`)}">` +
      `<rect class="box" x="${SHX}" y="${yy}" width="140" height="34" rx="4"/>` +
      `<text class="name" x="${SHX + 6}" y="${yy + 14}">${name}</text>` +
      `<text class="dims" x="${SHX + 6}" y="${yy + 27}">${PONLY ? pc.trim()
        : flatten(dims) + (dtId && this._ctl.dtype && !this._ctl.marks ? '' : pc)}</text></g>`);   // the dtype mirror takes the params' spot (the count stays in the tooltip + grouped box)
      // MIRRORED mark: the shared expert lives inside the grouped node, so
      // its button toggles the same mark (both buttons re-render in sync).
      // The dtype MIRROR takes the same top-right slot in the dtype tier
      // (where no mark button renders — the 140px box has room for one
      // button, and only the full tier would want both; there the grouped
      // boxes carry the dtype lever alone)
      if (markId && this._ctl.marks) P.push(modeBtn([markId], SHX + 140 - 28, yy + 1));
      else if (dtId && this._ctl.dtype)
        // compact, bottom-right: the top row belongs to the name ('shared
        // gate/up' runs to ~x+97); the dims line is shorter, so a 40px
        // mirror clears it (foreignObject buttons are invisible to the
        // linter's text rules — the geometry here is hand-checked)
        P.push(`<foreignObject x="${SHX + 96}" y="${yy + 12}" width="42" height="20">` +
          `<button xmlns="http://www.w3.org/1999/xhtml" class="st dtb" data-dt="${dtId}" style="color:${C(DT_STYLE[dt(dtId)])};width:40px" ` +
          `title="toggle precision: bf16 ⇄ ${FP8K} — mirrors the grouped box (one matmul, two buttons)">${dt(dtId)}</button></foreignObject>`);
    };
    if (!DET) {
      const g0 = Math.max(22, chipSpace(['norm2']) + 10) + (TABS ? (FLAPS ? 36 : 16) : 0);
      tensorChip(['norm2'], SX2 + 14, z + 4);
      wire(SX2, z, z + g0);
      if (TABS) { if (FLAPS) drawTabs(z + g0 - 46); encTop = z + g0 - 20; }
      z += g0;
    } else {
      // the shared expert runs on EVERY token as its own plain GEMMs — fork
      // norm2-out here; its boxes are drawn row-aligned with the routed ones
      tensorChip(['norm2'], SX2 + 14, z + 4);
      const nGap = Math.max(38, chipSpace(['norm2']) + 20) + (TABS ? (FLAPS ? 36 : 24) : 0);
      wire(SX2, z, z + nGap);
      // flapless: a deeper top band — the label row sits ABOVE the shared-
      // expert feed rail (which runs 10px over the router row)
      if (TABS) { if (FLAPS) drawTabs(z + nGap - 46); encTop = z + nGap - (FLAPS ? 20 : 28); }
      shTop = z + nGap - 10;
      P.push(`<circle cx="${SX2}" cy="${shTop}" r="2.5" fill="${C('#898781')}"/>` +
        `<path class="wire" d="M ${SX2} ${shTop} L ${shMid} ${shTop}"/>`);
      z += nGap;
    }
    z = mmBox(['router'], C2, z);
    const gateX = C2 + 306;                    // top-k weights rail — far enough right that elbows clear the arrowheads
    let gateTop = 0;
    if (DET) {
      // the top-k weights are a DEDICATED second output of the top-k block
      // (right edge) — not a duplicated tensor like the residual/shared forks
      gateTop = z + 9;
      // 'biased top-k' scopes the bias to SELECTION; the gating weights are
      // a separate path (renorm of the ORIGINAL scores) — the old
      // '+bias · top-k · scale' read as one pipeline, wrongly implying
      // biased scores become weights
      z = micro('sigmoid · biased top-k · renorm+scale', C2, z, W,
        'two paths from the sigmoid scores: SELECTION adds the learned e_score_correction_bias ' +
        '(group-limited top-8); the GATING WEIGHTS ignore the bias — the original sigmoid scores of ' +
        `the selected 8, renormalized to sum 1, × the routed scaling factor 2.5\nparameters: ${PARAMS.routerBias.toLocaleString('en-US')}`,
        pk(PARAMS.routerBias).trim(), 'router_bias');
      // quant tiers: the label rides the rail's VERTICAL run (right of it,
      // mid-descent — open space there; the static tiers' compressed rows
      // put other rails in that pocket, so they keep the below-pill spot)
      P.push(`<path class="wire" d="M ${C2 + W} ${gateTop} L ${gateX} ${gateTop}"/>` +
        (PONLY || this._ctl.quant ? '' : `<text class="tensor tidle" x="${C2 + 198}" y="${z + 11}">top-k weights · 8</text>`));
      this._gateLbl = (yDisp) => PONLY || !this._ctl.quant ? '' : `<text class="tensor tidle" x="${gateX - 4}" y="${yDisp}" text-anchor="end">top-k weights · 8</text>`;
    }
    z = wireOut(['router'], SX2, z, DET ? { name: 'router state' } : undefined);
    const dispTop = z;
    z = opNode('dispatch', DET ? 'a2a dispatch (permute + comm) → EP group' : 'a2a dispatch → EP group', C2, z, 'comm');
    // the top-k weights are dispatched too: the rail enters the a2a alongside
    // the tokens and re-emerges as the per-expert weights (Megatron: probs in,
    // expert_probs out of hybridep_dispatch)
    if (DET) P.push(
      `<path class="wire" d="M ${gateX} ${gateTop} L ${gateX} ${dispTop + 7} L ${C2 + W + 1} ${dispTop + 7}" marker-end="url(#arr)"/>` +
      `<path class="wire" d="M ${C2 + W} ${dispTop + 16} L ${gateX} ${dispTop + 16}"/>` +
      this._gateLbl(dispTop - 3));   // end-anchored at the rail's DOWN-elbow, just above its entry into the a2a
    z = wireOut(['dispatch'], SX2, z);
    const g2 = z + 3; z += 21;
    const rowG = z;
    z = mmBox(['ffn_gate_up'], C2, z, ['gate_up'], DET ? 'ffn gate/up (grouped ×8)' : undefined);
    if (DET) {
      P.push(`<path class="wire" d="M ${shMid} ${shTop} L ${shMid} ${rowG - 18}" marker-end="url(#arr)"/>` +
        `<text class="grplabel" x="${SHX}" y="${rowG - 6}">shared expert (every token)</text>`);
      shBox('shared gate/up', '7168 → 2×2048',
        'one plain GEMM per token — follows the ffn gate/up mark and dtype (its FLOPs are counted in the grouped strip)', rowG,
        pk(DSV3.hidden * 2 * DSV3.moeInter, false, 'd'), 'gate_up', 'ffn_gate_up');
      if (this._ctl.dtype)
        tensorChip(['gate_up'], shMid + 14, z + 4, { name: 'gate, up (sh)', tdims: '2×2048', frac: 1 / nExp, chip: 'gate_up:sh', flat: true });
    }
    // the SwiGLU-input quantize: dtype tiers draw it as its own pill (the
    // gate/up GEMM emits bf16 — never stashed; the quantized copy is the
    // stash), the structure/AC tiers hang the stash chip straight off the
    // GEMM (quantize = identity under bf16)
    if (this._ctl.dtype) {
      z = wireOut(['gate_up'], SX2, z, DET ? { name: 'gate, up (routed)', frac: DSV3.topk / nExp, flat: true } : { flat: true });
      if (DET) wire(shMid, rowG + 34, z);
      const rowQ = z;
      z = opNode('quant', DET ? 'quantize · for the stash' : 'quantize', C2, z);
      if (DET) {
        micro('quantize', SHX, rowQ, 140, QUANT_TIP, '', 'quant');
        tensorChip(['quant'], shMid + 14, z + 4, { name: 'gate, up (sh)', tdims: '2×2048', frac: 1 / nExp, chip: 'quant:sh' });
      }
      z = wireOut(['quant'], SX2, z, DET ? { name: 'gate, up (routed)', tdims: `${DSV3.topk}×2×2048`, frac: DSV3.topk / nExp } : undefined);
      if (DET) wire(shMid, rowQ + 18, z);
    } else {
      if (DET) tensorChip(['quant'], shMid + 14, z + 4, { name: 'gate, up (sh)', tdims: '2×2048', frac: 1 / nExp, chip: 'quant:sh' });
      z = wireOut(['quant'], SX2, z, DET ? { name: 'gate, up (routed)', tdims: `${DSV3.topk}×2×2048`, frac: DSV3.topk / nExp } : undefined);
      if (DET) wire(shMid, rowG + 34, z);
    }
    // gate-at-swiglu, not gate-at-combine: by linearity the router weights can
    // multiply the swiglu output before the down-proj (one fused kernel,
    // a fused swiglu-and-scale kernel) — this is what makes the expert outputs a pure
    // intermediate instead of a stash for the combine's backward
    if (DET) P.push(`<path class="wire" d="M ${gateX} ${dispTop + 16} L ${gateX} ${z + 13} L ${C2 + W + 1} ${z + 13}" marker-end="url(#arr)"/>`);
    const rowS = z;
    z = opNode('swiglu', DET ? 'SwiGLU · × top-k weight' : 'SwiGLU', C2, z);
    if (DET) {
      micro('SwiGLU (ungated)', SHX, rowS, 140, "follows the routed SwiGLU's mark — one graph node covers both", '', 'swiglu');
      tensorChip(['swiglu'], shMid + 14, z + 4, { name: 'swiglu out (sh)', tdims: '2048', frac: 1 / nExp, short: true });
    }
    z = wireOut(['swiglu'], SX2, z, DET ? { name: 'swiglu out (routed)', tdims: `${DSV3.topk}×2048`, frac: DSV3.topk / nExp } : undefined);
    if (DET) wire(shMid, rowS + 18, z);
    const rowD = z;
    z = mmBox(['ffn_down'], C2, z, undefined, DET ? 'ffn down (grouped ×8)' : undefined);
    if (DET) {
      shBox('shared down', '2048 → 7168',
        'one plain GEMM per token — follows the ffn down mark and dtype; its output joins the routed sum', rowD,
        pk(DSV3.moeInter * DSV3.hidden, false, 'd'), 'ffn_down', 'ffn_down');
      tensorChip(['ffn_down'], shMid + 14, z + 4, { name: 'shared out', tdims: '7168', frac: 1 / nExp });
      shBot = rowD + 34;
    }
    // group tally like the MLA label/tabs; the routing description (top-8 of
    // 256) lives on the router/dispatch boxes, not here
    {
      // local: this rank hosts 256/EP experts
      const nR = LOCAL ? DSV3.routedExperts / EPn
        : this.activeView ? DSV3.topk : DSV3.routedExperts;   // fired vs resident
      // multiplied sizes fold ×N into the box numbers, so the ×N leaves the
      // label too — showing both would read as "multiply again" (overcount)
      grp(C2, g2, z + 5, DET
        ? (CUM ? `routed experts${LOCAL ? ` (${nR}/rank)` : ''} · ${fmtPV(PARAMS.expert * nR * KMUL, 'e')}${facTxt('e')}`
          : FLAT ? `routed experts${LOCAL ? ` (${nR}/rank)` : ''} · ${fmtPV(PARAMS.expert * nR, 'e')}`
                 : `routed experts ×${nR} · ${fmtPV(PARAMS.expert, 'e')}`)
        : (CUM ? `experts · ${fmtPV(PARAMS.expert * (nR + DSV3.sharedExperts) * KMUL, 'e')}`
          : FLAT ? `experts · ${fmtPV(PARAMS.expert * (nR + DSV3.sharedExperts), 'e')}`
                 : `experts ×${nR + DSV3.sharedExperts} · ${fmtPV(PARAMS.expert, 'e')}`));
    }
    z = wireOut(['ffn_down'], SX2, z + 5);
    z = opNode('combine', DET ? 'a2a combine (comm + unpermute · sum)' : 'a2a combine (weighted by router)', C2, z, 'comm');
    // combine's output wire runs all the way into the x2 add; the add itself
    // is kept below column 1's residual box + x1 chip, so the x1 → x2 rail
    // turns right in clear space instead of crossing them
    tensorChip(['combine'], SX2 + 14, z + 4);
    const zc = z;                                  // combine box bottom
    if (!DET) {
      encBot = zc + Math.max(22, chipSpace(['combine']) + 10) + 4;
      if (ONLY === 'ffn') {
        z = zc + Math.max(22, chipSpace(['combine']) + 10);
      } else {
      z = Math.max(z + Math.max(22, chipSpace(['combine']) + 10) + 13, col1End - 4);
      plus(SX2, z);
      wire(SX2, zc, z - 11);
      addBox(SX2 + 26, z, 'residual add',
        'one fused add kernel (Megatron: add_shared_and_residual) — routed output + shared output + residual x1');
      P.push(lockBtn(SX2 + 26 + 126 - 30, z - 10));
      }
    } else {
      // pedagogical split: (routed + shared) first, then the residual add.
      // Megatron fuses all three into one add_shared_and_residual kernel.
      // (The routed+shared sum is INTERNAL to the MoE FFN, so the component
      // view keeps it; only the residual add belongs to the block wiring.)
      const zA = zc + Math.max(22, chipSpace(['combine']) + 10) + 34;
      wire(SX2, zc, zA - 11);
      plus(SX2, zA);
      P.push(`<path class="wire" d="M ${shMid} ${shBot} L ${shMid} ${zA} L ${SX2 + 11} ${zA}" marker-end="url(#arr)"/>`);
      addBox(SX2 + 26, zA - 24, 'add — routed + shared',
        'routed + shared expert outputs — its backward needs nothing; a ↻ mark replays it (and must pull its inputs into the stash: honestly wasteful). Megatron fuses this with the residual add (add_shared_and_residual)', 178);
      P.push(modeBtn(['moe_add'], SX2 + 26 + 178 - 30, zA - 34));
      encBot = zA + 14;   // the routed+shared add is MoE-internal — inside the box
      if (ONLY === 'ffn') {
        z = zA;
      } else {
      const zB = Math.max(zA + 34, col1End - 4);
      wire(SX2, zA + 9, zB - 11);
      plus(SX2, zB);
      addBox(SX2 + 26, zB, 'residual add', 'residual add — x1 + the ffn output');
      P.push(lockBtn(SX2 + 26 + 126 - 30, zB - 10));
      z = zB;
      }
    }
    }  // end MoE column
    if (ONLY === 'ffn') {
      // component view: output hands off to the block wiring's residual add
      P.push(`<line class="wire" x1="${SX2}" y1="${z + 9}" x2="${SX2}" y2="${z + 24}" marker-end="url(#arr)"/>` +
        `<text class="tensor tidle" x="${SX2 + 8}" y="${z + 24}">→ ⊕ residual add (block wiring)</text>`);
    } else {
    // block output: a short down arrow out of the second residual add (= the next block's x0)
    P.push(`<line class="wire" x1="${SX2}" y1="${z + 9}" x2="${SX2}" y2="${z + 26}" marker-end="url(#arr)"/>` +
      `<text class="tensor tidle" x="${SX2 + 8}" y="${z + 24}">x2 (block output)</text>`);
      P.push(`<path class="wire" d="M ${SX1} ${x1Y + 9} L ${SX1} ${z} L ${SX2 - 11} ${z}" marker-end="url(#arr)"/>`);
      // branch off the bottom rail up to norm2 (single output from the x1 add)
      P.push(`<circle cx="${midX}" cy="${z}" r="2.5" fill="${C('#898781')}"/>` +
        `<path class="wire" d="M ${midX} ${z} L ${midX} 6 L ${SX2} 6 L ${SX2} ${norm2Top}" marker-end="url(#arr)"/>`);
    }
    if (TABS) {
      // the enclosure the active tab fuses into — scoped to the kind-dependent
      // region (past norm2, before the residual add), fixed extent regardless
      // of kind (the MoE footprint) so it doesn't move across flips. In
      // cumulative mode the tabs are hidden, so the border goes too (the
      // reserved space stays — flip stability).
      {
        const x0 = C2 - 14, x1 = x0 + (DET ? 500 : 385) + 14, y0 = encTop, y1 = encBot, r = 8;
        if (!FLAPS) {
          // kind-pinned: a plain enclosure with a static label, and the
          // REGION TOGGLE — set every MoE-FFN mark at once. Idiomatically
          // tristate: ↻ none / ↻ all / mixed, where 'mixed' both DISPLAYS an
          // in-between state and REMEMBERS the last one (the custom-chip
          // pattern), so 'all' never destroys a hand-tuned composition.
          const rx0 = Math.max(x1 - 210, C2 + 348);   // right of the shared-expert spine (shMid = C2+342)
          const regionCtl = regionToggle(FFN_RIDS, 'ffnMixed', rx0, y0 + 2, x1 - rx0 - 4);
          P[P.indexOf('__ENC__')] =
            `<rect x="${x0}" y="${y0}" width="${x1 - x0}" height="${y1 - y0}" rx="${r}" fill="${C('#fcfcfb')}" stroke="${C('#c3c2b7')}"/>` +
            `<text class="grplabel" x="${x0 + 56}" y="${y0 + 11}">MoE FFN ${this.getAttribute('ctx') ? '' : `\u00d7${DSV3.layers - (DSV3.denseLayers ?? 3)} `}\u00b7 ${fmtPV(PARAMS.moeFfnBlk)}</text>` +
            regionCtl;
        } else {
          // active tab footprint: the outline leaves a gap there instead of an
          // opaque eraser (which bleeds the border through when the tab fades)
          const [tx, tw] = this.kind === 'dense' ? [C2 + 42, 148] : [C2 + 198, 168];
          P[P.indexOf('__ENC__')] = HIDT >= 1 ? '' :
            `<g opacity="${(1 - HIDT).toFixed(3)}">` +
            `<rect x="${x0}" y="${y0}" width="${x1 - x0}" height="${y1 - y0}" rx="${r}" fill="${C('#fcfcfb')}" stroke="none"/>` +
            `<path d="M ${tx + tw} ${y0} L ${x1 - r} ${y0} Q ${x1} ${y0} ${x1} ${y0 + r} L ${x1} ${y1 - r} ` +
            `Q ${x1} ${y1} ${x1 - r} ${y1} L ${x0 + r} ${y1} Q ${x0} ${y1} ${x0} ${y1 - r} ` +
            `L ${x0} ${y0 + r} Q ${x0} ${y0} ${x0 + r} ${y0} L ${tx} ${y0}" fill="none" stroke="${C('#c3c2b7')}"/></g>`;
        }
      }
    }
    }  // end FFN column (skipped in only="mla" mode)
    const col2End = ONLY === 'mla' ? 0 : z + 42;   // room for the add label under the plus

    // ---- head row (scope="model" only: other scopes show the block or a
    // column alone, making no claims about the surrounding stack) ----
    let h = Math.max(col1End, col2End) + 10;
    let lmH = -20;
    // local: the fit bar — this rank's WHOLE-STAGE bytes as a flush stacked
    // bar (memory-bar segment colors) against the 80 GiB H100 capacity tick.
    // Segments follow the legend checkboxes and lerp with the knob tween;
    // the total label snaps. Activations approximate mixed-kind stages with
    // the current kind's per-layer stash.
    // per-comp byte tallies for the margin legend (target values, snap):
    // local = this rank's whole stage (matches the bar); otherwise the
    // current view's block ×N
    if (OPTIM && !LOCAL) {
      const kindP = this.kind === 'dense' ? PARAMS.denseBlock : PARAMS.moeBlock;
      const m2 = CUM ? KMUL : 1;
      this._segTotals = COMPS.map((c) => kindP * c.bpp * m2);
      if (CONS) this._segTotals.push(ana.savedBytes * TOK * m2);
    }
    if (LOCAL) {   // the fit bar renders in its own row under the controls (this._barHtml)
      const cap = HARDWARE[HWk].memGB * 2 ** 30;   // the capacity yardstick (GiB) of this rank's GPU
      const moeExp = PARAMS.expert * DSV3.routedExperts;
      // activation bytes are priced BY KIND: the dense front stashes at the
      // dense rate (no router state, no dispatched tokens), the MoE layers
      // at theirs — same marks/recipe for both (cells D1/D2). Vocab-side
      // activations charge ×1 microbatch (D3: the head's logits + fp32 loss
      // live only across that microbatch; the embed residual is chunk 0's
      // x0) — memory.js's head/embed convention.
      const anaD = this._anaMemo.anaD ?? ana;
      // env assembly + rate decompositions live in cells.js (cellsEnv):
      // the whole cell graph is constructible in Node from plain state
      const envOf = (S) => cellsEnv(S, ana, anaD, this._anaMemo.anaMF, this._anaMemo.anaDF);
      const segB = (S) => {
        const { get } = buildCells(envOf(S));
        return [get('W1'), get('G1'), get('O1'), get('A1')];
      };
      // per component: [experts, non-expert blocks, emb+lm head] sub-cells —
      // the solo breakdown and its pin factors ride these
      const partsFor = (S) => {
        const { get } = buildCells(envOf(S));
        return [['W2', 'W3', 'W4'], ['G2', 'G3', 'G4'], ['O2', 'O3', 'O4']].map((ids) => ids.map(get))
          .concat([ACT_BUCKETS.map((_, k) => get(`A${k + 2}`))]);
      };
      this._cells = (opts) => buildCells({ ...envOf(Snow), ...opts });   // tooltips (exact) + the bound <dsv3-sheet> (may pass simplify)
      this._segParts = partsFor(Snow);
      const nowB = segB(Snow);
      this._segTotals = nowB;
      // The chart is UNSTACKED rows on a FIXED log₂ axis; the row labels are
      // the legend (names in the gutter, click to solo) and the ABSOLUTE
      // values sit at the bar ends, where the log axis gives them meaning.
      // Soloing a param component accordions its breakdown open beneath it.
      // Everything below builds the LAYOUT for the CURRENT state only —
      // animation is blendFit's job, never the model's.
      const { x0, bw, lo: LO, hi: HI } = BAR_GEO;   // 256 MiB … 16 TiB, 16 doublings
      const rowH = FIT_ROWH, topY = 14;
      const px = (b) => x0 + Math.max(0, Math.min(1, (Math.log2(Math.max(b, 1)) - LO) / (HI - LO))) * bw;
      const on = [...COMPS.map((c) => this[c.prop] ? 1 : 0), this.showActs ? 1 : 0];
      const colors = [...COMPS.map((c) => C(c.color)), C('#eda100')];
      // the in-flight count the cells charge THIS kind's layers (P6 / P6d — the byte-peak moment under interleaving)
      const IF2 = buildCells(envOf(Snow)).get(this.kind === 'dense' && SCHED === 'interleaved' ? 'P6d' : 'P6');
      const names = ['weights', `gradients (${GRADB === 2 ? 'bf16' : 'fp32'})`, 'optimizer states', `activations ×${fmtIF(IF2)}mb`];
      const totalN = nowB.reduce((t2, b) => t2 + b, 0);
      const pin = this._pinCfg;
      const pinTotal = pin ? pin.segs.reduce((t2, b) => t2 + b, 0) : 0;
      const badge = (cur, base) => pin ? facBadge(cur, base) : '';
      // a ghost only where there IS a delta — a coincident ghost just
      // serrates an unchanged bar (same 5% log threshold as the badge)
      const ghostOf = (cur, pinB, color) => pinB && Math.abs(Math.log2((cur || 1) / pinB)) >= 0.05
        ? { px: px(pinB), op: 0.7, color, true: pinB } : null;
      // the accordion: a soloed param component opens its breakdown below
      // it; snapshot 'parts' figures open EVERY visible component at once
      const onCount = on.reduce((t2, o2) => t2 + o2, 0);
      const sIdx = onCount === 1 && on.indexOf(1) <= 3 ? on.indexOf(1) : -1;
      const SNAP2 = this.hasAttribute('snapshot');
      const ALLPARTS = SNAP2 && this.hasAttribute('parts');
      const openRows = ALLPARTS ? [0, 1, 2, 3].filter((j) => on[j]) : sIdx >= 0 ? [sIdx] : [];
      // ALLPARTS forces EVERY sub-row, the acts 'other' catch-all included —
      // a bucket hitting zero mid-deck must hold its row (0 B) so the chart
      // never resizes between steps
      const partIdxsOf = (j) => (this._segParts[j] ?? [])
        .map((b, k) => b > 0 || (pin?.parts?.[j]?.[k] ?? 0) > 0 || ALLPARTS ? k : -1)
        .filter((k) => k >= 0);
      const subAbove = (i) => openRows.reduce((t2, j) => t2 + (j < i ? partIdxsOf(j).length * rowH : 0), 0);
      const subHTot = openRows.reduce((t2, j) => t2 + partIdxsOf(j).length * rowH, 0);
      // snapshots drop the off components' rows outright — a dimmed name you
      // can't click is a dead affordance in a figure (interactive views keep
      // them: they're the solo/restore legend). The total keeps full mass.
      const NOTOT = SNAP2 && this.hasAttribute('nototal');   // intro beats: the shading says "doesn't fit"; the tally beat lands the whole
      let vp = 0;
      const posOf = [...on.map((o2) => SNAP2 && !o2 ? -1 : vp++), NOTOT ? -1 : vp++];
      const yOf = (i) => topY + posOf[i] * rowH + subAbove(i) + (i === 4 ? 4 : 0);
      const axisY = topY + vp * rowH + subHTot + 5 + 4;
      const rows = [];
      for (let i = 0; i < 4; i++) {
        if (posOf[i] < 0) continue;
        const abs = nowB[i];
        const b = on[i] ? abs : 0;   // hidden components park at the axis floor
        // a part filter turns the soloed row into a stack one level down:
        // grey = unselected parts, colored tip = the selected part. The pair
        // is CANONICAL (base zero-width without a filter) so every comp row
        // blends against every other form without a representation switch.
        const soloSel = i === sIdx && this.partSel != null;
        const selSum = soloSel ? this._segParts[i].reduce((a2, b2, k) => a2 + (this.partSel === k ? b2 : 0), 0) : b;
        const grey = soloSel ? Math.max(0, b - selSum) : 0;
        const pinB = pin ? pin.segs[i] : 0;
        rows.push({ key: `seg:${i}`, type: 'comp', id: String(i), y: yOf(i), op: 1,
          nameOp: on[i] ? 1 : 0.35, name: names[i], color: colors[i],
          cell: ['W1', 'G1', 'O1', 'A1'][i],   // the cell whose formula this row's tooltip shows
          prop: i < COMPS.length ? COMPS[i].prop : 'showActs', abs,
          segs: [
            { key: 'base', x0, x1: grey ? px(grey) : x0, color: C('#c3c2b7'), op: 1 },
            { key: 'tip', x0: grey ? px(grey) + 1 : x0, x1: px(b), color: colors[i],
              op: on[i] ? 1 : 0.35, bar: String(i), true: b },
          ],
          ghost: on[i] ? ghostOf(abs, pinB, colors[i]) : null,
          val: on[i] ? { x: px(b) + 5, op: 1, text: `${fmtBytes(abs)}${badge(abs, pinB)}`,
            true: abs, pin: pinB || '' } : null });
        // the accordion sub-rows open right below their component's row:
        // param components break down by sharding class (clickable — the
        // part filter); the acts row breaks down PER OP, like the chips
        if (openRows.includes(i) && partIdxsOf(i).length) {
          const names2 = i === 3 ? ACT_BUCKETS.map((b2) => b2.label) : ['experts', 'non-expert', 'emb + lm head'];
          const clickable = i < 3;
          for (const [k3, k2] of partIdxsOf(i).entries()) {
            const bP = this._segParts[i][k2];
            const pinP0 = pin?.parts?.[i]?.[k2];
            const pinP = pinP0 && Math.abs(Math.log2((bP || 1) / pinP0)) >= 0.05 ? pinP0 : 0;
            rows.push({ key: `part:${i}:${k2}`, type: 'part', id: `part:${i}:${k2}`,
              y: yOf(i) + rowH + k3 * rowH + 2, nameOp: 1,
              op: 0.4 + 0.6 * (clickable && i === sIdx ? psel(this.partSel ?? null, k2) : 1),   // dim unselected parts
              cell: i === 3 ? `A${k2 + 2}` : 'WGO'[i] + (k2 + 2),   // the sub-cell this row IS
              name: names2[k2], color: colors[i], part: clickable ? k2 : null,
              segs: [{ key: 'bar', x0, x1: px(bP), color: colors[i], op: 0.55, bar: `part:${i}:${k2}`, true: bP }],
              ghost: pinP ? { px: px(pinP), op: 0.7, color: colors[i], true: pinP } : null,
              val: { x: px(bP) + 5, op: 1, text: `${fmtBytes(bP)}${badge(bP, pinP)}`,
                true: bP, pin: pinP || '' } });
          }
        }
      }
      if (!NOTOT) {
        // the TOTAL row never resizes: it always shows ALL components. Under
        // a solo it becomes a stacked bar — grey "other" base + the on
        // components as colored tips — so the visible colored width IS the
        // factor you could gain by optimizing only what's highlighted. All
        // on, the tips park zero-width at the bar end (they render nothing
        // and blend smoothly into the stacked form).
        const topSum = nowB.reduce((a2, b2, j) => a2 + (on[j] ? b2 : 0), 0);
        const stacked = !on.every(Boolean) && totalN - topSum > totalN * 0.001;
        let acc = stacked ? totalN - topSum : totalN;
        const tips = [0, 1, 2, 3].map((j) => {
          const w2 = stacked && on[j] ? nowB[j] : 0;
          const t0 = px(acc) + 1; acc += w2;
          return { key: `tip:${j}`, x0: t0, x1: px(acc), color: colors[j], op: 1 };
        });
        rows.push({ key: 'total', type: 'comp', id: 'total', y: yOf(4), op: 1, nameOp: 1,
          name: 'total', color: C('#52514e'), prop: null, abs: totalN, cell: 'T1',
          segs: [{ key: 'base', x0, x1: px(stacked ? totalN - topSum : totalN),
            color: stacked ? C('#c3c2b7') : C('#52514e'), op: 1,
            bar: stacked ? null : 'total', true: stacked ? null : totalN }, ...tips],
          ghost: ghostOf(totalN, pinTotal, C('#898781')),
          val: { x: px(totalN) + 5, op: 1, text: `${fmtBytes(totalN)}${badge(totalN, pinTotal)}`,
            true: totalN, pin: pinTotal || '' } });
      }
      const SHOWLBL = pin && !(SNAP2 && this.getAttribute('knobs'));
      const L1 = {
        // header: PP1 needs no locus (the prose owns the framing); a
        // pipelined chart names whose bytes these are — one GPU's stage
        hdr: PPn === 1 ? 'logarithmic:' : `one GPU, rank ${STG} of PP${PPn} (logarithmic):`,
        axisY, HB: axisY + 38, capPx: px(cap), capLbl: `${HARDWARE[HWk].memGB} GiB (${HW_SHORT[HWk]})`,   // +38: the distances legend band (shared with the save label)
        facs: facsOf(this.getAttribute('facs')),
        unit: !this.hasAttribute('barsonly') && !SNAP2 ? `= ${fmtBytes(PB_UNIT * 2)} / square` : null,
        lbl: SHOWLBL ? `saved: ${pin.label}` : null, rows,
      };
      const F = this._ftween;
      const L = F ? blendFit(F.L0, L1, F.t) : L1;
      this._fitL = L;   // what's on screen NOW — the origin of the next transition
      this._barHtml = fitSvg(L);
    }
    if (SCOPE === 'model') {   // the surrounding stack: ×61 rule, final norm, lm head, loss
      P.push(`<line class="wire" x1="${C1 - 20}" y1="${h}" x2="${C2 + W + 20}" y2="${h}" stroke-dasharray="3 3"/>`);
      P.push(`<text class="grplabel" x="${C1 - 20}" y="${h - 5}">× 61 blocks, then:</text>`);
      h += 10;
      const lm = MATMULS.find(m => m.id === 'lm_head');
      P.push(`<rect class="op" x="${C1}" y="${h + 6}" width="150" height="22" rx="11"/>` +
        `<text class="oplabel" x="${C1 + 12}" y="${h + 21}">final RMSNorm<tspan class="dims"> (${fmtP(DSV3.hidden)})</tspan></text>`);
      P.push(`<line class="wire" x1="${C1 + 150}" y1="${h + 17}" x2="${C1 + 180}" y2="${h + 17}" marker-end="url(#arr)"/>`);
      const lmFlops = 2 * DSV3.hidden * DSV3.vocab / (this.view === 'combined' ? this.dispLayers : 1);
      const lmRows = this._ctl.quant ? barRows(lmFlops, dt('lm_head'), 224, dtPm('lm_head')) : 0;
      lmH = (BQ ? 38 : 34) + lmRows * 6;   // lm-head text sits 2px lower than mmBox text
      P.push(`<g data-op="lm_head" data-tip="${escAttr(`${fmtNum(lmFlops)} FLOP/token\n${lm.dimsNote}\nparameters: ${PARAMS.embed.toLocaleString('en-US')}`)}">` +
        `<rect class="box" x="${C1 + 184}" y="${h}" width="240" height="${lmH}" rx="4"/>` +
        `<text class="name" x="${C1 + 192}" y="${h + 14}">${lm.label}</text>` +
        `<text class="dims" x="${C1 + 192}" y="${h + 28}">${PONLY ? pstr('lm_head').trim() : flatten(lm.dims) + pstr('lm_head')}</text></g>` + dtBtn('lm_head', C1 + 184 + 240 - 58, h + 7));
      flopBar(C1 + 192, h + 33, lmFlops, dt('lm_head'), 224, dtPm('lm_head'));
      P.push(`<line class="wire" x1="${C1 + 424}" y1="${h + 17}" x2="${C1 + 454}" y2="${h + 17}" marker-end="url(#arr)"/>`);
      P.push(`<rect class="op" x="${C1 + 458}" y="${h + 6}" width="140" height="22" rx="11"/>` +
        `<text class="oplabel" x="${C1 + 470}" y="${h + 21}">softmax / loss</text>`);
    }

    // ---- per-layer FLOP tally: fwd + bwd + recompute replay as full-width
    // RIBBONS in the figure's one time unit (bf16 fwd+bwd spans the runway).
    // Headed by the dynamic stash readout — the caption is gone, but the
    // stashed-bytes number reacts to every knob, so it lives in the widget.
    const T = [];
    const eq = (n) => flopEq(n.flopsTok, opDt(n.id));
    const opDtP = (id) => {   // the op's PREVIOUS dtype (tween endpoint)
      const d2 = opDt(id);
      return d2 === 'vector' ? d2 : (dtPm(id === 'gate_up' ? 'ffn_gate_up' : id) ?? d2);
    };
    const eqP = (n) => flopEq(n.flopsTok, opDtP(n.id));
    const fwdOps = Object.values(ana.byId).filter(n => n.flopsTok > 0);
    const fwdEq = fwdOps.reduce((t, n) => t + eq(n), 0);
    const replayOps = fwdOps.filter(n => ana.replayed.has(n.id));
    const replayEq = replayOps.reduce((t, n) => t + eq(n), 0);
    let ty = 2;
    if (this._ctl.quant) {
      const M2b = this.dispLayers * this.dispInflight * TOK;
      const xt = this.getAttribute('xtag');
      const totNow = ana.savedBytes * M2b;
      T.push(`<text class="name" x="0" y="${ty + 10}">stashed for backward: ` +
        `${(ana.savedBytes / 1024).toFixed(0)} KiB/token·layer × ${this.dispLayers} layers` +
        ` × ${this.dispInflight} in flight × ${TOK} tokens${TOK !== 4096 ? ' (4096 ÷ TP)' : ''} = ${(totNow / 2 ** 30).toFixed(1)} GiB` +
        (xt ? `<tspan class="dims"> (${esc(xt)})</tspan>` : '') + '</text>');
      ty += 18;
      // the ghost baselines THIS widget's lever at its do-nothing setting,
      // HOLDING the other lever where the reader put it: the dtype tier
      // ghosts all-bf16 at the CURRENT recompute policy (so it moves when
      // the policy segment moves — the overhang isolates what precision
      // alone bought); every other tier ghosts the untreated anchor
      // (recompute none · all-bf16 · no fp8ᵀ). The ▼×N badge is the
      // lever's exact factor. Width lerps in pixel space (rule 9).
      const DT_TIER = this.getAttribute('controls') === 'dtype';
      const anchorOf = (mks) => analyze(blockGraph(this.kind, DSV3, resolveMatmuls({ recipe: 'bf16' }), 4096),
        DT_TIER ? mks : RECOMPUTE_PRESETS.none, false).savedBytes * M2b;
      const anchor = anchorOf(this.marks);
      // a policy flip moves the anchor too — its edge lerps like the bar
      const anchorPx0 = anchorOf(VQ?.prev?.marks ?? this.marks);
      // the total: a SOLID LINEAR bar over a unit RULER (minor = 1 GiB,
      // major = 8 GiB) — length-first reading, countability on the axis.
      // The AC section lives in the near-fitting regime, exactly where
      // linear counting earns its keep. Ghost = dashed amber OVERHANG only
      // (the stretch the levers shaved off); a dashed tick if the stash
      // ever exceeds the untreated anchor. Width lerps (rule 9).
      // the capacity line follows the section's hardware (ctx.hw, else the
      // instance's); the ruler's px/GiB shrinks so a Blackwell-sized cap
      // still lands on the runway (80 GiB at 6 px = 480 px; 276 GiB at 2 px)
      const HWq = JSON.parse(this.getAttribute('ctx') ?? '{}').hw ?? this.hw ?? 'h100';
      const capGiB = HARDWARE[HWq].memGB, PXG = capGiB > 100 ? 2 : 6, MINOR = PXG === 6 ? 1 : 4;
      const pxT = (b) => TB_X + b / 2 ** 30 * PXG;
      const totPx = lerpQ(pxT((anaP?.savedBytes ?? ana.savedBytes) * M2b), pxT(totNow));
      const cy0 = ty + 3;
      T.push(`<text class="dims" x="0" y="${cy0 + 10}">total</text>`);
      const aPx = lerpQ(pxT(anchorPx0), pxT(anchor));
      if (aPx > totPx + 1)
        T.push(`<rect x="${totPx.toFixed(1)}" y="${cy0 + 2}" width="${(aPx - totPx).toFixed(1)}" height="9" fill="none" stroke="${C('#d19023')}" stroke-dasharray="2 2"/>`);
      else if (aPx < totPx - 1)
        T.push(`<line x1="${aPx.toFixed(1)}" y1="${cy0 - 1}" x2="${aPx.toFixed(1)}" y2="${cy0 + 13}" stroke="${C('#d19023')}" stroke-dasharray="2 2"/>`);
      T.push(`<rect x="${TB_X}" y="${cy0 + 2}" width="${(totPx - TB_X).toFixed(1)}" height="9" fill="${C('#eda100')}" data-true="${totNow}"/>`);
      const rext = Math.max(totPx, aPx, pxT((capGiB + 8) * 2 ** 30)) - TB_X + 6;
      const ry = cy0 + 14;
      T.push(`<line x1="${TB_X}" y1="${ry}" x2="${(TB_X + rext).toFixed(1)}" y2="${ry}" stroke="${C('#c3c2b7')}" stroke-width="1"/>`);
      for (let u = 0; u * MINOR * PXG <= rext; u++) {
        const x = TB_X + u * MINOR * PXG, major = u % 8 === 0;
        T.push(`<line x1="${x}" y1="${ry}" x2="${x}" y2="${ry + (major ? 5 : 2.5)}" stroke="${C('#c3c2b7')}" stroke-width="1"/>`);
        if (major && u > 0) T.push(`<text x="${x}" y="${ry + 14}" text-anchor="middle" font-size="8.5" fill="${C('#898781')}">${u * MINOR}</text>`);
      }
      T.push(`<text class="dims" x="${(TB_X + rext + 10).toFixed(1)}" y="${ry + 14}">GiB · minor tick = ${MINOR} GiB · LINEAR</text>`);
      const capX = pxT(capGiB * 2 ** 30);
      // the badge rides the bar end, but never under the 80 GiB cap label;
      // bar = ghost edge already says untreated — no fallback text needed
      let bdgX = Math.max(totPx, aPx) + 8;
      if (bdgX > capX - 4 && bdgX < capX + 44) bdgX = capX + 44;
      const bdg = facBadge(totNow, anchor);
      if (bdg) T.push(`<text class="dims" x="${bdgX.toFixed(1)}" y="${cy0 + 10}">${bdg}</text>`);
      T.push(`<line x1="${capX}" y1="${cy0 - 2}" x2="${capX}" y2="${cy0 + 14}" stroke="${C('#d03b3b')}"/>` +
        `<text class="dims" x="${capX + 3}" y="${cy0 + 1}" fill="${C('#d03b3b')}">${capGiB} GiB</text>`);
      ty = cy0 + 32;
    }
    T.push(`<text class="grplabel" x="0" y="${ty + 9}">per-layer compute as TIME at ${HWPn} peak — one picket ≈ ${PICKET_US} µs per 4096-token microbatch (10 MFLOP/token at the bf16 rate):</text>`);
    ty += 14;
    const DT_ORDER = { bf16: 0, e5m6: 1, e4m3: 2, mxfp8: 2, fp32: 3 };
    const ribbon = (label, list, wOf, num, comm, cOv, traffic) => {
      T.push(`<text class="dims" x="0" y="${ty + 9}">${label}</text>`);
      let cx = TB_X, cy = ty + 3;
      const sorted = [...list].sort((p, q) => (DT_ORDER[opDt(p.id)] ?? 3) - (DT_ORDER[opDt(q.id)] ?? 3));
      // picket runs at the in-box unit (the fwd ribbon = the boxes' pickets
      // laid end to end); per-op counts Bresenham'd over the exact cumulative
      const per = Math.floor(TB_AVAIL / 3);
      let cum = 0, drawn = 0;
      const lite = cOv === 'light';
      for (const n of sorted) {
        cum += wOf(n) / FUNIT;
        const color = !cOv || lite ? barColor(opDt(n.id), opDtP(n.id)) : cOv;
        for (const upto = Math.round(cum); drawn < upto; drawn++)
          T.push(`<rect x="${TB_X + (drawn % per) * 3}" y="${cy + Math.floor(drawn / per) * 7}" width="2" height="5" fill="${color}"${lite ? ' fill-opacity="0.55"' : ''}/>`);
      }
      cx = TB_X + (drawn % per) * 3; cy += Math.floor(drawn / per) * 7;
      T.push(`<text class="dims" x="${(cx + 10).toFixed(1)}" y="${cy + 6}">${num}</text>`);
      // costs in FOREIGN currencies ride the ribbon as pills, named not
      // priced — no pickets, since the ruler meters only the GEMM floor
      let px = cx + 10 + num.length * 5.2 + 8;
      const pill = (txt, tip, bg, stroke, ink, sc = 1) => {
        const cw = (txt.length + 2) * 4.7 + 16;
        if (px + cw > 1080) { px = TB_X + 10; cy += 17; }   // never clip: wrap under the ribbon
        // sc: eased presence — scales about the pill's left-middle anchor
        const tf = sc === 1 ? '' : ` transform="translate(${px.toFixed(1)} ${cy + 3}) scale(${sc.toFixed(3)}) translate(${(-px).toFixed(1)} ${(-cy - 3).toFixed(1)})" opacity="${sc.toFixed(3)}"`;
        T.push(`<g data-tip="${escAttr(tip)}"${tf}><rect x="${px.toFixed(1)}" y="${cy - 4.5}" width="${cw.toFixed(1)}" height="15" rx="7.5" fill="${bg}" stroke="${stroke}"/>` +
          `<text class="dims" x="${(px + 7).toFixed(1)}" y="${cy + 6}" style="fill:${ink}">+ ${txt}</text></g>`);
        px += cw + 6;
      };
      if (comm?.length)     // replayed a2a wears the diagram's violet comm pill
        pill(`a2a ${comm.join(' + ')}`,
          'communication, not FLOPs — the replay re-runs the all-to-all; its exposed cost depends on overlap, so no number is claimed.',
          C('#f3f1fb'), C('#6b5bd2'), C('#4636a3'));
      if (traffic)          // HBM traffic (quantization round trips): bronze — bytes on the move
        pill(traffic.txt, traffic.tip, C('#f8f2e6'), C('#8c5a19'), C('#6f4712'), traffic.s ?? 1);
      ty = cy + 13;
    };
    const wFwd = (n) => lerpQ(eqP(n), eq(n));
    // replay membership lerps too: an op entering the replay set pours in
    // from zero, a leaving one drains out
    const wRep = (n) => lerpQ(anaP?.replayed.has(n.id) ? eqP(n) : 0, ana.replayed.has(n.id) ? eq(n) : 0);
    // the e4m3ᵀ trade's OTHER side, priced by the fusion rule: a pill only
    // where there is NO PRODUCER TO FUSE INTO. OFF = DeepSeek's convention:
    // backward re-quantizes every wgrad-read fp8 stash — a pure HBM round
    // trip over a COLD stash (and fusing the transpose into the wgrad
    // prologue loses to tile reuse), so a fusion-independent floor exists
    // and the pill claims it. ON pays NO pill: the second orientation rides
    // the forward quantize kernel's epilogue — the extra write is real but
    // belongs to the same deliberately-unpriced class as every other stash
    // write; its cost story is the +GiB already on the bar.
    // presence EASES on ANY cause of appearance/disappearance (the ᵀ toggle,
    // a recipe flip emptying the would-dual set, a recompute change): the
    // pill is computed for BOTH tween endpoints from their own snapshots,
    // scale lerps between existence, and the surviving side supplies the
    // content (numbers snap, per convention)
    const pillFor = (mm2, mks, transp) => {
      if (transp) return 0;
      const cf = analyze(blockGraph(this.kind, DSV3, mm2, 4096), mks, true);
      return [...cf.dual].reduce((t2, i2) => t2 + (cf.byId[i2]?.outBytes ?? 0), 0) * TOK;
    };
    const tbNow = pillFor(this.matmuls, marks, this.transposed);
    const tbPrev = VQ?.prev?.mm ? pillFor(VQ.prev.mm, VQ.prev.marks ?? marks, VQ.prev.transposed ?? this.transposed) : tbNow;   // the LOCAL tween's snapshot has no mm — only the quant tween eases the pill
    const pillS = VQ ? (tbPrev ? 1 : 0) + ((tbNow ? 1 : 0) - (tbPrev ? 1 : 0)) * VQ.t : (tbNow ? 1 : 0);
    let traffic = null;
    if (pillS > 0.01) {
      const tB = tbNow || tbPrev;   // a vanishing pill shrinks out wearing its OLD content
      const us = (b) => Math.round(b / (HWP.hbm / 1e6));
      const mib = (b) => Math.round(b / 2 ** 20);
      if (tB) traffic = { s: pillS, txt: `requantᵀ ≈ ${us(2 * tB)} µs`,
        tip: `e4m3ᵀ OFF — DeepSeek's convention: the stash keeps ONE orientation, and backward re-quantizes every fp8 stash a weight gradient reads (dequantize → transpose → quantize into 128×1 tiles). A pure HBM round trip — the stash is cold, nothing to fuse into: read + write ≈ ${mib(2 * tB)} MiB ≈ ${us(2 * tB)} µs per microbatch·layer at 3.35 TB/s. Bandwidth, not GEMM FLOPs: named here, never metered by the ruler. (ᵀ ON avoids this by widening the forward quantize kernel's write — fusable, so unpriced like every stash write; its cost is the extra GiB on the bar.)` };
    }
    ribbon('fwd', fwdOps, wFwd, '1.00×');
    ribbon('bwd', fwdOps, (n) => 2 * wFwd(n), '2.00× (dgrad + wgrad)', null, null, traffic);
    ribbon('recompute', fwdOps, wRep, `+${(replayEq / fwdEq).toFixed(2)}×`, ana.replayComm, 'light');
    {
      // the compute ruler: TIME at H100 peak — ticks in ms per 4096-token
      // microbatch·layer (1 ms = 989 GFLOP at the bf16 rate ≈ 241.5
      // MFLOP/token ≈ 72px), so the µs pills riding the ribbons are
      // directly commensurable with the GEMM runs. Minor = 0.5 ms, labels
      // every 1 ms, majors keep the taller tick at 5 ms.
      const PPF = 3 / FUNIT;                       // px per (bf16-rate) FLOP/token
      const FPMS = HWP.flops.bf16 / 4096 / 1e3;   // FLOP/token per ms of peak bf16
      const rext = 2 * fwdEq * PPF + 12;           // bwd is always the longest ribbon
      const ry = ty + 2;
      T.push(`<line x1="${TB_X}" y1="${ry}" x2="${(TB_X + rext).toFixed(1)}" y2="${ry}" stroke="${C('#c3c2b7')}" stroke-width="1"/>`);
      for (let u = 0; u * 0.5 * FPMS * PPF <= rext; u++) {
        const x = TB_X + u * 0.5 * FPMS * PPF, major = u % 10 === 0;
        T.push(`<line x1="${x.toFixed(1)}" y1="${ry}" x2="${x.toFixed(1)}" y2="${ry + (major ? 5 : 2.5)}" stroke="${C('#c3c2b7')}" stroke-width="1"/>`);
        if (u > 0 && u % 2 === 0) T.push(`<text x="${x.toFixed(1)}" y="${ry + 14}" text-anchor="middle" font-size="8.5" fill="${C('#898781')}">${u / 2}</text>`);
      }
      T.push(`<text class="dims" x="${(TB_X + rext + 10).toFixed(1)}" y="${ry + 14}">ms per mb·layer · LINEAR</text>`);
      ty = ry + 20;
    }
    T.push(`<text class="dims" x="${TB_X}" y="${ty + 8}">= ${(3 + replayEq / fwdEq).toFixed(2)}× fwd per training step</text>`);
    ty += 12;
    // honesty line: the vector work the ruler does NOT meter, and what of it
    // the CURRENT policy re-runs in backward — recomputing an RMSNorm or
    // RoPE is cheap (bandwidth-bound) but not free, and the GEMM-only model
    // deliberately leaves it unpriced (no epilogue-fusion story)
    {
      const cnt = {};
      for (const n of fwdOps) if (opDt(n.id) === 'vector' && ana.replayed.has(n.id))
        cnt[n.label.replace(/ \(.*\)/, '')] = (cnt[n.label.replace(/ \(.*\)/, '')] ?? 0) + 1;
      const lst = Object.entries(cnt).map(([k, c]) => c > 1 ? `${k} ×${c}` : k).join(' + ');
      T.push(`<text class="dims" x="${TB_X}" y="${ty + 8}" opacity="0.8">not priced (bandwidth-bound vector work — the hollow dashed marks): `
        + (lst ? `this policy's backward re-runs ${lst}, unmetered — cheap, not free.` : `this policy re-runs no vector work in backward.`) + '</text>');
    }
    const TW = TB_X + TB_AVAIL + 166;
    const tallyEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    tallyEl.setAttribute('width', TW); tallyEl.setAttribute('height', ty + 16);
    tallyEl.setAttribute('viewBox', `0 0 ${TW} ${ty + 16}`);
    tallyEl.innerHTML = T.join('');
    this._tallySvg = tallyEl;

    const H = h + lmH + (this._ctl.quant && this._noKind ? 6 : 14);   // ctx'd quant tiers: tighter tail (laptop vspace; the lint clips-guard patrols)

    // barsonly/snapshot never mount this svg — the string work above already
    // produced the fit chart (_barHtml) and totals; parsing thousands of
    // diagram nodes on every tween frame was the animation stutter
    if (this.hasAttribute('barsonly') || this.hasAttribute('snapshot')) return null;
    const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svgEl.setAttribute('width', WIDTH); svgEl.setAttribute('height', H);
    svgEl.setAttribute('viewBox', `0 0 ${WIDTH} ${H}`);
    // scaling lives in the .lv svg CSS (with a narrow-viewport override that
    // disables it in favor of horizontal scroll) — no inline style, it would win
    // the cascade over the media rule
    svgEl.innerHTML = `<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">` +
      `<path d="M 0 0 L 8 4 L 0 8 z" fill="${C('#898781')}"/></marker></defs>` + P.join('');
    for (const b of svgEl.querySelectorAll('button[data-dt]')) {
      b.onclick = () => {
        const mutate = () => {
          // per-op two-state toggles: bf16 ⇄ the instance's fp8 flavor
          // (e4m3 tile-scaled on the Hopper recipes, mxfp8 on the Blackwell
          // one). o_proj has no per-op lever (its tag is pinned COMPUTE; the
          // stash lever is the E5M6 checkbox). Stale states exit via bf16.
          const cycle = { bf16: FP8K, e4m3: 'bf16', fp8: 'bf16', mxfp8: 'bf16' };
          this.matmuls[b.dataset.dt] = cycle[this.matmuls[b.dataset.dt]] ?? 'bf16';
        };
        if (this.hasAttribute('local') && this.getAttribute('lens') === 'param-bytes') {
          const prev = this._snapLocal(); mutate(); this._tweenLocal(prev);
        } else if (this._ctl.quant) this._tweenQuant(mutate);
        else { mutate(); this.render(); this.changed(); }
      };
    }
    for (const b of svgEl.querySelectorAll('button[data-mark]')) {
      b.onclick = () => this.toggleMark(b.dataset.mark.split(','));
    }
    for (const b of svgEl.querySelectorAll('button[data-regionact]')) {
      b.onclick = () => {
        if (b.disabled) return;
        const rids = b.dataset.rids.split(','), memP = b.dataset.mem + ':prev';
        const snap = () => Object.fromEntries(rids.map(i => [i, this.marks[i] === true]));
        const stName = () => rids.every(i => this.marks[i] === true) ? '\ud83d\udcbe all'
          : rids.every(i => this.marks[i] !== true) ? '\u21bb all' : 'mixed';
        const apply = (m) => { for (const i of rids) { if (m[i]) this.marks[i] = true; else delete this.marks[i]; } };
        this._segMem ??= {};
        if (b.dataset.on === '1') {   // active chip: swap with the previous pick
          const pv = this._segMem[memP];
          if (!pv) return;
          this._segMem[memP] = { sel: stName(), st: snap() };
          this._tweenQuant(() => apply(pv.st));
          return;
        }
        const act = b.dataset.regionact;
        this._segMem[memP] = { sel: stName(), st: snap() };
        this._tweenQuant(() => {
          if (act === 'save') for (const i of rids) this.marks[i] = true;
          else if (act === 'redo') for (const i of rids) delete this.marks[i];
          else apply(this._segMem?.[b.dataset.mem] ?? {});
        });
      };
    }
    for (const b of svgEl.querySelectorAll('[data-kind]')) {
      b.onclick = () => {
        if (this.kind === b.dataset.kind) return;
        this.kind = b.dataset.kind; this.render(); this.changed(true);
      };
    }
    return svgEl;
  }
  // instant tooltips; click a tipped element (not a button) to pin.
  // Besides [data-tip] prose, a raw-bytes cross-check lens rides along:
  // hovering a rounded byte label (data-raw tspan, or a fit-chart value's
  // data-true) shows the unrounded count.
  attachTip(root) {
    const tip = el('div', 'lv-tip');
    root.append(tip);
    let pinned = false;
    let stack = [];   // pinned CELL drill-down: one path through the formula graph, growing downward
    const place = (ev) => {
      const r = root.getBoundingClientRect();
      tip.style.left = Math.min(ev.clientX - r.left + 14, r.width - 280) + 'px';
      tip.style.top = (ev.clientY - r.top + 14) + 'px';
    };
    const rawOf = (ev) => {
      const t = ev.target.closest?.('[data-raw], text[data-true]');
      const v = t && +(t.dataset.raw ?? t.dataset.true);
      return Number.isFinite(v) ? `${v.toLocaleString('en-US', { maximumFractionDigits: 2 })} B` : null;
    };
    // ---- cell tooltips (the full model's quantity graph, src/cells.js):
    // hover = the quantity's formula; pin (click) = the formula's names
    // become clickable and each click pushes that cell's own entry below —
    // a stack, so only one path through the graph is ever on screen
    const fmtVal = (c) => c.unit === 'B'
      ? `${fmtBytes(c.value)} (${c.value.toLocaleString('en-US', { maximumFractionDigits: 2 })} B)`
      : c.unit === 'p' ? `${fmtP(c.value)} params (${c.value.toLocaleString('en-US', { maximumFractionDigits: 2 })})`
        : c.unit === 'B/tok' ? `${c.value.toLocaleString('en-US', { maximumFractionDigits: 2 })} B/token`
          : c.unit === 'B/e' ? `${c.value} B/elem`   // dyadic: String() is exact
            : c.value.toLocaleString('en-US', { maximumFractionDigits: 2 });
    const entry = (c, k) => {
      const d = el('div', 'lv-cellent');
      d.dataset.k = k;
      const h = el('div');
      const b = document.createElement('b'); b.textContent = c.id;
      b.dataset.jump = c.id; b.className = 'celljump'; b.title = 'show this row in the formula sheet';
      h.append(b, ` · ${c.label}`);
      const f = el('div', 'lv-cellfx');
      if (c.expr) {
        f.append('= ');
        for (const tok of c.expr.split(/([A-Z]\d+[a-z]?)/)) {
          if (/^[A-Z]\d+[a-z]?$/.test(tok)) {
            const s = el('span', 'cellref'); s.textContent = tok; s.dataset.cell = tok; f.append(s);
          } else if (tok) f.append(tok);
        }
        f.append(` = ${fmtVal(c)}`);
      } else f.append(`= ${fmtVal(c)}`);   // a leaf: the label says where it comes from
      d.append(h, f);
      return d;
    };
    const renderStack = () => {
      const cells = this._cells?.();
      if (!cells) return false;
      tip.replaceChildren(...stack.map((id, k) => entry(cells.byId.get(id), k)));
      if (pinned) {
        const hint = el('div', 'lv-cellhint');
        hint.textContent = 'click a name to expand it below · click elsewhere to close';
        tip.append(hint);
      }
      return true;
    };
    root.addEventListener('mousemove', (ev) => {
      if (pinned) return;
      const cellEl = ev.target.closest?.('[data-cell]');
      if (cellEl && this._cells) {
        stack = [cellEl.dataset.cell];
        if (renderStack()) { tip.style.display = 'block'; place(ev); return; }
      }
      const raw = rawOf(ev);
      const t = ev.target.closest?.('[data-tip]');
      if (raw) { tip.textContent = raw; tip.style.display = 'block'; place(ev); }
      else if (t) { tip.textContent = t.dataset.tip; tip.style.display = 'block'; place(ev); }
      else tip.style.display = 'none';
    });
    // clicks INSIDE the pinned tip drill (and never close it): a formula
    // name truncates the stack to its own entry and pushes its cell below
    tip.addEventListener('click', (ev) => {
      if (!pinned) return;
      ev.stopPropagation();
      const j = ev.target.closest?.('b[data-jump]');
      if (j) { document.querySelector(`dsv3-sheet[layer="${this.id}"]`)?.reveal(j.dataset.jump); return; }
      const ref = ev.target.closest?.('.cellref');
      if (!ref) return;
      stack = stack.slice(0, +ref.closest('.lv-cellent').dataset.k + 1).concat(ref.dataset.cell);
      renderStack();
    });
    const unpin = () => { pinned = false; stack = []; tip.classList.remove('pinned'); tip.style.pointerEvents = 'none'; tip.style.display = 'none'; };
    root.addEventListener('click', (ev) => {
      if (pinned) { unpin(); return; }
      const cellEl = ev.target.closest?.('[data-cell]');
      if (cellEl && this._cells && !ev.target.closest('button, select, [data-prop], [data-part]')) {
        pinned = true; tip.classList.add('pinned'); tip.style.pointerEvents = 'auto';
        stack = [cellEl.dataset.cell];
        renderStack(); tip.style.display = 'block'; place(ev);
        return;
      }
      const t = ev.target.closest?.('[data-tip]');
      if (t && !ev.target.closest('button, select')) {
        pinned = true; tip.classList.add('pinned');
        tip.textContent = t.dataset.tip; tip.style.display = 'block'; place(ev);
      }
    });
    root.addEventListener('mouseleave', () => { if (!pinned) tip.style.display = 'none'; });
  }
}
if (typeof customElements !== 'undefined' && !customElements.get('dsv3-layer')) {
  customElements.define('dsv3-layer', Dsv3Layer);
}

// <dsv3-sheet> lives in src/sheet.js; defined HERE so element upgrade
// order is unchanged (the sheet mounts after the layer it reads).
if (typeof customElements !== 'undefined' && !customElements.get('dsv3-sheet')) {
  customElements.define('dsv3-sheet', Dsv3Sheet);
}

// ---- <dsv3-pp-schedule> custom element -------------------------------------------
// Compact pipeline-schedule strip: one row per PP stage, time flows right.
// F cells take one slot, B cells two (backward ≈ 2× forward FLOPs), and each
// cell is numbered with its microbatch. Linked to a local-lens layer via
// layer="id": it follows the layer's PP degree, schedule knob and stage
// selection (the selected stage's row is tinted). Under 1F1B it draws pp+4
// microbatches — enough warmup + steady state + cooldown to read the pattern;
// ×1 mb draws the single-microbatch fiction (an F wave down, a B wave up).
// Cell colors follow the byte language: a forward stashes activations
// (amber), a backward consumes them to make gradients (orange). No state of
// its own; standalone instances read pp/sched/stage attributes instead.
const PPS_CSS = `
dsv3-pp-schedule { display: block; margin: 14px 0; }
.pps { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-0b0b0b);
  border: 1px solid var(--c-e1e0d9); border-radius: 6px; background: var(--c-fcfcfb); padding: 8px 10px;
  width: max-content; max-width: 100%; box-sizing: border-box; }   /* cards hug their strip (wide strips cap + scroll) */
.pps .top { display: flex; align-items: flex-start; gap: 12px; padding-bottom: 8px; flex-wrap: wrap; }
${knobCss('.pps .top')}
.pps .hd { color: var(--c-52514e); font-size: 11.5px; align-self: center; }
.pps .stghit { cursor: pointer; }
.pps g.lane { cursor: pointer; }
.pps g.lane.pin rect[data-stash] { stroke-width: 1.6; }
@media (hover: hover) { .pps .stghit:hover { opacity: 0.6; } }
.pps .scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; overscroll-behavior-x: none; }
.pps .scroll.pannable { cursor: grab; }
.pps .scroll.panning { cursor: grabbing; user-select: none; }
.pps svg { display: block; }
`;
class Dsv3PpSchedule extends HTMLElement {
  connectedCallback() {
    this._sig = '';
    this._m = this.getAttribute('mb') === 'auto' ? 'auto' : +(this.getAttribute('mb') ?? 64);
    const style = document.createElement('style'); style.textContent = PPS_CSS;
    this._root = el('div', 'pps');
    this._top = el('div', 'top');
    this._hd = el('div', 'hd');
    this._top.append(this._hd);
    this._scr = el('div', 'scroll');
    // the sX axis IS the stage picker: click a row's gutter to select it;
    // once clicked the strip holds focus, so ↑/↓ walk the stages
    this._scr.tabIndex = -1;
    this._scr.style.outline = 'none';
    // Perfetto-style panning, DIRECTLY: press and drag pans the timeline
    // (the strip is usually wider than the column); a stationary click still
    // picks a stage or pins a lane. A 4px horizontal threshold splits the
    // two — once a drag becomes a pan, it suppresses the click that follows.
    // Shift+drag pans from the first pixel (and a shift-click never picks).
    this._scr.addEventListener('mousemove', (e) => {
      if (!this._scr.classList.contains('panning'))
        this._scr.classList.toggle('pannable', e.shiftKey || this._scr.scrollWidth > this._scr.clientWidth + 2);
    });
    this._scr.addEventListener('mousedown', (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      this._justPanned = false;
      const sx = e.clientX, sl = this._scr.scrollLeft;
      let live = e.shiftKey;                 // shift: pan from the first pixel
      if (live) this._scr.classList.add('panning');
      const mv = (e2) => {
        if (!live && Math.abs(e2.clientX - sx) <= 4) return;
        if (!live) { live = true; this._scr.classList.add('panning'); }
        this._justPanned = true;
        this._scr.scrollLeft = sl - (e2.clientX - sx);
      };
      const up = () => {
        document.removeEventListener('mousemove', mv);
        document.removeEventListener('mouseup', up);
        this._scr.classList.remove('panning');
      };
      document.addEventListener('mousemove', mv);
      document.addEventListener('mouseup', up);
    });
    this._scr.addEventListener('click', (e) => {
      if (e.shiftKey || this._justPanned) { this._justPanned = false; return; }   // that was a pan, not a pick
      const g = e.target.closest('g.lane');
      if (g) {   // clicking an in-flight bar lights up ITS two ops in the
        // schedule — the F that made the stash and the B that frees it — and
        // dims the rest; click again (or reconfigure) to release
        const mb = +g.dataset.mb, v = +g.dataset.v;
        const same = this._pinHl && this._pinHl.mb === mb && this._pinHl.v === v;
        this._scr.querySelector('g.lane.pin')?.classList.remove('pin');
        this._pinHl = same ? null : { mb, v };
        if (!same) g.classList.add('pin');
        this._hl(this._pinHl?.mb ?? null, this._pinHl?.v);
        return;
      }
      const t = e.target.closest('[data-stage]');
      if (!t || this.hasAttribute('noflight')) return;   // no braid → nothing to pick for
      this._scr.focus({ preventScroll: true });
      this._pick(+t.dataset.stage);
    });
    this._scr.addEventListener('keydown', (e) => {
      const d = e.key === 'ArrowDown' ? 1 : e.key === 'ArrowUp' ? -1 : 0;
      if (!d || this.hasAttribute('noflight')) return;
      e.preventDefault();
      const { pp, stage } = this.cfg();
      this._pick(Math.min(pp - 1, Math.max(0, stage + d)));
    });
    this._root.append(this._top, this._scr);
    this.append(style, this._root);
    const lid = this.getAttribute('layer');
    if (!lid) { this.sync(); return; }
    const bind = () => {   // the layer upgrades async; poll briefly until it's live
      const l = document.getElementById(lid);
      if (l?.render) { this._layer = l; l.addEventListener('recipe', () => this.sync()); this.sync(); }
      else setTimeout(bind, 30);
    };
    bind();
  }
  // stage picking: bound strips drive the LAYER (both widgets move); a
  // standalone strip keeps its own selection and just redraws
  _pick(v) {
    const l = this._layer;
    if (l) { if (l.stage !== v) l.setLocal(() => { l.stage = v; }); }
    else if (this._stage !== v) { this._stage = v; this._sig = ''; this.sync(); }
  }
  _hl(mb, v) {
    const svg = this._scr.querySelector('svg');
    if (!svg) return;
    for (const n of svg.querySelectorAll('[data-mb]')) {
      const on = mb == null || (+n.dataset.mb === mb && +n.dataset.v === v);
      n.style.opacity = on ? '' : 0.22;
    }
  }
  // the SAME pipeline group the layer's mini-head wears (PP stepper over
  // powers of two, stage select with layer-assignment labels, 1F1B/×1 mb
  // segments) — changes drive the LAYER (l.setLocal), and the layer's
  // 'recipe' event circles back to redraw both widgets in sync.
  cfg() {
    const l = this._layer;
    return {
      pp: l?.pp ?? +(this.getAttribute('pp') ?? LOCAL_PAR.pp),
      sched: l?.sched ?? (this.getAttribute('sched') ?? '1f1b'),
      stage: l?.stage ?? this._stage ?? +(this.getAttribute('stage') ?? 0),
      vpp: l?.vpp ?? +(this.getAttribute('vpp') ?? 1),
      fold: l?.fold ?? (this.getAttribute('fold') ?? 'reflect'),
      a2a: l?.a2a ?? this.hasAttribute('a2a'),
    };
  }
  sync() {
    const { pp, sched, stage, vpp, fold, a2a } = this.cfg();
    const sig = `${pp}|${sched}|${stage}|${vpp}|${fold}|${a2a}|${this._m}`;
    if (sig === this._sig) return;   // the layer's tween fires 'recipe' every frame
    const grew = this._sig.split('|').slice(0, 2).join('|') !== `${pp}|${sched}`;
    const prevH = this._sig && grew ? this._scr.getBoundingClientRect().height : 0;
    this._sig = sig;
    this.draw(pp, sched, stage, vpp, fold, a2a);
    if (prevH) {   // animate the reflow: same deterministic 12-frame ease-out
      const target = this._scr.scrollHeight;
      const FR = 12; let f = 0;
      this._scr.style.overflowY = 'hidden';
      this._scr.style.height = prevH + 'px';
      const step = () => {
        f++; const q = Math.min(1, f / FR);
        const p = q < 0.5 ? 4 * q * q * q : 1 - (-2 * q + 2) ** 3 / 2;   // cubic in-out, like every tween
        this._scr.style.height = (prevH + (target - prevH) * p) + 'px';
        if (f < FR) setTimeout(step, 16);
        else { this._scr.style.height = ''; this._scr.style.overflowY = ''; }
      };
      setTimeout(step, 16);
    }
  }
  // The OFFICIAL DualPipeV program (deepseek-ai/DualPipe, dualpipev.py):
  // eight steps per rank. Backward splits zero-bubble style into B (input
  // grads, one slot) and W (weight grads, one slot, deferred via a FIFO);
  // the steady state runs F of one chunk fused with the OTHER chunk's B as
  // one overlapped block (drawn as a split cell; we model its F result
  // available one slot in, its B at block end — the real kernels interleave).
  // Requires m ≥ 2pp. Emits the same cell records as the generic engine.
  _officialDPV(pp, m) {
    const D = 2 * pp;
    const prog = [];
    for (let r = 0; r < pp; r++) {
      const ops = [], wq = [];
      const v0 = r, v1 = D - 1 - r;
      let f0 = 0, f1 = 0, b0 = 0, b1 = 0;
      const F0 = () => ops.push({ ph: 'F', v: v0, mb: f0++ });
      const F1 = () => ops.push({ ph: 'F', v: v1, mb: f1++ });
      const B = (v, mb, zb) => { ops.push({ ph: 'B', v, mb, zb }); if (zb) wq.push({ v, mb }); };
      const W = () => { const e = wq.shift(); if (e) ops.push({ ph: 'W', v: e.v, mb: e.mb }); };
      const FB = (vF, mbF, vB, mbB) => ops.push({ ph: 'FB', vF, mbF, vB, mbB });
      for (let i = 0; i < (pp - r - 1) * 2; i++) F0();                      // 1: nF0
      for (let i = 0; i < r + 1; i++) { F0(); F1(); }                       // 2: nF0F1
      for (let i = 0; i < pp - r - 1; i++) { B(v1, b1++, true); W(); F1(); } // 3: nB1W1F1
      for (let i = 0; i < m - 2 * pp + r + 1; i++) {                        // 4: main nF0B1F1B0
        FB(v0, f0++, v1, b1++); FB(v1, f1++, v0, b0++);
      }
      for (let i = 0; i < pp - r - 1; i++) { B(v1, b1++, false); FB(v1, f1++, v0, b0++); } // 5: nB1F1B0
      let zb = false;                                                       // 6: nB1B0 (2nd half zb)
      for (let i = 0; i < r + 1; i++) {
        if (i === (r + 1 >> 1) && r % 2 === 1) zb = true;
        B(v1, b1++, zb);
        if (i === (r + 1 >> 1) && r % 2 === 0) zb = true;
        B(v0, b0++, zb);
      }
      for (let i = 0; i < pp - r - 1; i++) { W(); B(v0, b0++, true); }      // 7: nWB0
      for (let i = 0; i < r + 1; i++) W();                                  // 8: nW
      prog.push({ ops, i: 0 });
    }
    return this._resolveProg(prog, pp, D).cells;
  }
  // Megatron-Core's interleaved 1F1B (schedules.py,
  // forward_backward_pipelining_with_interleaving), per rank: warmup
  // forwards, then strict 1F1B pairs, then cooldown backwards. The k-th
  // forward on a rank is (chunk ⌊(k mod PP·V)/PP⌋, microbatch
  // ⌊k/(PP·V)⌋·PP + k mod PP): forwards run chunk-major in groups of PP
  // microbatches; backwards mirror it with the chunks reversed. Warmup =
  // 2(PP−r−1) + (V−1)·PP + 1 chunk-forwards (V = 1 is the plain schedule:
  // PP−r−1). No B/W split (Megatron's default). Requires m % PP = 0.
  _officialInterleaved(pp, vpp, m, a2a = false) {
    const total = m * vpp, G = pp * vpp;
    const chunkOf = (k, fwd) => { const c = Math.floor((k % G) / pp); return fwd ? c : vpp - 1 - c; };
    const mbOf = (k) => Math.floor(k / G) * pp + (k % pp);
    const prog = [];
    for (let r = 0; r < pp; r++) {
      const W = Math.min(vpp === 1 ? pp - r - 1 : (pp - r - 1) * 2 + (vpp - 1) * pp + (a2a ? 1 : 0), total);
      const ops = [];
      const F = (k) => ops.push({ ph: 'F', v: chunkOf(k, true) * pp + r, mb: mbOf(k) });
      const B = (k) => ops.push({ ph: 'B', v: chunkOf(k, false) * pp + r, mb: mbOf(k) });
      for (let k = 0; k < W; k++) F(k);
      for (let i = 0; i < total - W; i++) { F(W + i); B(i); }
      for (let i = total - W; i < total; i++) B(i);
      prog.push({ ops, i: 0 });
    }
    return this._resolveProg(prog, pp, G).cells;
  }
  // resolve a per-rank op program's timing: rank-sequential, F waits on
  // F@v−1, B on B@v+1 (its own F at the deepest stage), W only on rank
  // order; fused blocks wait on both. Shared by the official DualPipeV
  // program and reader-written custom programs; a program whose deps can
  // never be met simply stops (stalled = the ops left waiting).
  _resolveProg(prog, pp, D) {
    const done = new Map(), cells = [], rankT = Array(pp).fill(0);
    const depF = (v, mb) => v === 0 ? 0 : done.get(`F${mb}@${v - 1}`);
    const depB = (v, mb) => v === D - 1 ? done.get(`F${mb}@${v}`) : done.get(`B${mb}@${v + 1}`);
    let progress = true;
    while (progress) {
      progress = false;
      for (let r = 0; r < pp; r++) {
        const q = prog[r];
        while (q.i < q.ops.length) {
          const o = q.ops[q.i];
          let rec;
          if (o.ph === 'F' || o.ph === 'B') {
            const dep = o.ph === 'F' ? depF(o.v, o.mb) : depB(o.v, o.mb);
            if (dep === undefined) break;
            const t0 = Math.max(rankT[r], dep), t1 = t0 + (o.ph === 'B' && !o.zb ? 2 : 1);
            rec = [{ s: r, v: o.v, mb: o.mb, ph: o.ph, t0, t1, zb: o.zb, chunk: Math.floor(o.v / pp) }];
            done.set(`${o.ph}${o.mb}@${o.v}`, t1);
            rankT[r] = t1;
          } else if (o.ph === 'W') {
            const t0 = rankT[r];
            rec = [{ s: r, v: o.v, mb: o.mb, ph: 'W', t0, t1: t0 + 1, chunk: Math.floor(o.v / pp) }];
            rankT[r] = t0 + 1;
          } else {   // FB: fused forward+backward, 3 slots
            const dF = depF(o.vF, o.mbF), dB = depB(o.vB, o.mbB);
            if (dF === undefined || dB === undefined) break;
            const t0 = Math.max(rankT[r], dF, dB);
            rec = [
              { s: r, v: o.vF, mb: o.mbF, ph: 'F', t0, t1: t0 + 1, fuse: 1, chunk: Math.floor(o.vF / pp) },
              { s: r, v: o.vB, mb: o.mbB, ph: 'B', t0: t0 + 1, t1: t0 + 3, fuse: 2, chunk: Math.floor(o.vB / pp) },
            ];
            done.set(`F${o.mbF}@${o.vF}`, t0 + 1);
            done.set(`B${o.mbB}@${o.vB}`, t0 + 3);
            rankT[r] = t0 + 3;
          }
          cells.push(...rec);
          q.i++; progress = true;
        }
      }
    }
    const stalled = prog.reduce((t, q) => t + (q.ops.length - q.i), 0);
    return { cells, stalled };
  }
  draw(pp, sched, stage, vpp = 1, fold = 'reflect', a2a = false) {
    this._pinHl = null;
    // one chain of VIRTUAL stages with 1F1B admission per stage, vpp·pp deep;
    // placement per vstagesOf (wrap = Megatron interleaving, reflect = the
    // V/DualPipeV zigzag). Each rank interleaves its chunk queues greedily,
    // one op at a time (the official DualPipeV also overlaps F+B blocks;
    // this doesn't — but the residency it draws IS the modeled law).
    // F_mb@v waits on F_mb@(v−1); B_mb@v on B_mb@(v+1), or its own forward
    // on the deepest stage. Durations: F = 1 slot, B = 2 (~2× the FLOPs).
    const D = vpp * pp;
    const ILV = sched === 'interleaved' && pp > 1;
    // interleaved 'auto': enough microbatches (a multiple of PP) to reach the
    // steady state on rank 0 — its warmup is PP·V + PP − 1 chunk-forwards
    const m = sched === 'one' ? 1 : this._m === 'auto' ? (ILV ? pp * (vpp + 2) : D + 4) : this._m;
    const stagesOf = Array.from({ length: pp }, (_, r) => vstagesOf(r, pp, vpp, fold));
    // the schedule IS DualPipeV: the official program whenever it exists
    // (pp > 1, steady state reachable); the greedy engine covers the rest
    // (the ×1 mb wave, m < 2·PP, PP1) on the same virtual chain. The braid
    // shows what the DRAWING holds; the peak label calls out any
    // drawn-vs-law gap.
    const OFFICIAL = sched === '1f1b' && pp > 1 && m >= 2 * pp;
    const MEGATRON = ILV && m % pp === 0;
    const qs = OFFICIAL || MEGATRON ? [] : Array.from({ length: D }, (_, v) => {
      const wu = Math.min(D - 1 - v, m); const items = [];
      for (let j = 0; j < wu; j++) items.push(['F', j]);
      for (let j = wu; j < m; j++) items.push(['F', j], ['B', j - wu]);
      for (let j = Math.max(m - wu, 0); j < m; j++) items.push(['B', j]);
      return { items, i: 0 };
    });
    const done = new Map();
    let cells = OFFICIAL ? this._officialDPV(pp, m) : MEGATRON ? this._officialInterleaved(pp, vpp, m, a2a) : [];
    const rankT = Array(pp).fill(0);
    let progress = !OFFICIAL && !MEGATRON;
    while (progress) {
      progress = false;
      for (let r = 0; r < pp; r++) {
        // among this rank's queue heads, run the one ready EARLIEST
        // (ties: backward first, then the deeper virtual stage — drain bias)
        let best = null;
        for (const v of stagesOf[r]) {
          const q = qs[v];
          if (q.i >= q.items.length) continue;
          const [ph, mb] = q.items[q.i];
          const dep = ph === 'F'
            ? (v === 0 ? 0 : done.get(`F${mb}@${v - 1}`))
            : (v === D - 1 ? done.get(`F${mb}@${v}`) : done.get(`B${mb}@${v + 1}`));
          if (dep === undefined) continue;
          const cand = { v, q, ph, mb, ready: Math.max(rankT[r], dep) };
          if (!best || cand.ready < best.ready
            || (cand.ready === best.ready && cand.ph === 'B' && best.ph === 'F')
            || (cand.ready === best.ready && cand.ph === best.ph && cand.v > best.v)) best = cand;
        }
        if (!best) continue;
        const t0 = best.ready, t1 = t0 + (best.ph === 'F' ? 1 : 2);
        cells.push({ s: r, v: best.v, mb: best.mb, ph: best.ph, chunk: Math.floor(best.v / pp), t0, t1 });
        done.set(`${best.ph}${best.mb}@${best.v}`, t1);
        rankT[r] = t1; best.q.i++; progress = true;
      }
    }
    const T = Math.max(1, ...cells.map(c => c.t1));
    const U = 10, RH = 14, GAP = 2, GUT = 34;   // slot width / row height / row gap / stage gutter

    // ---- in-flight lanes for the SELECTED stage: each stash is one bar —
    // the F that makes it, an amber tail while it is held, and the B that
    // frees it. The braid's thickness IS the in-flight count; a dashed line
    // marks the modeled peak (resident once the forward completes).
    const vset = new Set(stagesOf[Math.min(stage, pp - 1)]);
    const byKey = new Map();
    for (const c of cells) {
      if (!vset.has(c.v) || c.ph === 'W') continue;
      const e = byKey.get(`${c.v}:${c.mb}`) ?? { mb: c.mb, v: c.v, chunk: c.chunk };
      if (c.ph === 'F') { e.f0 = c.t0; e.f1 = c.t1; } else { e.b0 = c.t0; e.b1 = c.t1; }
      byKey.set(`${c.v}:${c.mb}`, e);
    }
    const stash = [...byKey.values()].sort((a, b) => a.f0 - b.f0);
    const laneEnd = [];   // lowest-free-lane allocation over each bar's [f0, b1]
    for (const e of stash) {
      let ln = laneEnd.findIndex(t => t <= e.f0);
      if (ln < 0) { ln = laneEnd.length; laneEnd.push(0); }
      laneEnd[ln] = e.b1; e.lane = ln;
    }
    // modeled peak, in the model's convention: resident from F end to B end
    const evts = stash.flatMap(e => [[e.f1, 1], [e.b1, -1]]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    let live = 0, peakN = 0;
    for (const [, d] of evts) peakN = Math.max(peakN, live += d);

    const RH2 = 12, HDR = 18;                   // lane height / section header
    // 'noflight' drops the in-flight braid — a pure-schedule figure (the
    // single-microbatch V doesn't need a one-lane braid narrating it)
    const FLIGHT = !this.hasAttribute('noflight');
    const schedH = pp * (RH + GAP) - GAP;
    const laneY0 = schedH + 10 + HDR;
    const H = FLIGHT ? laneY0 + laneEnd.length * (RH2 + GAP) - GAP + 4 : schedH + 4;
    const W = GUT + T * U + 1;
    const rowY = (s) => s * (RH + GAP);
    const laneY = (ln) => laneY0 + ln * (RH2 + GAP);
    const P = [`<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family="system-ui">`];
    if (FLIGHT && pp > 1) P.push(`<rect class="stghl" x="0" y="${rowY(stage)}" width="${W}" height="${RH}" fill="${C('#fff3d1')}"/>`);
    for (let s = 0; s < pp; s++) {
      const on = FLIGHT && s === stage;
      P.push(`<text x="${GUT - 6}" y="${rowY(s) + RH - 4}" text-anchor="end" font-size="9.5"`
        + ` font-weight="${on ? 600 : 400}" fill="${on ? C('#0b0b0b') : C('#898781')}">s${s}</text>`);
      if (FLIGHT) P.push(`<rect class="stghit" data-stage="${s}" x="0" y="${rowY(s)}" width="${GUT - 2}" height="${RH}" fill="${C('#fff3d1')}" opacity="0"/>`);
    }
    // later chunks wear progressively deeper shades of the same hues
    // (VPP2 reflect: light down pass, dark up pass); W = deferred weight
    // grads, the pale dashed cells of the zero-bubble split
    const STY = {
      F: [[C('#fdeab5'), C('#eda100'), C('#7a5200')], [C('#f6cd74'), C('#c98800'), C('#5c3d00')],
        [C('#eab04a'), C('#a86e00'), C('#4a3000')], [C('#d69432'), C('#875600'), C('#3a2500')]],
      B: [[C('#fbd4c0'), C('#eb6834'), C('#7a2f12')], [C('#f3ac8b'), C('#c74e1d'), C('#5c2410')],
        [C('#e58a63'), C('#a63c12'), C('#471b09')], [C('#d16b42'), C('#88300c'), C('#361406')]],
      W: [[C('#fdefe8'), C('#eb6834'), C('#7a2f12')], [C('#f9ded0'), C('#c74e1d'), C('#5c2410')],
        [C('#f3c8b3'), C('#a63c12'), C('#471b09')], [C('#ecb298'), C('#88300c'), C('#361406')]],
    };
    // beyond four chunks per rank, neighbouring chunks share a shade
    const shade = (k) => Math.min(3, vpp <= 4 ? k : Math.floor(k * 4 / vpp));
    for (const c of cells) {
      const [fill, stroke, ink] = STY[c.ph][shade(c.chunk)];
      const xr = GUT + c.t0 * U, span = (c.t1 - c.t0) * U;
      // contiguity IS the fusion cue: ops wear a 2.5px trailing gap, but a
      // fused F&B pair is drawn flush — two full-height cells sharing an edge
      const x = c.fuse === 2 ? xr - 0.5 : xr + 0.5;
      const w = c.fuse === 1 ? span - 1 : c.fuse === 2 ? span - 2 : span - 3;
      const dash = c.ph === 'W' ? ' stroke-dasharray="2 1.5"' : '';
      P.push(`<rect data-cell="${c.ph}${c.mb}@${c.s}" data-mb="${c.mb}" data-v="${c.v}" data-t0="${c.t0}" data-t1="${c.t1}" x="${x}" y="${rowY(c.s) + 0.5}" width="${w}" height="${RH - 1}" fill="${fill}" stroke="${stroke}" stroke-width="0.8"${dash}/>`);
      // no wide convention for narrow double digits — shrink the font instead
      const fs = w >= 12 || c.mb < 10 ? 7.5 : 6;
      if (pp <= 32 && c.mb < (w >= 12 ? 1000 : 100))
        P.push(`<text data-mb="${c.mb}" data-v="${c.v}" x="${x + w / 2}" y="${rowY(c.s) + RH - 4}" text-anchor="middle" font-size="${fs}" fill="${ink}">${c.mb}</text>`);
    }
    // ---- the in-flight section (same svg → the horizontal scroll is shared)
    const IFm = peakN / vpp;
    if (FLIGHT) {
    const law = inflightOf(sched, stage, pp, vpp, fold, null, null, { a2a });
    const lawTag = Math.abs(IFm - law) > 1e-9 ? ` — the model charges ${law} (its 1F1B law)` : '';
    P.push(`<text x="0" y="${laneY0 - 7}" font-size="10" fill="${C('#52514e')}">in flight on s${Math.min(stage, pp - 1)}`
      + ` — each bar: the F that stashes a microbatch, held (amber) until the B that frees it.`
      + ` The peak is what the memory bars charge</text>`);
    for (const e of stash) {
      const [f, fs2, fi] = STY.F[shade(e.chunk)], [bf, bs, bi] = STY.B[shade(e.chunk)];
      const y = laneY(e.lane);
      P.push(`<g class="lane" data-mb="${e.mb}" data-v="${e.v}">`);
      // hitbox: the whole row band over the stash's span, not just the marks
      P.push(`<rect x="${GUT + e.f0 * U}" y="${y - GAP / 2}" width="${(e.b1 - e.f0) * U}" height="${RH2 + GAP}" fill="transparent"/>`);
      P.push(`<rect x="${GUT + e.f1 * U}" y="${y + RH2 / 2 - 2}" width="${(e.b0 - e.f1) * U}" height="4" fill="${C('#fdeab5')}" data-stash-tail="1"/>`);
      P.push(`<rect data-stash="F${e.mb}" x="${GUT + e.f0 * U + 0.5}" y="${y + 0.5}" width="${(e.f1 - e.f0) * U - 1}" height="${RH2 - 1}" fill="${f}" stroke="${fs2}" stroke-width="0.8"/>`);
      P.push(`<rect data-stash="B${e.mb}" x="${GUT + e.b0 * U + 0.5}" y="${y + 0.5}" width="${(e.b1 - e.b0) * U - 1}" height="${RH2 - 1}" fill="${bf}" stroke="${bs}" stroke-width="0.8"/>`);
      if (pp <= 32 && e.mb < 100) {
        P.push(`<text x="${GUT + (e.f0 + e.f1) / 2 * U}" y="${y + RH2 - 3}" text-anchor="middle" font-size="${e.mb < 10 ? 7.5 : 6}" fill="${fi}">${e.mb}</text>`);
        P.push(`<text x="${GUT + (e.b0 + e.b1) / 2 * U}" y="${y + RH2 - 3}" text-anchor="middle" font-size="7.5" fill="${bi}">${e.mb}</text>`);
      }
      P.push('</g>');
    }
    // the modeled peak, annotated like a dimension line: a bracket spanning
    // the braid at a steady-state moment when the count is maximal
    let live2 = 0, tPk = 0, tPkEnd = T;
    for (let i = 0; i < evts.length; i++) {
      live2 += evts[i][1];
      if (live2 === peakN) { tPk = evts[i][0]; tPkEnd = evts[i + 1]?.[0] ?? T; break; }
    }
    const bx = GUT + (tPk + tPkEnd) / 2 * U;
    const by0 = laneY(0) + 1, by1 = laneY(peakN - 1) + RH2 - 1;
    P.push(`<path d="M ${bx - 4} ${by0} h 8 M ${bx} ${by0} V ${by1} M ${bx - 4} ${by1} h 8" stroke="${C('#0b0b0b')}" stroke-width="1.2" fill="none" pointer-events="none"/>`);
    P.push(`<text data-peak="${IFm}" x="${bx + 7}" y="${(by0 + by1) / 2 + 3.5}" font-size="10" font-weight="600" fill="${C('#0b0b0b')}" stroke="${C('#fcfcfb')}" stroke-width="3" paint-order="stroke" pointer-events="none">${IFm} mb in flight (peak)${vpp > 1 ? ` = ${peakN} chunks` : ''}${lawTag}</text>`);
    }
    P.push('</svg>');
    const ppTag = this._layer ? '' : `PP${pp} · `;   // the knob group already names PP
    const vppTag = MEGATRON
      ? `interleaved 1F1B (Megatron) · each rank runs ${vpp} chunk${vpp > 1 ? 's' : ''}, later chunks darker · forwards issued`
        + ` chunk-major in groups of PP microbatches · warmup 2(PP−r−1) + (VP−1)·PP${a2a ? ' + 1 (a2a overlap)' : ''} chunk-forwards, then strict 1F1B`
      : OFFICIAL
      ? 'DualPipeV (official program) · down-pass chunk light, up-pass dark · F and B drawn touching = one'
        + ' overlapped F&B block · pale dashed W = deferred weight grads (B alone = input grads)'
      : vpp > 1
        ? `DualPipeV, greedy fill-in (the official program needs ≥ 2·PP microbatches)`
          + ' · each rank runs 2 chunks, up-pass darker · chunks scheduled 1F1B'
        : '';
    this._hd.textContent = sched === 'one'
      ? `${ppTag}one microbatch at a time — an F wave down the stages, then a B wave back up`
      : vpp > 1 || MEGATRON
        ? `${ppTag}${m} microbatches shown · ${vppTag}`
        : `${ppTag}${m} microbatches shown — F = forward (one slot), B = backward (two: ~2× the FLOPs)`;
    this._scr.innerHTML = P.join('');
  }
}

// ---- <dsv3-pp-fold> custom element -----------------------------------------
// How a pipeline fold distributes PARAMETERS over ranks. Two views of the
// same D-chunk split of the stack (pp × vpp; 02: DualPipeV's 16 chunks at
// PP8 on the 63-slot split; 03: Megatron layouts): 'virtual' lays the chunks
// out as if each were a rank of a D-deep chain; 'folded' places them on
// their physical rank — the V pairs chunk v with D−1−v on rank min(v, D−1−v),
// the wrap deals chunk v to rank v mod PP. The toggle
// ANIMATES the fold (each chunk segment flies to its rank; total height is
// reserved, so nothing reflows). A model-stack MINIMAP on the left lights
// the hovered rank's layers — on rank 0 BOTH ends of the model light at
// once, the fold's signature (emb AND head on one rank). Parameter counts
// are exact (ppStage over the 16-chunk split; experts ÷ EP — the essay
// arrives here with EP64 already applied; ep="" overrides).
const PPF_CSS = `
dsv3-pp-fold { display: block; margin: 14px 0; }
.pf { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-0b0b0b);
  border: 1px solid var(--c-e1e0d9); border-radius: 6px; background: var(--c-fcfcfb); padding: 8px 10px;
  width: max-content; max-width: 100%; box-sizing: border-box; }   /* hug the chart, like snapshot cards */
.pf .top { display: flex; align-items: flex-end; gap: 14px; padding-bottom: 6px; }
${knobCss('.pf .top')}
.pf .top button.cyc { font: 12px ui-monospace, monospace; padding: 3px 12px; border: 1px solid var(--c-c3c2b7);
  border-radius: 4px; background: var(--c-ffffff); cursor: pointer; }
@media (hover: hover) { .pf .top button.cyc:hover { background: var(--c-f3f2ee); } }
.pf svg { display: block; }
.pf .ro { font-size: 11.5px; color: var(--c-52514e); min-height: 17px; margin-top: 2px; }
`;
class Dsv3PpFold extends HTMLElement {
  connectedCallback() {
    this.ep = +(this.getAttribute('ep') ?? 64);
    // geometry: PP ranks × VP chunks per rank; fold = 'reflect' (the V:
    // chunk v pairs with D−1−v, 02's DualPipeV) or 'wrap' (Megatron
    // interleaving: chunk v on rank v mod PP); a layout string (Megatron's
    // grammar, see parseLayout) replaces the 63-slot split and implies wrap
    this.pp = +(this.getAttribute('pp') ?? 8);
    this.vpp = +(this.getAttribute('vpp') ?? 2);
    this.layout = this.getAttribute('layout');
    this.fold = this.layout ? 'wrap' : (this.getAttribute('fold') ?? 'reflect');
    this.D = this.pp * this.vpp;
    this.view = this.getAttribute('view') === 'physical' ? 'physical' : 'virtual';
    this._t = this.view === 'physical' ? 1 : 0;
    this._hover = null;
    const style = document.createElement('style'); style.textContent = PPF_CSS;
    this._root = el('div', 'pf');
    this._top = el('div', 'top');
    this._btn = document.createElement('button');   // ONE button — click to cycle
    this._btn.type = 'button'; this._btn.className = 'cyc';
    this._btn.onclick = () => this._go(this.view === 'virtual' ? 'physical' : 'virtual');
    this._top.append(this._btn);
    this._bars = el('div');
    this._ro = el('div', 'ro');   // hover readout (height reserved)
    this._root.append(this._top, this._bars, this._ro);
    this.append(style, this._root);
    // EP knob in the HOUSE style (the mesh group's pattern): a labeled
    // group, 'EP' text, and the standard −/value/+ stepper — rank-RESIDENT
    // params depend on the sharding, so it's a knob, not fine print
    const grp = el('span', 'pargrp');
    const lab = el('div', 'parlab'); lab.textContent = 'expert sharding'; grp.append(lab);
    const row = el('div', 'parrow');
    const txt = el('span'); txt.style.cssText = 'color:var(--c-52514e);font-size:11px;'; txt.textContent = 'EP';
    const eg = el('span', 'stp'); eg.dataset.knob = 'ep';
    const OPTS = [1, 2, 4, 8, 16, 32, 64];
    const sel = document.createElement('select'); sel.className = 'v';
    for (const o of OPTS) sel.append(new Option(o, o));
    const ebtn = (t, di) => {
      const b = document.createElement('button');
      b.textContent = t; b.type = 'button'; b.dataset.dir = di;
      b.onclick = () => { const j = OPTS.indexOf(this.ep) + di; if (OPTS[j] != null) this._setEP(OPTS[j]); };
      return b;
    };
    sel.onchange = () => this._setEP(+sel.value);
    eg.append(ebtn('−', -1), sel, ebtn('+', +1));
    row.append(txt, eg); grp.append(row);
    this._epUI = { sel, eg };
    this._top.append(grp);
    this.chunks = this._chunks();
    this.render();
  }
  // the D chunks: exact params from the SAME split the memory model uses
  // (the slot split, or the layout's chunks), experts ÷ the current EP
  // (rank-RESIDENT parameters, not whole-model). An MTP chunk carries no
  // params here (MTP is outside the model; the strip still shows the slot).
  _chunks() {
    const moeExp = PARAMS.expert * DSV3.routedExperts;
    return Array.from({ length: this.D }, (_, c) => {
      const g = ppStage(c, this.D, 1, this.layout ? 'wrap' : 'reflect', this.layout);
      const seg = g.segs[0];
      const moeLocal = PARAMS.moeBlock - moeExp + moeExp / this.ep;
      const slotsP = [...(g.emb ? [PARAMS.embed] : []),
        ...Array.from({ length: seg.hi - seg.lo }, (_, i) => seg.lo + i < 3 ? PARAMS.denseBlock : moeLocal),
        ...(g.head ? [PARAMS.embed + PARAMS.finalNorm] : [])];
      return { c, lo: seg.lo, hi: seg.hi, dense: g.dense, moe: g.moe, mtp: seg.mtp ?? 0,
        emb: g.emb, head: g.head, slotsP, p: slotsP.reduce((a, b) => a + b, 0) };
    });
  }
  // EP moves: bars and axis TWEEN between shardings (linear pixel lerp;
  // labels snap to the new exact values, per the house convention)
  _setEP(v) {
    if (v === this.ep) return;
    this._epFrom = this.chunks.map((k) => [...k.slotsP]);
    this.ep = v;
    this.chunks = this._chunks();
    const N = 12; let f = 0;
    const gen = this._gen = (this._gen ?? 0) + 1;
    this._ept = 0; this.render();
    const step = () => {
      if (this._gen !== gen) return;
      f++; this._ept = fitEase(Math.min(1, f / N));
      this.render();
      if (f < N) setTimeout(step, 16);
      else { this._epFrom = null; this.render(); }
    };
    setTimeout(step, 16);
  }
  // deterministic frame-stepped fold/unfold (nothing reflows: height fixed)
  _go(view) {
    if (view === this.view) return;
    this.view = view;
    const from = this._t, to = view === 'physical' ? 1 : 0;
    // TAPE-MEASURE physics: the unroll is graceful (~350 ms); the fold is
    // its reverse-video at recoil speed — a roll-up only reads as physical
    // when it SNAPS. The recoil gets a BUMPER: a mild ease-out on its
    // timeline, so the last stretch settles instead of slamming.
    const N = view === 'physical' ? 14 : 22; let f = 0;
    const gen = this._gen = (this._gen ?? 0) + 1;
    const step = () => {
      if (this._gen !== gen) return;
      f++; const q = Math.min(1, f / N);
      const q2 = view === 'physical' ? 1 - (1 - q) ** 1.6 : q;   // recoil decelerates into the stop
      this._t = from + (to - from) * q2;   // RAW progress: per-chunk easing lives in render
      this.render();
      if (f < N) setTimeout(step, 16);
    };
    setTimeout(step, 16);
  }
  // physical placement: the V pairs chunk c with D−1−c on rank min(c, D−1−c);
  // wrap deals chunk c to rank c mod PP
  _rankOf(c) { return this.fold === 'wrap' ? c % this.pp : Math.min(c, this.D - 1 - c); }
  _rankChunks(r) { return this.chunks.filter((k) => this._rankOf(k.c) === r); }
  render() {
    const t = this._t, D = this.D, pp = this.pp;
    this._btn.textContent = this.view === 'virtual' ? `fold onto the ${pp} ranks ⤵` : `unroll into ${D} chunks ⤴`;
    this._epUI.sel.value = String(this.ep);
    for (const b of this._epUI.eg.querySelectorAll('button')) {
      const j = [1, 2, 4, 8, 16, 32, 64].indexOf(this.ep) + +b.dataset.dir;
      b.disabled = j < 0 || j > 6;
    }
    // ONE svg, D UNIFORM rows — each row IS a span of layers: the grouped
    // stack on the left (emb cap · layer cells, dense dark · head cap), the
    // parameter bar beside it. Folding never re-pitches the rows: the
    // later-pass bars POP onto their rank's row (after the bars already
    // there) and rows 0–PP−1 relabel s0–s(PP−1); the vacated rows keep
    // their spans (the stack stays readable — the fold moves COST, not layers).
    // left → right: the CONTINUOUS model strip (63 slots: emb · 61 layers ·
    // head, top to bottom — tokens flow down the page; + an MTP slot when
    // the layout carries one), chunk brackets, a routing fan of
    // square-cornered leaders stringing each span to its bar row, then the
    // v/s axis, range labels, and the bars
    const SL = 8, STW = 16, BRX = SL + STW + 2, RX0 = BRX + 8;
    const MTP = this.chunks.some((k) => k.mtp) ? 1 : 0;   // an MTP layer in the layout: one more strip slot, longer range labels
    const GUT = RX0 + pp * 1.6 + 22, LX = GUT + 6, X0 = LX + (MTP ? 86 : 64);
    // blocks (PROTOTYPE): the linear chart leans into the site convention —
    // squares/units for linear, bars for log. Each slot renders as tall-thin
    // unit rects at the global byte quantum (448 MiB bf16 = 224M params),
    // layer separations widen, and the x-scale is FIXED by the unit (the EP
    // knob changes the unit count, so EP1 gets enormous — that's the honest
    // picture; the essay arrives here with EP64 already applied). SOLID
    // bars on a FIXED graduated scale: countability lives in the shared
    // unit RULER under the bars (minor tick = 128 MiB bf16, major = 1 GiB,
    // 8 minors to a major), labels in bytes.
    const RH = 14, PV = 21;
    // params per ruler unit · px per unit. The unit is 128 MiB bf16 (02's
    // EP64 ranks hold ~9 GiB); when a rank holds much more (03: EP32 over
    // two ranks ≈ 36 GiB) the ruler coarsens ×4 to 512 MiB so the widest
    // rank still fits the column instead of overflowing it
    const UNIT0 = 128 * 2 ** 20 / 2, UPX = 6;
    const maxRankP = Math.max(...Array.from({ length: pp }, (_, r) => this._rankChunks(r).reduce((s, k) => s + k.p, 0)));
    const KU = maxRankP / UNIT0 * UPX > 1000 ? 4 : 1;
    const UNITP = UNIT0 * KU;
    const pTs = (c) => this.chunks[c].slotsP.map((v, i) =>
      this._epFrom ? this._epFrom[c][i] + (v - this._epFrom[c][i]) * this._ept : v);
    const pT = (c) => pTs(c).reduce((a, b) => a + b, 0);
    const rankP = (r) => this._rankChunks(r).reduce((s, k) => s + k.p, 0);   // labels/data: the exact NEW values
    const scale = UPX / UNITP;
    const wOf = (k) => pT(k.c) * scale;
    const rankW = (r) => this._rankChunks(r).reduce((s, k) => s + wOf(k), 0);
    // where a later-pass chunk DOCKS on its rank's row: after every earlier
    // chunk of that rank (the V's partner, or the wrap's lower chunks)
    const dockX = (k) => X0 + this._rankChunks(this._rankOf(k.c)).filter((k2) => k2.c < k.c).reduce((s, k2) => s + wOf(k2) + 3, 0);
    const H = D * PV + 30;
    const W = X0 + Math.max(...Array.from({ length: pp }, (_, r) => rankW(r))) + 270;
    // ONE authored motion — the UNROLL (crease-outward release, the first
    // flying chunk first, decelerating arrivals — the paper-unrolling read):
    // the fold plays the same video in reverse, at recoil tempo (see _go).
    // t stays raw; per-chunk easing lives inside each chunk's slice.
    const tG = fitEase(t);
    const STAG = 0.4, NFLY = Math.max(1, D - pp - 1);
    const tcOf = (c) => c < pp ? tG
      : 1 - fitEase(Math.min(1, Math.max(0, ((1 - t) - STAG * ((c - pp) / NFLY)) / (1 - STAG))));
    const lerpC = (a, b, c) => a + (b - a) * tcOf(c);
    const hv = this._hover;
    const hvRank = hv == null ? null : this._rankOf(hv);
    const folded = t > 0.5;
    const hotRow = (row) => hv != null && (folded ? this._rankOf(row) === hvRank : row === hv);
    const yRow = (row) => row * PV + (PV - RH) / 2;
    const B = [`<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family="system-ui">`];
    for (let row = 0; row < D; row++) {
      const k = this.chunks[row];
      if (hotRow(row)) B.push(`<rect x="0" y="${row * PV}" width="${W}" height="${PV}" fill="${C('#fff3d1')}"/>`);
      // axis: v-labels crossfade into s-labels on rows 0–PP−1; the vacated
      // rows keep their v-labels, dimmed (the chunks still live THERE)
      const sOp = row < pp ? t : 0, vOp = row < pp ? 1 - t : 1 - 0.65 * t;
      B.push(`<text x="${GUT - 4}" y="${yRow(row) + RH - 3}" text-anchor="end" font-size="9.5" fill="${C('#898781')}" opacity="${vOp.toFixed(2)}">v${row}</text>`);
      if (row < pp) B.push(`<text x="${GUT - 4}" y="${yRow(row) + RH - 3}" text-anchor="end" font-size="9.5" font-weight="600" fill="${C('#52514e')}" opacity="${sOp.toFixed(2)}">s${row}</text>`);
      const span = k.hi > k.lo ? `L${k.lo}–${k.hi - 1}` : '';
      B.push(`<text x="${LX}" y="${yRow(row) + RH - 3}" font-size="9" fill="${C('#898781')}">${k.emb ? 'emb·' : ''}${span}${k.mtp ? (span ? '·' : '') + 'mtp' : ''}${k.head ? '·head' : ''}</text>`);
    }
    // the model strip: 63 slots top-to-bottom (64 with an MTP layer),
    // UNBROKEN (verticality is the point — the fold moves cost, never the
    // model). Chunk spans wear a bracket, and a square-cornered leader
    // strings each span to its bar row; folded, a rank's leaders share a
    // routing lane, so both ends of the model visibly wire into one rank.
    const NS = 63 + MTP, HEADSLOT = 62 + MTP;
    const SH = (D * PV - 4) / NS, TOPY = 2;
    const slotY = (k2) => TOPY + k2 * SH;
    // a chunk's slot range: emb = slot 0, layer l = slot l+1, mtp = slot 62, head = the last
    const chunkSlots = (k) => ({ s0: k.emb ? 0 : k.hi > k.lo ? k.lo + 1 : k.mtp ? 62 : HEADSLOT,
      s1: k.head ? NS : k.mtp ? 63 : k.hi + 1 });
    const hotChunks = hv == null ? [] : folded ? this._rankChunks(hvRank) : [this.chunks[hv]];
    for (let sl = 0; sl < NS; sl++) {
      const cap = sl === 0 || sl >= 62;   // emb · mtp · head wear the cap shade
      const l = sl - 1;
      const inHot = hotChunks.some((k) => sl >= chunkSlots(k).s0 && sl < chunkSlots(k).s1);
      // highlight keeps each cell KIND's relative darkness (caps darkest,
      // dense mid, MoE light) — amber says "selected", shade still says what
      const fill = inHot ? (cap ? C('#8a5f00') : l < 3 ? C('#d19023') : C('#f6cd74'))
        : cap ? C('#8f8d86') : l < 3 ? C('#aba89f') : C('#dcdad2');
      const mtpCell = MTP && sl === 62;   // the MTP slot: dashed — placed, not modeled
      B.push(`<rect${cap ? '' : ` data-layer="${l}"`} x="${SL}" y="${slotY(sl).toFixed(1)}" width="${STW}" height="${(SH - 1).toFixed(1)}"${cap ? ' rx="2"' : ''} fill="${mtpCell ? 'none' : fill}"${mtpCell ? ` stroke="${fill}" stroke-dasharray="2 1.5"` : ''}/>`);
    }
    for (const k of this.chunks) {
      const { s0, s1 } = chunkSlots(k);
      const y0 = slotY(s0) + 0.5, y1 = slotY(s1) - 1.5, ym = (y0 + y1) / 2;
      const r = this._rankOf(k.c), down = k.c < pp;
      const rx = RX0 + r * 1.6;                        // the rank's chunks share a lane
      const yb = lerpC(yRow(k.c), yRow(down ? k.c : r), k.c) + RH / 2;   // leaders ride their chunk's stagger
      const hot2 = hotRow(k.c);
      const st = hot2 ? `stroke="${C('#7a5200')}" stroke-width="1.3"` : `stroke="${C('#c3c2b7')}" stroke-width="0.9"`;
      B.push(`<path d="M ${BRX} ${y0.toFixed(1)} h 3 V ${y1.toFixed(1)} h -3" fill="none" ${st}/>`);
      B.push(`<path d="M ${BRX + 3} ${ym.toFixed(1)} H ${rx.toFixed(1)} V ${yb.toFixed(1)} H ${GUT - 12}" fill="none" ${st}${hot2 ? '' : ' opacity="0.75"'}/>`);
    }
    // chunk bars: first-pass rows keep their bar; later-pass bars FLY to
    // their rank's row and dock after the bars already there. Vocab shares
    // wear a dashed outline (emb at the front, head at the back — the s0 imbalance)
    for (const k of this.chunks) {
      const c = k.c, down = c < pp, r = this._rankOf(c);
      const x = down ? X0 : lerpC(X0, dockX(k), c);
      const y = down ? yRow(c) : lerpC(yRow(c), yRow(r), c);
      const w = wOf(k);
      const hot2 = hv != null && (folded ? this._rankOf(c) === hvRank : c === hv);
      B.push(`<g data-chunk="${c}" data-params="${k.p}">`);
      // one segment per SLOT (emb · layers · head), hairline gaps — the bar
      // is the strip's span turned sideways: layers are countable, dense
      // runs visibly wider than an EP-sharded MoE layer, and the vocab
      // slots keep their dashed not-a-layer treatment
      const ws = pTs(c).map((v) => v * scale);
      let sx = x;
      for (const [i2, w2] of ws.entries()) {
        const vocab = (k.emb && i2 === 0) || (k.head && i2 === ws.length - 1);
        // dense layers wear a darker shade of their pass color (the strip's
        // dense-vs-MoE distinction, carried into the bar)
        const layer = k.lo + (i2 - (k.emb ? 1 : 0));
        const dense = !vocab && layer < 3;
        // color = DEPTH in the chain (a stepwise gradient, slot 0 light →
        // the last slot dark): the continuous version of the pass split.
        // Folded, a rank's pieces wear graded shades, and the V's rank 0
        // holds both extremes — the fold as a color story.
        const gT = (chunkSlots(k).s0 + i2) / (NS - 1);
        const segFill = fitColor(C('#bcd8f3'), C('#134a8e'), gT);
        const ink = gT < 0.5 ? C('#0b3d75') : C('#dcebfa');
        B.push(`<rect x="${sx.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(0.5, w2 - 1).toFixed(1)}" height="${RH}" fill="${segFill}"/>`);
        // only the ODD segments name themselves: emb/head (below) and the
        // dense blocks — ordinary MoE layers stay blank (they're the norm)
        if (dense && w2 > 30)
          B.push(`<text x="${(sx + w2 / 2).toFixed(1)}" y="${y + RH - 4}" text-anchor="middle" font-size="8.5" fill="${ink}">dense</text>`);
        if (vocab) {
          // the dashed outline alone says not-a-layer — no white wash: the
          // depth ramp must hold (the head is the last slot, the DARKEST point)
          B.push(`<rect x="${sx.toFixed(1)}" y="${y.toFixed(1)}" width="${Math.max(0.5, w2 - 1).toFixed(1)}" height="${RH}" fill="none" stroke="${gT < 0.5 ? C('#0b3d75') : C('#bcd8f3')}" stroke-dasharray="2.5 2"/>`);
          if (w2 > 30) B.push(`<text x="${(sx + w2 / 2).toFixed(1)}" y="${y + RH - 4}" text-anchor="middle" font-size="8.5" fill="${ink}">${k.emb && i2 === 0 ? 'emb' : 'head'}</text>`);
        }
        sx += w2;
      }
      if (hot2) B.push(`<rect x="${(x - 1).toFixed(1)}" y="${(y - 1).toFixed(1)}" width="${(w + 1).toFixed(1)}" height="${RH + 2}" fill="none" stroke="${C('#7a5200')}" stroke-width="1.2"/>`);
      B.push('</g>');
      const val = (pp2) => fmtBytes(pp2 * 2);   // bytes: the ruler's currency
      B.push(`<text x="${(x + w + 4).toFixed(1)}" y="${(y + RH - 3).toFixed(1)}" font-size="9.5" fill="${C('#898781')}" opacity="${(1 - t).toFixed(2)}">${val(k.p)}</text>`);
      // the rank total rides the LAST chunk docking on that row
      const last = this._rankChunks(r).at(-1);
      if (!down && last.c === c) B.push(`<text data-ranktotal="${r}" data-params="${rankP(r)}" x="${(x + w + 4).toFixed(1)}" y="${(y + RH - 3).toFixed(1)}" font-size="9.5" fill="${C('#898781')}" opacity="${t.toFixed(2)}">${val(rankP(r))}</text>`);
    }
    // the x-axis: these bars are LINEAR (everything else on the site is
    // log₂) — the unit RULER says so out loud, without atomizing the bars
    {
      const ay = D * PV + 4;
      const ext = Math.max(...Array.from({ length: pp }, (_, r) => rankW(r))) + UPX;
      B.push(`<line x1="${X0}" y1="${ay}" x2="${(X0 + ext).toFixed(1)}" y2="${ay}" stroke="${C('#c3c2b7')}" stroke-width="1"/>`);
      for (let u = 0; u * UPX <= ext; u++) {
        const x = X0 + u * UPX, major = u % 8 === 0;
        B.push(`<line x1="${x}" y1="${ay}" x2="${x}" y2="${ay + (major ? 5 : 2.5)}" stroke="${C('#c3c2b7')}" stroke-width="1"/>`);
        if (major && u > 0) B.push(`<text x="${x}" y="${ay + 14}" text-anchor="middle" font-size="8.5" fill="${C('#898781')}">${u / 8 * KU}</text>`);
      }
      B.push(`<text x="${(X0 + ext + 10).toFixed(1)}" y="${ay + 14}" font-size="9" fill="${C('#52514e')}">GiB bf16 · minor tick = ${128 * KU} MiB · LINEAR</text>`);
    }
    // whole-row hitboxes (stack, label, and bar band alike — easy hovering)
    for (let row = 0; row < D; row++)
      B.push(`<rect data-row="${row}" x="0" y="${row * PV}" width="${W}" height="${PV}" fill="transparent"/>`);
    B.push('</svg>');
    this._bars.innerHTML = B.join('');
    const svg = this._bars.querySelector('svg');
    svg.addEventListener('mouseover', (e) => {
      const r2 = e.target.closest('[data-row], g[data-chunk]');
      if (!r2) return;
      const c = r2.dataset.row != null ? +r2.dataset.row : +r2.dataset.chunk;
      if (c !== this._hover) { this._hover = c; this.render(); }
    });
    svg.addEventListener('mouseleave', () => { this._hover = null; this.render(); });
    this._readout();
  }
  _readout() {
    const hv = this._hover, folded = this._t > 0.5;
    const rng = (k) => [k.emb ? 'emb' : null, k.hi > k.lo ? `L${k.lo}–${k.hi - 1}` : null, k.mtp ? 'MTP' : null,
      k.head ? 'final norm + lm head' : null].filter(Boolean).join(' + ');
    if (hv == null) {
      this._ro.textContent = 'hover a row — left: the span of layers it holds · dashed = the vocab share (emb / head)';
    } else if (folded) {
      const r = this._rankOf(hv), ks = this._rankChunks(r);
      const p = ks.reduce((s, k) => s + k.p, 0);
      this._ro.textContent = `s${r} = ${ks.map((k) => `v${k.c}`).join(' + ')} = ${ks.map(rng).join(' · ')}`
        + ` — ${fmtP(p)} params on this rank${this.fold === 'reflect' && r === 0 ? ' (the fold’s heaviest: both ends of the model)' : ''}`;
    } else {
      const k = this.chunks[hv];
      this._ro.textContent = `v${hv} = ${rng(k)}`
        + ` (${k.dense ? `${k.dense} dense + ` : ''}${k.moe} MoE) — ${fmtP(k.p)} params`;
    }
  }
}
if (typeof customElements !== 'undefined' && !customElements.get('dsv3-pp-fold')) {
  customElements.define('dsv3-pp-fold', Dsv3PpFold);
}

if (typeof customElements !== 'undefined' && !customElements.get('dsv3-pp-schedule')) {
  customElements.define('dsv3-pp-schedule', Dsv3PpSchedule);
}

// ---- <dsv3-beat-deck> custom element ----------------------------------------------
// The optimization story as a SLIDESHOW over one fit chart. Each <section>
// child is a step: a FULL config in data-config (never a patch — so stepping
// past a hypothetical reverts it automatically), an optional
// data-hypothetical (dashed not-real card + tag), and the step's caption as
// its HTML. The reader advances explicitly; a forward step
// saves the last REAL step as the baseline and pours the bars to the new
// config through the layer's own knob tween, so ghosts and ▲/▼ badges
// narrate every move. Backward and jump navigation snap (no reverse fiction).
// The current step persists in the URL (d:<id>).
const DECK_CSS = `
dsv3-beat-deck { display: block; margin: 14px 0 26px; }
.deck-nav { display: flex; align-items: center; gap: 10px; margin: 0 0 6px; }
.deck-nav button { font: 12px ui-monospace, monospace; padding: 2px 12px; border: 1px solid var(--c-c3c2b7);
  border-radius: 4px; background: var(--c-ffffff); cursor: pointer; }
@media (hover: hover) { .deck-nav button:hover:not(:disabled) { background: var(--c-f3f2ee); } }
.deck-nav button:disabled { color: var(--c-dedcd3); cursor: default; }
.deck-step { font: 11px ui-monospace, monospace; color: var(--c-52514e); }
.deck-hyp { font: italic 11px system-ui; color: var(--c-898781); }
.deck-mod { font: italic 11px system-ui; color: var(--c-b05f00); }
.deck-nav button.deck-rst { color: var(--c-b05f00); border-color: var(--c-b05f00); padding: 1px 8px; }
.deck-cap { max-width: 760px; font-size: 15px; color: var(--c-1c1c1a); line-height: 1.5; }
.deck-cap p { margin: 6px 0; }
`;
class Dsv3BeatDeck extends HTMLElement {
  connectedCallback() {
    this._steps = [...this.querySelectorAll('section')].map((sec) => ({
      cfg: JSON.parse(sec.dataset.config ?? '{}'),
      hyp: sec.dataset.hypothetical,
      cap: sec.innerHTML,
    }));
    this.textContent = '';
    const style = document.createElement('style'); style.textContent = DECK_CSS;
    const nav = el('div', 'deck-nav');
    const btn2 = (t, cls) => { const b = document.createElement('button'); b.textContent = t; b.className = cls; return b; };
    this._first = btn2('« start', 'deck-first'); this._prev = btn2('‹ back', 'deck-prev');
    this._next = btn2('next ›', 'deck-next'); this._last = btn2('end »', 'deck-last');
    this._ind = el('span', 'deck-step');
    this._hyp = el('span', 'deck-hyp');   // hypothetical callout: lives in the FIXED nav row
    this._mod = el('span', 'deck-mod');   // detour callout: knobs fiddled off the slide
    this._rst = btn2('↩ back to the slide', 'deck-rst');   // the detour's undo
    this._rst.style.display = 'none';
    this._rst.onclick = () => this._rewind();
    nav.append(this._first, this._prev, this._ind, this._next, this._last, this._hyp, this._mod, this._rst);
    this._first.onclick = () => this.go(0);
    this._prev.onclick = () => this.go(this._i - 1);
    this._next.onclick = () => this.go(this._i + 1);
    this._last.onclick = () => this.go(this._steps.length - 1);
    // the chart: a snapshot-mode layer (no URL state of its own) that the
    // deck drives programmatically. Its knobs are LIVE: the reader may
    // fiddle mid-slide — a marked DETOUR that stepping rewinds first
    const l = this._layer = document.createElement('dsv3-layer');
    for (const [k, v] of [['snapshot', ''], ['live', ''], ['local', ''], ['cumulative', ''], ['lens', 'param-bytes'],
      ['recipe', 'bf16'], ['recompute', 'none'], ['controls', 'static'], ['detail', ''], ['nocaption', ''],
      ['knobs', 'cluster,pipeline,mesh,zero']])   // the per-slide config panel
      l.setAttribute(k, v);
    // any layer change (knob fiddles included) re-syncs the detour indicator
    const changed0 = l.changed?.bind(l) ?? (() => {});
    l.changed = (w) => { changed0(w); this._syncMod(); };
    this._cap = el('div', 'deck-cap');
    this.append(style, nav, l, this._cap);
    // reserve the TALLEST caption: slides must never change the deck's
    // height (the reader is mid-story — nothing below may shift)
    let capH = 0;
    for (const st2 of this._steps) { this._cap.innerHTML = st2.cap; capH = Math.max(capH, this._cap.offsetHeight); }
    this._cap.style.minHeight = `${capH}px`;
    this.tabIndex = -1; this.style.outline = 'none';
    this.addEventListener('keydown', (e) => {
      const d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0;
      if (d) { e.preventDefault(); this.go(this._i + d); }
    });
    this._i = -1;
    const st = this.id ? readUrlState('d:' + this.id) : null;
    this.go(Math.max(0, Math.min(this._steps.length - 1, st?.i ?? 0)), true);
  }
  // baseline: the last REAL (non-hypothetical) step before i — a hypothetical
  // is a parenthesis, never something later deltas are measured against
  _baseOf(i) {
    for (let j = i - 1; j >= 0; j--) if (!this._steps[j].hyp) return this._steps[j];
    return null;
  }
  // every step is a FULL config: unlisted knobs mean the DEFAULT, not
  // "whatever the previous slide left behind" — otherwise a hypothetical's
  // sched/recipe would leak forward
  _full(c) {
    return { ...CFG_DEFAULTS, recipe: 'bf16', recompute: 'none', ...c };
  }
  // the slide's exact config vs the layer's knobs: any difference is a
  // reader DETOUR (the caption no longer describes the chart)
  _fiddled() {
    if (!(this._i >= 0)) return false;   // pre-init renders fire changed() too
    const c = this._full(this._steps[this._i].cfg);
    const l = this._layer;
    // vpp/fold are DERIVED from pp — never knobs, never a detour
    const g = schedGeom(c);
    const stg = c.stage ?? peakStage(c.pp, c.ep, c.zero ?? 1, c.world, c.sched, g.vpp, g.fold, g.layout, g.a2a, c.tp ?? 1);
    return l.pp !== c.pp || l.ep !== c.ep || (l.zero ?? 0) !== c.zero || l.world !== c.world
      || (l.sched ?? '1f1b') !== c.sched || l.stage !== stg || l.vpp !== g.vpp || (l.hw ?? 'h100') !== (c.hw ?? 'h100')
      || !!l.a2a !== !!g.a2a || (l.gradB ?? 4) !== (c.gradB ?? 4) || !!l.fp8Params !== !!c.fp8Params || (l.tp ?? 1) !== (c.tp ?? 1);
  }
  _syncMod() {
    const f = this._fiddled();
    this._mod.textContent = f ? '✎ detour — the caption describes the slide, not your knobs' : '';
    this._rst.style.display = f ? '' : 'none';   // the way back rides the notice
    this._cap.style.opacity = f ? 0.55 : '';   // the caption visibly detaches
  }
  // pour the knobs back to the CURRENT slide's config (the detour's undo)
  _rewind() {
    const l = this._layer, prevF = l._snapLocal();
    l._applyCfg(this._full(this._steps[this._i].cfg));
    l._tweenFrames = 14; l._tweenLocal(prevF); l._tweenFrames = 12;
    this._syncMod();
  }
  go(i, instant = false) {
    if (i < 0 || i >= this._steps.length || i === this._i) return;
    const st = this._steps[i], l = this._layer, base = this._baseOf(i);
    // a detour rewinds FIRST (quick pour back to this slide's config), so the
    // step's delta always animates from the config its caption describes
    if (!instant && this._fiddled()) {
      this._rewind();
      setTimeout(() => this.go(i, instant), 14 * 16 + 40);
      return;
    }
    // fixed focus: the deck ALWAYS shows every bar and every sub-bar —
    // rows appearing/disappearing between slides reads as a scene change,
    // and the story is one scene (data-solo is accepted but ignored)
    for (const p of ['showWeights', 'showGrads', 'showOptim', 'showActs']) l[p] = true;
    l.setAttribute('parts', '');
    if (st.hyp != null) l.setAttribute('hypothetical', st.hyp); else l.removeAttribute('hypothetical');
    const prev = this._i >= 0 ? l._snapLocal() : null;   // where the reader is looking NOW
    const Ldisp = l._fitL;   // the on-screen layout — the baseline render below overwrites it
    if (base) {
      l._applyCfg(this._full(base.cfg)); l.render(); l._saveBaseline();
    } else l._pinCfg = null;
    l._applyCfg(this._full(st.cfg));
    this._i = i;   // before the tween: per-frame changed() reads it for the detour sync
    if (prev && !instant) {
      // slides narrate (~450 ms); knob fiddles stay at knob tempo (12)
      l._fitL = Ldisp; l._tweenFrames = 28; l._tweenLocal(prev); l._tweenFrames = 12;
    } else l.render();
    this._cap.innerHTML = st.cap;
    this._ind.textContent = `step ${i + 1} / ${this._steps.length}`;
    this._hyp.textContent = st.hyp != null ? (st.hyp || 'hypothetical — not what DSv3 did') : '';
    this._first.disabled = this._prev.disabled = i === 0;
    this._last.disabled = this._next.disabled = i === this._steps.length - 1;
    this._syncMod();
    if (this.id) writeUrlState('d:' + this.id, { i });
    this.focus({ preventScroll: true });
  }
}
if (typeof customElements !== 'undefined' && !customElements.get('dsv3-beat-deck')) {
  customElements.define('dsv3-beat-deck', Dsv3BeatDeck);
}

// ---- hidden debug mode: the page-wide audit overlay -------------------------
// Alt+A toggles it (persisted in the URL hash, key 'audit'): every fit chart
// grows a floating verification chip — the SAME auditFitCharts the battery
// runs, drawn over the page. All overlay logic lives in src/audit.js.
if (typeof document !== 'undefined' && typeof location !== 'undefined') {
  const sync = (on) => import('./audit.js').then((m) => m.auditOverlay(on));
  if (readUrlState('audit')?.on) sync(true);   // charts still upgrading: the overlay's observer catches them
  document.addEventListener('keydown', (e) => {
    if (e.code !== 'KeyA' || !e.altKey || e.ctrlKey || e.metaKey) return;
    const on = !readUrlState('audit')?.on;
    if (on) writeUrlState('audit', { on: 1 }); else clearUrlState('audit');
    sync(on);
  });
}
