// Canvas trace viewer, perfetto-flavored: WASD navigation, wheel zoom/pan,
// hover tooltips, click select, M marks, F focuses. Embeddable many times per
// page via the <dsv3-trace> custom element or the TraceViewer class.

import { fmtUs, fmtNum, DSV3 } from './model.js';
import { simulate, LEVELS } from './sim.js';
import { resolveMatmuls, MATMULS, RECIPES } from './memory.js';
import { blockGraph, analyze, RECOMPUTE_PRESETS } from './blockgraph.js';
import { PARAMS } from './params.js';

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

// byte components of the per-op strips (optim / consolidated variants), in the
// memory-bars stacked order with the memory-bars segment colors \u2014 the strips
// pre-teach the bar. bpp = bytes per parameter.
export const BYTE_COMPS = [
  // zthresh: the ZeRO level at which this component shards over its
  // replication group (1 = optimizer, 2 = + gradients, 3 = + weights/FSDP)
  { prop: 'showWeights', color: '#2a78d6', bpp: 2, zthresh: 3, label: 'weights (2 B/param)' },
  { prop: 'showGrads', color: '#eb6834', bpp: 4, zthresh: 2, label: 'gradients (fp32, 4 B/param)' },
  { prop: 'showOptim', color: '#1baf7a', bpp: 8, zthresh: 1, label: 'optimizer states (8 B/param)' },
];

// per-op activation buckets for the fit chart's acts breakdown — the same
// stashes the wire chips draw, grouped Haziza-style. 'other' is a catch-all
// remainder, so the list always PARTITIONS savedBytes (the visual audit's
// decomposition rule enforces the sum exactly).
export const ACT_BUCKETS = [
  { label: 'x0, x1 (residual)', ids: ['x0', 'x1'] },
  { label: 'norm outs', ids: ['norm1', 'norm2'] },
  { label: 'mla latents', ids: ['qkv_down', 'q_norm', 'kv_norm'] },
  { label: 'q · k,v', ids: ['q_up', 'kv_up'] },
  { label: 'attention out', ids: ['attn'] },
  { label: 'router state', ids: ['router'] },
  { label: 'dispatched tokens', ids: ['dispatch'] },
  { label: 'gate, up (routed+sh)', ids: ['gate_up'] },
  { label: 'swiglu out', ids: ['swiglu'] },
  { label: 'other', ids: [] },
];
// bucket an analysis's per-tensor stash bytes; the remainder lands in 'other'
export const actBucketsOf = (ana2) => {
  const named = ACT_BUCKETS.map((b) => b.ids.reduce((t, id) => t + (ana2.savedById?.[id] ?? 0), 0));
  named[ACT_BUCKETS.length - 1] = ana2.savedBytes - named.reduce((a, b) => a + b, 0);
  return named;
};

// fiat parallelism for the `local` variant (what one GPU holds): 2048 GPUs;
// PP degree, EP width, and ZeRO-1 are the layer's knobs. Layers split over
// the PP stages by a contiguous floor split: stage 0 gets the embedding (+
// the dense blocks while they last), the last stage gets final norm + lm head.
export const LOCAL_PAR = { world: 2048, pp: 16 };   // .pp = the default degree
// an AUTHORED config (snapshot from/to, deck steps) is complete by fiat:
// unlisted knobs mean these neutral nothing-applied defaults, never
// "whatever the widget's live defaults happen to be" — a published figure
// must not drift when the interactive defaults do
const CFG_DEFAULTS = { world: 2048, pp: 1, ep: 1, zero: 0, sched: '1f1b', vpp: 1, fold: 'reflect' };
// virtual-stage placement: with VPP = vpp chunks per rank the chain is
// vpp·pp virtual stages deep; 'wrap' places stage v on rank v mod pp
// (Megatron interleaving), 'reflect' bounces each pass (ZB-V / DualPipeV:
// rank 0 → pp−1 → 0 → …), so with even vpp the chain both starts AND ends
// on rank 0. DualPipeV ≡ vpp 2 + reflect.
export const vstagesOf = (r, pp, vpp = 1, fold = 'reflect') =>
  Array.from({ length: vpp }, (_, c) =>
    c * pp + (fold === 'reflect' && c % 2 ? pp - 1 - r : r));
export const ppStage = (s, pp = LOCAL_PAR.pp, vpp = 1, fold = 'reflect') => {
  const D = vpp * pp;
  const seg = (i) => {
    const lo = Math.floor(61 * i / D), hi = Math.floor(61 * (i + 1) / D);
    const dense = Math.max(0, Math.min(hi, 3) - Math.min(lo, 3));
    return { lo, hi, layers: hi - lo, dense, moe: hi - lo - dense };
  };
  const vs = vstagesOf(s, pp, vpp, fold);
  const segs = vs.map(seg);
  const sum = (k) => segs.reduce((t, g) => t + g[k], 0);
  return { segs, layers: sum('layers'), dense: sum('dense'), moe: sum('moe'),
    emb: vs.includes(0), head: vs.includes(D - 1) };
};

// save-everything bf16 activation bytes for ONE layer × one 4096-token
// microbatch (the local model's activation quantum), computed once
let ACT_LAYER_B = 0;
export const actLayerBytes = () => ACT_LAYER_B ||=
  analyze(blockGraph('moe', DSV3, resolveMatmuls({ recipe: 'bf16' }), 4096), {}, false).savedBytes * 4096;
// 1F1B admission per virtual stage: stage v of a D-deep chain holds D − v
// forward chunk-stashes at steady state (warmup depth; assumes ≥ D
// microbatches per step). A rank's residency in microbatch-equivalents is
// the sum over its hosted virtual stages, ÷ vpp (each chunk is 1/vpp of the
// rank's layers). vpp 1 → the plain 1F1B staircase pp − s; vpp 2 + reflect
// (DualPipeV) → a UNIFORM pp + ½ on every rank (the two depths always sum
// to 2pp+1 — 8.5 at PP8, vs the DSv3 paper's coarse PP+1 bound); wrap
// (Megatron interleaving) concentrates at rank 0: pp(vpp+1)/2 − s.
// 'one' = a single microbatch in flight.
export const inflightOf = (sched, s, pp, vpp = 1, fold = 'reflect') => {
  if (sched === 'one') return 1;
  const D = vpp * pp, s2 = Math.min(s, pp - 1);
  return vstagesOf(s2, pp, vpp, fold).reduce((t, v) => t + (D - v), 0) / vpp;
};

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
  const B = [`<text class="grplabel" x="2" y="9">${L.hdr}</text>`];
  // the infeasible region is SHADED, not a line; its label sits ON TOP,
  // leaving the bottom axis to the power-of-two labels
  B.push(`<rect x="${f1(L.capPx)}" y="${topY - 2}" width="${f1(x0 + bw - L.capPx)}" ` +
    `height="${f1(aY - topY - 1)}" fill="#0b0b0b" opacity="0.07"/>`);
  // unit swatch legend floats right in the header — only when the strip
  // squares it explains are actually mounted (pointless on bars-only views)
  if (L.unit) B.push(`<rect x="${x0 + bw - 96}" y="3" width="5" height="4" fill="#898781"/>` +
    `<text class="dims" x="${x0 + bw - 87}" y="9">${L.unit}</text>`);
  for (let e = LO; e <= HI; e += 1)   // the ×2 grid
    B.push(`<line x1="${f1(gx(e))}" y1="${topY - 2}" x2="${f1(gx(e))}" y2="${f1(aY - 3)}" stroke="#e1e0d9" stroke-width="1"/>`);
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
        `<text class="dims" x="12" y="${f1(y + 5.5)}">· ${r.name}</text>` +
        `<rect data-bar="${s.bar}" data-true="${s.true}" x="${f1(s.x0)}" y="${f1(y)}" width="${f1(Math.max(0.5, s.x1 - s.x0))}" height="5" fill="${s.color}" opacity="${fo(s.op)}"/>` +
        (r.ghost ? `<rect x="${x0}" y="${f1(y)}" width="${f1(Math.max(0.5, r.ghost.px - x0))}" height="5" ` +
          `fill="none" stroke="${r.ghost.color}" stroke-width="1" stroke-dasharray="2 2" opacity="${fo(r.ghost.op)}"/>` : '') +
        (r.val ? `<text class="dims" data-role="val:${r.id}" data-true="${r.val.true}" data-pin="${r.val.pin}" x="${f1(r.val.x)}" y="${f1(y + 5.5)}"${op(r.val.op)}>${r.val.text}</text>` : '') +
        `</g>`);
      continue;
    }
    // gutter: the name alone (whole-row hitbox; click to solo)
    B.push(`<g${r.prop ? ` data-prop="${r.prop}" style="cursor:pointer"` : ''}${op(r.nameOp)}>` +
      (r.prop ? `<rect x="0" y="${f1(y - 2)}" width="${x0 - 4}" height="${FIT_ROWH}" fill="transparent"/>` : '') +
      `<text class="dims" data-role="name:${r.id}" data-true="${r.abs}" x="2" y="${f1(y + 7)}" fill="${r.color}" font-weight="600">${r.name}</text></g>`);
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
    if (r.val) B.push(`<text class="dims" data-role="val:${r.id}" data-true="${r.val.true}" data-pin="${r.val.pin}" x="${f1(r.val.x)}" y="${f1(y + 7)}"${op(r.val.op)}>${r.val.text}</text>`);
  }
  // map-style DISTANCES legend: on a log axis a span IS a factor, so anchor
  // the important ones — the mesh dims (DP 2048 · EP 64 · PP 16) and a
  // halving — as one true-scale ruler below the axis
  const dy = aY + 18;
  const dpx = (f) => x0 + Math.log2(f) / (HI - LO) * bw;
  B.push(`<text class="dims" x="2" y="${f1(dy + 3)}">a span is a factor:</text>`);
  B.push(`<line x1="${x0}" y1="${f1(dy)}" x2="${f1(dpx(2048))}" y2="${f1(dy)}" stroke="#898781" stroke-width="1"/>`);
  for (const [f, lab] of [[1, ''], [2, '×2'], [16, '×16 (PP)'], [64, '×64 (EP)'], [2048, '×2048 (DP)']]) {
    B.push(`<line data-fac="${f}" x1="${f1(dpx(f))}" y1="${f1(dy - 4)}" x2="${f1(dpx(f))}" y2="${f1(dy + 4)}" stroke="#898781" stroke-width="1"/>`);
    if (lab) B.push(`<text class="dims" x="${f1(dpx(f))}" y="${f1(dy + 13)}" text-anchor="middle">${lab}</text>`);
  }
  // the pinned save label shares the legend band, right-aligned (its line is
  // always present now, so pinning still never reflows)
  if (L.lbl) B.push(`<text class="dims" x="${x0 + bw}" y="${f1(dy + 3)}" text-anchor="end">${L.lbl}</text>`);
  // the scrub overlay: cursor affordance AND arming region live exactly on
  // the bars band — not the captions, not below the axis
  B.push(`<rect class="scrub" x="${x0}" y="${topY - 2}" width="${bw}" height="${f1(aY - topY + 1)}" ` +
    `fill="transparent" style="cursor:col-resize"/>`);
  B.push(`<text class="dims" x="${f1(L.capPx)}" y="9" text-anchor="middle">80 GiB (H100)</text>`);
  return `<svg width="${w}" height="${f1(L.HB)}" viewBox="0 0 ${w} ${f1(L.HB)}">${B.join('')}</svg>`;
}

// the PP stage holding the most resident bytes under the local model (all
// components on, vocab counted on the end stages, activations under the
// schedule) — the default stage to show: the fully loaded rank.
export const peakStage = (pp, ep, zero, world = LOCAL_PAR.world, sched = '1f1b', vpp = 1, fold = 'reflect') => {
  const dp = world / pp;
  const bpp = (cls) => BYTE_COMPS.reduce((t, c) =>
    t + (zero >= c.zthresh ? c.bpp / (cls === 'e' ? dp / ep : dp) : c.bpp), 0);
  const moeExp = PARAMS.expert * DSV3.routedExperts;
  const dB = PARAMS.denseBlock * bpp('d');
  const mB = (PARAMS.moeBlock - moeExp) * bpp('d') + (moeExp / ep) * bpp('e');
  let best = 0, bestV = -1;
  for (let s2 = 0; s2 < pp; s2++) {
    const g = ppStage(s2, pp, vpp, fold);
    const v = g.dense * dB + g.moe * mB
      + (((g.emb ? 1 : 0) + (g.head ? 1 : 0)) * PARAMS.embed * bpp('d'))
      + g.layers * actLayerBytes() * inflightOf(sched, s2, pp, vpp, fold);
    if (v > bestV) { bestV = v; best = s2; }
  }
  return best;
};

// compact stage-option label: '0: L0\u20132 \u00b7 3d+emb', '15: L57\u201360 \u00b7 head \u00b7 peak',
// a rank's hosted chunks joined: '0: L0\u20132+L58\u201360 \u00b7 3d+emb+head \u00b7 peak'
// (l = the local-lens layer whose knobs pick the split)
export const stageLabelOf = (o, l) => {
  const pp = l.pp ?? LOCAL_PAR.pp, world = l.world ?? LOCAL_PAR.world;
  const g = ppStage(o, pp, l.vpp ?? 1, l.fold);
  const rng = (s) => s.lo >= s.hi ? '' : s.lo === s.hi - 1 ? `L${s.lo}` : `L${s.lo}\u2013${s.hi - 1}`;
  const range = g.segs.map(rng).filter(Boolean).join('+') || '\u2014';
  const tags = [g.dense ? `${g.dense}d` : '', g.emb ? 'emb' : '',
    g.head ? 'head' : ''].filter(Boolean).join('+');
  const pk = o === peakStage(pp, l.ep, l.zero ?? 1, world, l.sched, l.vpp ?? 1, l.fold) ? ' \u00b7 peak' : '';
  return `${o}: ${range}${tags ? ` \u00b7 ${tags}` : ''}${pk}`;
};

// parameter-count formatter for the dims parentheticals ('(29M \u00d7256)' / '(7.5B)')
// change-badge magnitude formatting (▲×N / ▼×N), shared with the visual audit
export const facNum = (v) => v >= 100 || Math.abs(v - Math.round(v)) < 0.02 * v ? String(Math.round(v)) : v.toFixed(1);

export const fmtP = (n) => n >= 1e9 ? (n / 1e9).toFixed(1) + 'B'
  : n >= 9.95e6 ? Math.round(n / 1e6) + 'M' : n >= 1e6 ? (n / 1e6).toFixed(1) + 'M'
  : n >= 1e3 ? (n / 1e3).toFixed(1).replace(/\.0$/, '') + 'K' : String(n);
import { downloadTrace, openInPerfetto } from './trace.js';

// shared light-card tooltip style (trace, memory bars, schematic)
const TIP_CARD = 'position: absolute; pointer-events: none; background: #fff; color: #1c1c1a; padding: 6px 9px;' +
  ' border: 1px solid #c3c2b7; border-radius: 5px; display: none; box-shadow: 0 2px 10px rgba(11,11,11,0.12);';

// Validated categorical palette (dataviz skill, light surface #fcfcfb).
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
.tv { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: #0b0b0b;
  border: 1px solid #e1e0d9; border-radius: 6px; background: #fcfcfb; overflow: hidden; }
.tv-bar { display: flex; align-items: center; gap: 8px; padding: 5px 8px; border-bottom: 1px solid #e1e0d9; flex-wrap: wrap; }
.tv-title { font-weight: 600; }
.tv-stats { color: #52514e; }
.tv-sp { flex: 1; }
.tv button { font: 11px system-ui; padding: 2px 8px; border: 1px solid #c3c2b7; border-radius: 4px;
  background: #fff; color: #0b0b0b; cursor: pointer; }
.tv button:hover { background: #f3f2ee; }
.tv-legend { display: flex; gap: 10px; padding: 3px 8px; border-bottom: 1px solid #e1e0d9;
  color: #52514e; font-size: 11px; flex-wrap: wrap; }
.tv-legend span { display: inline-flex; align-items: center; gap: 4px; }
.tv-legend i { width: 9px; height: 9px; border-radius: 2px; display: inline-block; }
.tv-wrap { position: relative; }
.tv canvas { display: block; outline: none; }
.tv-tip { ${TIP_CARD} font-size: 11px; max-width: 340px; z-index: 5; line-height: 1.45; }
.tv-tip b { color: #0b0b0b; }
.tv-foot { padding: 3px 8px; border-top: 1px solid #e1e0d9; color: #52514e; font-size: 11px;
  min-height: 16px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.tv-help { position: absolute; top: 6px; right: 6px; background: rgba(11,11,11,.92); color: #fff;
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
      memEl.style.color = m.fits ? '#52514e' : '#d03b3b';
      if (!m.fits) memEl.style.fontWeight = '600';
      this.statsEl.append(memEl);
    }
    this.buildRows();
    this.legendEl.innerHTML = '';
    for (const cat of this.catsPresent) {
      const s = el('span'); const i = el('i'); i.style.background = CATS[cat].c;
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
    ctx.fillStyle = '#fcfcfb'; ctx.fillRect(0, 0, w, h);
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
    ctx.fillStyle = '#f9f9f7'; ctx.fillRect(0, 0, this.w, RULER);
    const target = 110 * this.tpp;
    const pow = Math.pow(10, Math.floor(Math.log10(target)));
    const step = [1, 2, 5, 10].map(m => m * pow).find(s => s >= target) ?? pow * 10;
    ctx.font = '10px system-ui'; ctx.fillStyle = '#898781'; ctx.strokeStyle = '#e1e0d9';
    const start = Math.floor(this.tl / step) * step;
    for (let t = start; t < this.tOf(this.w); t += step) {
      const x = this.xOf(t);
      if (x < GUTTER) continue;
      ctx.beginPath(); ctx.moveTo(x, RULER - 4); ctx.lineTo(x, this.h); ctx.stroke();
      ctx.fillText(fmtUs(t), x + 3, 12);
    }
    ctx.strokeStyle = '#c3c2b7';
    ctx.beginPath(); ctx.moveTo(0, RULER + .5); ctx.lineTo(this.w, RULER + .5); ctx.stroke();
  }

  drawRow(row, y) {
    const { ctx } = this;
    if (row.kind === 'header') {
      ctx.fillStyle = '#f3f2ee'; ctx.fillRect(0, y, this.w, row.h);
      ctx.fillStyle = '#0b0b0b'; ctx.font = 'bold 10px system-ui';
      ctx.fillText(row.label, 6, y + 12);
      return;
    }
    ctx.fillStyle = '#898781'; ctx.font = '10px system-ui';
    ctx.fillText(row.label, 14, y + 12);
    const tr = this.tOf(this.w);
    row.lanes.forEach((lane, li) => {
      const ly = y + li * LANE + 1, lh = LANE - 2;
      let i = lowerBound(lane, this.tl);
      if (i > 0 && lane[i - 1].ts + lane[i - 1].dur > this.tl) i--;
      let mx0 = null, mx1 = 0, mcat = null; // merged run of sub-pixel slices
      const flushMerged = () => {
        if (mx0 == null) return;
        ctx.fillStyle = CATS[mcat]?.c ?? '#898781';
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
        const cat = CATS[s.cat] ?? { c: '#898781', ink: '#fff' };
        ctx.fillStyle = cat.c;
        ctx.fillRect(x0, ly, Math.max(sw - 0.5, 0.6), lh);
        if (sw > 34) {
          ctx.fillStyle = cat.ink;
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
    ctx.strokeStyle = '#2a78d6';
    for (const x of [x0, x1]) { ctx.beginPath(); ctx.moveTo(x, RULER); ctx.lineTo(x, this.h); ctx.stroke(); }
    const label = fmtUs(this.mark.t1 - this.mark.t0);
    ctx.font = 'bold 10px system-ui';
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = '#2a78d6'; ctx.fillRect((x0 + x1) / 2 - tw / 2 - 4, RULER + 2, tw + 8, 14);
    ctx.fillStyle = '#fff'; ctx.fillText(label, (x0 + x1) / 2 - tw / 2, RULER + 13);
  }

  drawSelection() {
    const s = this.sel;
    const y = this.rowY(this.selRow) + this.selLane * LANE + 1;
    if (y == null) return;
    this.ctx.strokeStyle = '#0b0b0b'; this.ctx.lineWidth = 1.5;
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
const DT_STYLE = { bf16: '#52514e', mxfp8: '#2a78d6', fp32: '#9c3a96' };
// the diagram's visual-language tokens (docs/diagram-grammar.md) — one
// definition, scoped into each widget's stylesheet (the anatomy plan too)
export const tokensCss = (s) => `
${s} .wire { stroke: #898781; stroke-width: 1.2; fill: none; }
${s} .box { fill: #fff; stroke: #c3c2b7; }
${s} .op { fill: #f3f2ee; stroke: #e1e0d9; }
${s} .comm { fill: #f3f1fb; stroke: #6b5bd2; }
${s} .res { fill: #fcfcfb; stroke: #c3c2b7; stroke-dasharray: 3 2; }
${s} .grp { fill: none; stroke: #e1e0d9; }
${s} .name { font: 600 11px system-ui; fill: #0b0b0b; }
${s} .dims { font: 9px system-ui; fill: #898781; }
${s} .oplabel { font: 10.5px system-ui; fill: #52514e; }
${s} .grplabel { font: italic 10px system-ui; fill: #898781; }
${s} .plus { font: 600 12px system-ui; fill: #52514e; }
`;
// the local-knob control-strip styles (steppers, grouped rows) — shared by
// the layer's mini-head and <dsv3-pp-schedule>'s replicated pipeline group
const knobCss = (s) => `
${s} .pargrp { display: inline-flex; flex-direction: column; gap: 2px;
  border: 1px solid #e1e0d9; border-radius: 6px; padding: 3px 8px 5px; align-self: stretch; }
${s} .pargrp.center { justify-content: center; }
${s} .parlab { font: italic 10px system-ui; color: #898781; }
${s} .parrow { display: flex; align-items: center; gap: 5px; min-height: 20px; }
${s} .stp { display: inline-flex; align-items: stretch; }
${s} .stp button { font: 12px ui-monospace, monospace; width: 20px; padding: 0 0 1px; border: 1px solid #c3c2b7; background: #fff; color: #52514e; cursor: pointer; }
${s} .stp button:hover:not(:disabled) { background: #f3f2ee; }
${s} .stp button:disabled { color: #dedcd3; cursor: default; }
${s} .stp button:first-child { border-radius: 4px 0 0 4px; }
${s} .stp button:last-child { border-radius: 0 4px 4px 0; }
${s} .stp button + button { border-left: none; }
${s} .stp button.on { background: #f3f2ee; color: #0b0b0b; font-weight: 600; cursor: default; }
${s} .stp button { width: auto; min-width: 20px; padding: 0 5px 1px; }
${s} .stp select.v { font: 11px ui-monospace, monospace; min-width: 4ch; padding: 2px 5px;
  border: 1px solid #c3c2b7; border-left: none; border-right: none; border-radius: 0;
  background: #fff; appearance: none; -webkit-appearance: none; text-align: center;
  text-align-last: center; cursor: pointer; }
${s} select { font: 12px system-ui; padding: 2px 6px; border: 1px solid #c3c2b7; border-radius: 4px; background: #fff; }
`;

const LAYER_CSS = `
dsv3-layer { display: block; margin: 14px 0 26px; }
.lv { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: #0b0b0b;
  border: 1px solid #e1e0d9; border-radius: 6px; background: #fcfcfb; padding: 10px 12px; position: relative; }
.lv-tip { ${TIP_CARD} font-size: 11.5px; max-width: 360px; z-index: 7; line-height: 1.5; white-space: pre-line; }
.lv-tip.pinned { border-color: #eda100; box-shadow: 0 2px 10px rgba(237,161,0,0.3); }
.lv-head { display: flex; align-items: center; gap: 8px; padding-bottom: 6px; color: #52514e; flex-wrap: wrap; }
.lv-head select { font: 12px system-ui; padding: 2px 6px; border: 1px solid #c3c2b7; border-radius: 4px; background: #fff; }
.lv-head .savebox { margin-left: auto; display: inline-flex; gap: 6px; align-items: flex-start;
  padding-left: 12px; border-left: 1px solid #e1e0d9; align-self: flex-start; }
${knobCss('.lv-head')}
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
dsv3-layer[snapshot] .lv-head button:disabled, dsv3-layer[snapshot] .lv-head select:disabled { opacity: 1; color: #0b0b0b; cursor: default; }
dsv3-layer[snapshot] .lv-head .stp button.on:disabled { background: #f3f2ee; color: #0b0b0b; }
dsv3-layer[snapshot] .lv-head .stp button:disabled:not(.on) { color: #c3c2b7; }
dsv3-layer[snapshot] .lv-head .stp:has(select.v) button { display: none; }
dsv3-layer[snapshot] .lv-head .stp:has(select.v) select.v { border-left: 1px solid #c3c2b7; border-right: 1px solid #c3c2b7; border-radius: 4px; }
dsv3-layer[snapshot] .lv-head select:disabled { appearance: none; -webkit-appearance: none; background: #fff; }
.lv-hyptag { font: italic 10.5px system-ui; color: #898781; padding-bottom: 4px; }
.lv-bar svg { display: block; margin: 2px 0 6px; max-width: 100%; height: auto; }
.lv-bar { position: relative; }
.lv-ruler { display: none; position: absolute; background: rgba(237, 161, 0, 0.12);
  border-left: 1px solid #0b0b0b; border-right: 1px solid #0b0b0b; pointer-events: none; }
.lv-ruler-lab { position: absolute; top: -2px; left: 100%; margin-left: 5px; white-space: nowrap;
  font: 11px ui-monospace, monospace; color: #0b0b0b; background: #fff8ea; padding: 1px 4px;
  border: 1px solid #eda100; border-radius: 3px; }
${tokensCss('.lv')}
.lv select.dt { font: 600 10px system-ui; width: 100%; height: 20px; border: 1px solid #c3c2b7;
  border-radius: 3px; background: #fff; }
.lv button.st { display: block; width: 100%; height: 18px; font: 10px system-ui; border-radius: 3px;
  cursor: pointer; text-align: left; padding: 0 5px; overflow: hidden; white-space: nowrap; }
.lv .st-save { background: #fff8ea; border: 1px solid #eda100; color: #0b0b0b; }
.lv .st-redo { background: #f3f2ee; border: 1px dashed #898781; color: #52514e; }
.lv button.st.mode { width: 24px; padding: 0; text-align: center; height: 20px; }
.lv button.st.dtb { width: 52px; padding: 0; text-align: center; height: 20px; font-weight: 600;
  background: #fff; border: 1px solid #c3c2b7; }
.lv text.tensor { font: 10px system-ui; }
.lv .tsave { fill: #7a5200; font-weight: 600; }
.lv .tdim { fill: #898781; font-weight: 400; }
.lv .micro { fill: #f7f6f1; stroke: #d8d6cb; }
.lv svg.hlm > :not(.hl):not(defs) { opacity: 0.3; }
.lv g[data-op].hl .dims { fill: #52514e; font-weight: 600; }
.lv .microlabel { font: italic 10px system-ui; fill: #52514e; }
.lv .tredo { fill: #52514e; font-style: italic; }
.lv .tidle { fill: #a8a69e; }
.lv-note { color: #898781; font-size: 11px; padding-top: 6px; max-width: 640px; }
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
    this.ep = st?.ep ?? 64;
    this.pp = st?.pp ?? LOCAL_PAR.pp;
    this.zero = st?.zero ?? (st?.zero1 === false ? 0 : 1);   // ZeRO level 0–3 (1 = DSv3)
    this.world = st?.world ?? LOCAL_PAR.world;               // cluster size (GPUs)
    this.sched = st?.sched === 'dpv' ? '1f1b' : st?.sched ?? '1f1b';   // admission: '1f1b' | 'one' (a single microbatch)
    this.vpp = st?.sched === 'dpv' ? 2 : st?.vpp ?? 2;       // virtual-pipeline degree; the default is DSv3's own schedule (VPP2·reflect = DualPipeV)
    this.fold = st?.fold ?? 'reflect';                       // chunk placement: 'reflect' (V/DualPipeV) | 'wrap' (Megatron)
    // default to the PEAK stage — the fully loaded rank is the story; the
    // selector is there to peek at the lighter ones
    this.stage = Math.min(st?.stage ?? peakStage(this.pp, this.ep, this.zero, this.world, this.sched, this.vpp, this.fold), this.pp - 1);
    // cumulative: every parameter parenthetical multiplies by the selected
    // kind's block count (×3 dense / ×58 MoE); the tabs hide — the kind then
    // comes from the plan selector alone
    this.cumulative = st?.cumulative ?? this.hasAttribute('cumulative');
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
    // recompute preset an authored row config pins
    const allIds = [...new Set([...Object.keys(RECOMPUTE_PRESETS.full), 'x1'])];
    const marksEff = (this.getAttribute('controls') ?? 'full') === 'static' ? {} : this.marks;
    const savedMap = Object.fromEntries(allIds.map(id => [id, marksEff[id] !== false]));
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
      kind: this.kind, cumulative: this.cumulative,
      showWeights: this.showWeights, showOptim: this.showOptim,
      showGrads: this.showGrads, showActs: this.showActs,
      ep: this.ep, stage: this.stage, pp: this.pp, zero: this.zero, world: this.world, sched: this.sched,
      vpp: this.vpp, fold: this.fold,
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
    const mutate = () => {
      const to = this.marks[ids[0]] === false ? true : false;
      for (const id of ids) { if (to) delete this.marks[id]; else this.marks[id] = false; }
    };
    if (this.hasAttribute('local') && this.getAttribute('lens') === 'param-bytes') {
      const prev = this._snapLocal(); mutate(); this._tweenLocal(prev);
    } else { mutate(); this.render(); this.changed(); }
  }
  // local-knob mutations shared by the head controls and external drivers
  // (<dsv3-pp-schedule>): callers go through setLocal so every change tweens
  setLocal(mutate) { const prev = this._snapLocal(); mutate(); this._tweenLocal(prev); }
  _setPP(v) {
    const world = this.world ?? LOCAL_PAR.world;
    this.pp = v;
    this.ep = Math.min(this.ep, world / v);
    this.stage = peakStage(v, this.ep, this.zero ?? 1, world, this.sched, this.vpp, this.fold);   // stage indices don't survive a resplit — jump to the new peak
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
      label: `EP${this.ep}·PP${this.pp}${(this.vpp ?? 1) > 1 ? `·VPP${this.vpp}${this.fold === 'wrap' ? 'w' : 'V'}` : ''}·stage ${this.stage}·ZeRO-${this.zero ? this.zero : 'off'}·${this.sched === 'one' ? '×1mb' : '1F1B'}·${this.world} GPUs`,
    };
  }
  // apply an authored config patch (snapshot 'from'/'to', sandbox jumps):
  // plain state keys plus recipe/recompute presets
  _applyCfg(patch) {
    const { recipe, recompute, stage, ...rest } = patch;
    Object.assign(this, rest);
    if (recipe) { this.setAttribute('recipe', recipe); this.matmuls = resolveMatmuls({ recipe }); }
    if (recompute) { this.setAttribute('recompute', recompute); this.marks = { ...RECOMPUTE_PRESETS[recompute] }; }
    this.stage = stage ?? peakStage(this.pp, this.ep, this.zero ?? 1,
      this.world ?? LOCAL_PAR.world, this.sched, this.vpp, this.fold);
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
      sched: this.sched ?? '1f1b', vpp: this.vpp ?? 1, fold: this.fold ?? 'reflect', cum: !!this.cumulative,
      // the pre-change analysis: stash-affecting knobs (precision, marks,
      // fp8ᵀ) lerp the diagram's chip squares between old and new bytes
      anaPrev: this._anaMemo?.ana };
  }
  _tweenLocal(prev) {
    this.changed(true);
    const kindOf = (S) => ppStage(Math.min(S.stage, S.pp - 1), S.pp, S.vpp, S.fold).moe ? 'moe' : 'dense';
    if (kindOf(prev) !== kindOf(this._snapLocal())) { this.render(); return; }
    this._frames((t) => { this._vtween = { t, prev }; }, () => { this._vtween = undefined; });
  }
  render() {
    this.innerHTML = '';
    // local lens: the kind follows the selected PP stage (stage 0 holds the
    // 3 dense blocks; every other stage holds only MoE blocks)
    if (this.hasAttribute('local') && this.getAttribute('lens') === 'param-bytes')
      this.kind = ppStage(this.stage ?? 1, this.pp, this.vpp, this.fold).moe ? 'moe' : 'dense';
    const style = document.createElement('style'); style.textContent = LAYER_CSS;
    const root = el('div', 'lv');
    // progressive disclosure: controls="static|marks|dtype|full" gates which
    // controls are rendered (the diagram and its derived annotations always draw)
    const cmode = this.getAttribute('controls') ?? 'full';
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
    for (const name of Object.keys(RECIPES)) {
      const o = document.createElement('option'); o.value = o.textContent = name; preset.append(o);
    }
    // recognize the current matmul dtypes as a recipe (dtype buttons may have
    // moved us off the attribute's preset), else show "custom"
    const mmKey = (m) => MATMULS.map(x => m[x.id]).join(',');
    const curRecipe = Object.keys(RECIPES).find(k => mmKey(resolveMatmuls({ recipe: k })) === mmKey(this.matmuls));
    preset.value = curRecipe ?? 'bf16';
    if (!curRecipe) {
      const o = document.createElement('option'); o.value = o.textContent = 'custom'; o.selected = true; preset.append(o);
    }
    const localTween = (mutate) => {   // stash knobs animate like every other knob
      if (this.hasAttribute('local') && this.getAttribute('lens') === 'param-bytes') {
        const prev = this._snapLocal(); mutate(); this._tweenLocal(prev);
      } else { mutate(); this.render(); this.changed(); }
    };
    preset.onchange = () => {
      if (preset.value === 'custom') return;
      localTween(() => {
        this.setAttribute('recipe', preset.value);
        this.matmuls = resolveMatmuls({ recipe: preset.value });
      });
    };
    if (this._ctl.dtype) head.append(preset);
    if (this._ctl.marks) head.append(' · recompute: ');
    const rsel = document.createElement('select');
    for (const name of Object.keys(RECOMPUTE_PRESETS)) {
      const o = document.createElement('option'); o.value = o.textContent = name; rsel.append(o);
    }
    const marksKey = (m) => Object.keys(m).filter(k => m[k] === false).sort().join(',');
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
      i.type = 'number'; i.value = get(); i.style.cssText = 'width:44px;font:12px system-ui;padding:1px 4px;border:1px solid #c3c2b7;border-radius:3px;';
      i.onchange = () => { set(Math.max(1, +i.value || 1)); this.render(); this.changed(true); };
      head.append(i);
    };
    if (this.view === 'combined') {
      numIn('×layers', () => this.dispLayers, (v) => this.dispLayers = v);
      numIn('×in-flight', () => this.dispInflight, (v) => this.dispInflight = v);
    }
    const reset = document.createElement('button');
    reset.textContent = 'reset';
    reset.style.cssText = 'font:11px system-ui;margin-left:auto;padding:2px 8px;border:1px solid #c3c2b7;border-radius:4px;background:#fff;cursor:pointer;';
    reset.onclick = () => {
      this.setAttribute('recipe', this._origRecipe);
      this.matmuls = resolveMatmuls({ recipe: this._origRecipe });
      this.marks = { ...RECOMPUTE_PRESETS[this._origRecompute] };
      this.view = 'combined';
      this.dispLayers = +(this.getAttribute('xlayers') ?? 61);
      this.dispInflight = +(this.getAttribute('xinflight') ?? 1);
      this.transposed = this.hasAttribute('transposed');
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
    tl.style.cssText = 'display:inline-flex;align-items:center;gap:4px;margin-left:8px;color:#52514e;';
    tl.title = 'Hopper tile-scaled fp8 (1×128 per-row scales): stashes feeding wgrad GEMMs are kept in ' +
      'BOTH quantization orientations — per-row scales don’t transpose. MXFP8’s power-of-2 block ' +
      'scales requantize the transpose exactly; leave off for Blackwell.';
    const tcb = document.createElement('input');
    tcb.type = 'checkbox'; tcb.checked = this.transposed;
    tcb.onchange = () => localTween(() => { this.transposed = tcb.checked; });
    tl.append(tcb, 'fp8ᵀ dual stash');
    if (this._ctl.dtype) head.append(tl);
    // local: the multiplier is the stage's block count, not the whole model's
    const KBLK = this.hasAttribute('local') && this.getAttribute('lens') === 'param-bytes'
      ? (this.kind === 'dense' ? ppStage(this.stage ?? 1, this.pp, this.vpp, this.fold).dense
        : ppStage(this.stage ?? 1, this.pp, this.vpp, this.fold).moe)
      : this.kind === 'dense' ? (DSV3.denseLayers ?? 3) : DSV3.layers - (DSV3.denseLayers ?? 3);
    const mkCumBtn = () => {
      const b = document.createElement('button');
      b.style.cssText = 'font:11px ui-monospace,monospace;padding:2px 8px;border:1px solid #c3c2b7;' +
        'border-radius:4px;background:#fff;cursor:pointer;margin-left:8px;min-width:9ch;box-sizing:content-box;';
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
      b.style.cssText = 'font:11px ui-monospace,monospace;padding:2px 8px;border:1px solid #c3c2b7;' +
        'border-radius:4px;background:#fff;cursor:pointer;margin-left:8px;';
      b.textContent = this.flatDims ? '24576' : '128\u00d7192';
      b.title = 'toggle sizes: factored (128\u00d7192) vs multiplied out (24576)';
      b.onclick = () => { this.flatDims = !this.flatDims; this.render(); this.changed(true); };
      return b;
    };
    const dl = document.createElement('label');
    dl.style.cssText = 'display:inline-flex;align-items:center;gap:4px;margin-left:8px;color:#52514e;';
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
    const marksEff2 = this._ctl.quant || this._ctl.marks ? this.marks : {};
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
      };
    }
    const ana = this._anaMemo.ana;
    let barSlot = null;   // the local fit bar renders under the parallelism row
    if (cmode !== 'static') root.append(head);
    else {
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
      // local toggles between one block and the stage total
      const cumCtl = this.hasAttribute('optim') && !this.hasAttribute('consolidated') ? [] : [mkCumBtn()];
      // no kind select when MLA-only (kind-independent) or when the tabs carry the flip
      if (KN('blocks')) {
        if (SCOPE === 'mla' || this.hasAttribute('tabs')) mini2.append(...sizeCtl, ...cumCtl);
        else mini2.append('block: ', mkKindSel(), ...(sizeCtl.length ? [' · '] : []), ...sizeCtl, ...cumCtl);
      }
      if (this.hasAttribute('local')) {
        // the fiat parallelism: 2048 GPUs fixed; EP width (or off), PP degree
        // (powers of two), the PP stage, and ZeRO-1 are the knobs. The kind
        // follows the stage; each stage option names its layer assignment.
        const mkSel = (opts, val, label, set) => {
          const s = document.createElement('select');
          for (const o of opts) s.append(new Option(label(o), o));
          s.value = String(val);
          s.onchange = () => { const prev = this._snapLocal(); set(+s.value); this._tweenLocal(prev); };
          return s;
        };
        const pp = this.pp ?? LOCAL_PAR.pp;
        const stageLabel = (o) => stageLabelOf(o, this);
        // EP/PP step by powers of two: a segmented − value + control
        const POW2 = [1, 2, 4, 8, 16, 32, 64];
        const mkStep = (get, set, fmt, max = 64, opts = POW2, min = 1) => {
          const wrap = el('span', 'stp');
          // the value chip is itself a dropdown: click the number to jump
          const val = document.createElement('select'); val.className = 'v';
          for (const o of opts.filter((o) => o <= max && o >= min)) val.append(new Option(fmt(o), o));
          val.value = String(get());
          val.onchange = () => { const prev = this._snapLocal(); set(+val.value); this._tweenLocal(prev); };
          const btn = (txt, dir, max) => {
            const b = document.createElement('button');
            b.textContent = txt; b.type = 'button';
            b.disabled = dir < 0 ? get() <= min : get() >= max;
            b.onclick = () => {
              const prev = this._snapLocal();
              set(dir < 0 ? get() / 2 : get() * 2);
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
        const txt2 = (t3) => { const sp = el('span'); sp.style.cssText = 'color:#52514e;font-size:11px;'; sp.textContent = t3; return sp; };
        const knob = (n, e2) => { e2.dataset.knob = n; return e2; };
        const gCluster = grp2('cluster');
        gCluster.append(row2(txt2('GPUs'), knob('gpus', mkStep(() => world,
          (v) => { this.world = v; this.ep = Math.min(this.ep, v / this.pp); },
          String, 16384, [128, 256, 512, 1024, 2048, 4096, 8192, 16384], 128))));
        const gPipe = grp2('pipeline');
        gPipe.append(
          row2(txt2('PP'), knob('pp', mkStep(() => this.pp, (v) => this._setPP(v), String, 64)),
            txt2('stage'), knob('stage', Object.assign(
              // fixed width: option labels change with every PP/ZeRO/VPP move
              // (ranges, the roaming ' · peak' tag) and the select must not
              // resize with them — worst-case VPP4 labels clip when closed
              mkSel([...Array(pp).keys()], this.stage, stageLabel, (v) => { this.stage = v; }),
              { style: 'width: 212px' }))));
        const gMesh = grp2('SPMD mesh');
        const txtR = (t3) => {   // right-aligned row labels, so the mesh rows line up
          const sp = txt2(t3);
          sp.style.cssText += 'display:inline-block;width:64px;text-align:right;';
          return sp;
        };
        gMesh.append(
          row2(txtR('non-expert:'), txt2(`DP ${world / pp}`)),
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
        // the schedule, decomposed: admission (1F1B steady state vs a single
        // microbatch), VPP degree (chunks per rank), and chunk placement —
        // reflect (the V: ZB-V/DualPipeV) vs wrap (Megatron interleaving).
        // DualPipeV ≡ VPP2 + reflect: uniform PP+½ in flight, emb+head on
        // rank 0. The schedule is an assumption worth breaking open.
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
        const sw2 = knob('sched', seg2([['1f1b', '1F1B'], ['one', '×1 mb']],
          () => this.sched ?? '1f1b', (k) => { this.sched = k; }));
        const vw2 = knob('vpp', mkStep(() => this.vpp ?? 1,
          (v) => { this.vpp = v; this.stage = peakStage(this.pp, this.ep, this.zero ?? 1, world, this.sched, v, this.fold); },
          String, 4, [1, 2, 4]));
        const fw2 = knob('fold', seg2([['wrap', 'wrap'], ['reflect', 'V']],
          () => this.fold ?? 'reflect',
          (k) => { this.fold = k; this.stage = peakStage(this.pp, this.ep, this.zero ?? 1, world, this.sched, this.vpp ?? 1, k); }));
        gPipe.append(row2(txt2('sched'), sw2, txt2('VPP'), vw2, fw2));
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
        leg.style.cssText = 'color:#52514e;margin-left:10px;font-size:11px;white-space:nowrap;';
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
          leg.append(...comps2.map((c) => cb(c.label, c.color, c.prop)));
          if (cons2) leg.append(cb(this.hasAttribute('local')
            ? `saved activations (bf16, ×4096 tok × ${inflightOf(this.sched ?? '1f1b', this.stage ?? 1, this.pp ?? LOCAL_PAR.pp, this.vpp, this.fold)} in flight)`
            : 'saved activations (bf16, ×4096 tokens)', '#eda100', 'showActs'));
          const u = el('span');
          u.style.cssText = 'display:inline-flex;align-items:center;gap:3px;';   // swatch centers regardless of baseline
          u.innerHTML = `· ${sw('#898781')} <span>= ${fmtBytes(unit)}</span>`;
          leg.append(u);
        } else {
          leg.style.cssText += 'display:inline-flex;align-items:center;gap:3px;';
          leg.innerHTML = `${sw('#2a78d6')} <span>= ${fmtBytes(unit)}</span>`;
        }
        if (KN('legend')) mini2.append(leg);
      }
      if (this.hasAttribute('local')) {
        // pin a baseline config: the log bars then carry ticks at the pinned
        // values and ×N/÷N factors — "I ×256'ed this and it /256'ed that"
        const mkBtn = (txt, title, fn) => {
          const b = document.createElement('button');
          b.style.cssText = 'font:11px ui-monospace,monospace;padding:2px 8px;border:1px solid #c3c2b7;' +
            'border-radius:4px;background:#fff;cursor:pointer;';
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
        if (!this._pinCfg?.state) { rst.disabled = true; rst.title = 'save a config first'; rst.style.color = '#c3c2b7'; rst.style.cursor = 'default'; }
        saveBox.append(rst, reset);   // factory reset (built above; also clears the save)
        reset.textContent = 'reset all';
        reset.style.cssText = 'font:11px ui-monospace,monospace;padding:2px 8px;border:1px solid #c3c2b7;' +
          'border-radius:4px;background:#fff;cursor:pointer;';   // match the save cluster's face
        if (KN('save')) mini.append(saveBox);
        // the preexisting AC + precision knobs (built above for the full
        // head; the static head never displays it, so they move here)
        const plab = (t2) => { const sp = el('span'); sp.style.cssText = 'color:#52514e;font-size:11px;margin-left:8px;'; sp.textContent = t2; return sp; };
        if (KN('prec')) mini2.append(plab('precision:'), preset, plab('recompute:'), rsel, tl);
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
    const M2 = this.view === 'combined' ? this.dispLayers * this.dispInflight * 4096 : 1;
    const parts = [
      !this._ctl.quant ? '' :
      (this.view === 'combined'
        ? `stashed for backward: ${(ana.savedBytes * M2 / 2 ** 30).toFixed(1)} GiB total = ${(ana.savedBytes / 1024).toFixed(0)} KiB/token\u00b7layer \u00d7 ${this.dispLayers} layers \u00d7 ${this.dispInflight} in-flight \u00d7 4096 tokens (set layers/in-flight to your PP stage to tally with the memory bars) \u00b7 `
        : `stashed for backward: ${(ana.savedBytes / 1024).toFixed(0)} KiB/token\u00b7layer \u00b7 `) +
      `backward replays +${(ana.replayFrac * 100).toFixed(0)}% of fwd FLOPs` +
      (ana.replayComm.length ? ` + a2a ${ana.replayComm.join('+')}` : '') + '.',
      this._ctl.marks
        ? 'The \ud83d\udcbe/\u21bb button on each op chooses save-output vs recompute-in-backward; the wire below shows the derived result \u2014'
        : this._ctl.quant
          ? 'Each wire label is an output, tagged with the recompute policy\u2019s derived result \u2014'
          : 'Each wire label is an output, tagged with whether backward reads it \u2014',
      this._ctl.quant
        ? '\u2193 \u2191 \u21c5 saved for backward, read by the op below / above / both (\u25aa = 4 KiB/token; violet boxes = communication), ' +
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
          + `RoPE is fused into the q/kv paths${this._ctl.quant ? ' and always recomputed' : ''} (negligible).`,
      !this._ctl.quant ? '' :
      'The block strip inside each op is its FLOP cost as time at peak, scaled so the block\u2019s largest op fills one row (' +
      'mxfp8 counted half \u2014 2\u00d7 peak; fp32 counted double \u2014 half peak; dtype colors here and on the saved-tensor tags: blue mxfp8, dark bf16, plum fp32); ' +
      'the lm head uses the same scale \u2014 per-token vocab work, independent of depth. Norms/SwiGLU ' +
      'get a muted fig-leaf block (bandwidth-bound, compute precision unspecified).',
      this._ctl.dtype ? 'One click on a dtype button cycles bf16 \u2192 mxfp8 \u2192 fp32.' : '',
      this._ctl.quant
        ? 'The tally at right totals fwd + bwd (2\u00d7 fwd \u2014 dgrad + wgrad; sdpa likewise) + replay'
          + (this._ctl.marks ? ' \u2014 marking ops \u21bb grows its replay row.' : '.')
        : '',
      this._ctl.dtype
        ? 'The fp8\u1d40 toggle models Hopper tile-scaled fp8: any fp8 stash a wgrad GEMM reads is kept in both ' +
          'quantization orientations (\u1d40\u00d72 tags) because per-row scales don\u2019t transpose; MXFP8\u2019s ' +
          'power-of-two block scales requantize exactly, so Blackwell keeps one.'
        : '',
    ];
    note.textContent = parts.filter(Boolean).join(' ');
    // nocaption: the page explains the diagram in its own prose
    if (!this.hasAttribute('nocaption')) {
      const foot = el('div', 'lv-foot2');
      foot.append(note);
      if (cmode !== 'static') foot.append(this._tallySvg);
      root.append(foot);
    }
    if (this._ctl.quant) this.attachTip(root);   // no tooltips on the structure-only tier
    this.append(style, root);
    this.applyHl();
  }
  // spreadsheet-style highlighting: mark the boxes whose parameters a
  // clicked tally row sums (dsv3-param-tally drives this)
  // ids = null clears; ids = [] fades EVERYTHING (a selected sum with no
  // cells in this diagram — e.g. the embedding row greys the block out)
  highlightOps(ids) { this._hl = ids ? new Set(ids) : null; this.applyHl(); }
  applyHl() { applyHighlight(this, this._hl); }
  buildSvg(ana, anaM = null) {
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
    const EPn = this.ep ?? 64, PPn = this.pp ?? LOCAL_PAR.pp, STG = this.stage ?? 1;
    const ZL = this.zero ?? 1;                               // ZeRO level (0 off · 1 optim · 2 +grads · 3 +weights)
    const WORLD = this.world ?? LOCAL_PAR.world;
    const DPn = WORLD / PPn;
    const EDP = WORLD / PPn / EPn;                           // expert-DP (EP=1: = DP — no expert parallelism)
    const SCHED = this.sched ?? '1f1b';
    const VPPn = this.vpp ?? 1, FOLD = this.fold ?? 'reflect';
    const stg = ppStage(STG, PPn, VPPn, FOLD);
    // knob tween (local): squares pour between the OLD and NEW configuration —
    // each component's effective factor (bytes/param × EP share × stage ×N)
    // lerps with this._vtween.t; numbers snap to the new config
    const IFN = inflightOf(SCHED, STG, PPn, VPPn, FOLD);     // microbatches in flight on this stage
    const Snow = { ep: EPn, pp: PPn, stage: STG, zero: ZL, world: WORLD, sched: SCHED, vpp: VPPn, fold: FOLD, cum: !!this.cumulative };
    const dLoc = (S) => {
      const g = ppStage(Math.min(S.stage, S.pp - 1), S.pp, S.vpp, S.fold);
      const kmul = this.kind === 'dense' ? g.dense : g.moe;
      const dp = (S.world ?? LOCAL_PAR.world) / S.pp;
      return { mult: S.cum ? kmul : 1, eFrac: 1 / S.ep,
        acts: (S.cum ? kmul : 1) * inflightOf(S.sched ?? '1f1b', S.stage, S.pp, S.vpp, S.fold),
        bpp: (c, cls) => (S.zero ?? 1) >= c.zthresh ? c.bpp / (cls === 'e' ? dp / S.ep : dp) : c.bpp };
    };
    const fEff = (c, cls, S) => {
      const d = dLoc(S);
      return d.bpp(c, cls) * d.mult * (cls === 'e' ? d.eFrac : 1);
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
        ? `<tspan fill="#d03b3b" font-weight="600"> ▲×${facNum(r)}</tspan>`
        : `<tspan fill="#0b0b0b" font-weight="600"> ▼×${facNum(1 / r)}</tspan>`;
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
    const bppOf = (c, cls) => !LOCAL || ZL < c.zthresh ? c.bpp
      : c.bpp / (cls === 'e' ? EDP : DPn);
    const BPPT = (cls = 'd') => COMPS.reduce((t, c) => t + (this[c.prop] ? bppOf(c, cls) : 0), 0);
    const clsOf = (id) => LOCAL && this.kind === 'moe' && (id === 'ffn_gate_up' || id === 'ffn_down') ? 'e' : 'd';
    // param-bytes always shows sizes multiplied out (factored ×256 byte chains
    // pull no weight there; the sizes toggle is hidden in that lens)
    const FLAT = this.flatDims || PBYTES;
    // static/params tiers can never fill the in-box strip band (FLOP strips
    // need a dtype tier, param strips need the bytes lens) — compact boxes
    // instead of reserving space for strips that can't appear
    const BQ = this._ctl.quant || PBYTES;
    const BH = BQ ? 38 : 32;      // bold matmul box height
    const HBH = BQ ? 60 : 32;     // narrow half-column box (buttons + strip sit below the text)
    // quant tiers carry byte-quantity labels (e.g. attention's lse) that need
    // more room between the columns; the static tier keeps its published width
    const W = 290, C1 = 60,
      C2 = (ONLY === 'ffn' ? 60 : !this._ctl.quant ? 512 : this.detail ? 576 : 524)
        + (ONLY !== 'ffn' && this._ctl.quant && this.hasAttribute('tabs') ? 20 : 0);
    const SX1 = C1 + 22, SX2 = C2 + 22, RAIL1 = C1 - 26;
    const WIDTH = ONLY === 'mla' ? C1 + W + 250
      : C2 + W + (this.detail ? (this._ctl.quant ? 264 : 224) : 180); // right margin fits aux labels (+ shared column in detail; quant byte tags are wider)
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
      e: (CUM ? KMUL : 1) * BPPT('e') / EPn,
      a: (CUM ? KMUL : 1) * IFN,
    };
    // cumulative is always shown multiplied out — factored ×256 ×58 chains
    // are noise; the sizes toggle keeps governing dims and per-block factoring
    // param-bytes lens: the VISIBLE bytes per parameter (bf16 weights = 2 B,
    // + 8 B optimizer when shown), formatted as binary bytes — the number on a
    // box always totals exactly the squares drawn in it
    const fmtPB = (nParams, cls) => fmtBytes(nParams * BPPT(cls));
    const fmtPV = (n, cls = 'd') => PBYTES ? fmtPB(n, cls) : fmtP(n);
    const pk = (n, noK = false) => {
      if (PBYTES && !BPPT()) return '';   // nothing visible, nothing to number
      if (LOCAL && this.partSel != null && this.partSel !== 1) return '';   // norms/micro ops are non-expert
      const v = (CUM && !noK ? fmtPV(n * KMUL) : fmtPV(n)) + facTxt('d');
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
    const marks = this._ctl.quant || this._ctl.marks ? this.marks : {};   // static: save everything
    const state = (id) => {
      const n = ana.byId[id];
      // the checkpoint-anchor lock only means something when a replay exists
      // to terminate at it: with recompute none, x0 is just a saved tensor
      if (n.always) return ana.replayed.size ? 'pin' : (ana.neededSaved.has(id) ? 'save' : 'idle');
      if (marks[id] === false) return 'redo';
      return ana.neededSaved.has(id) ? 'save' : 'idle';
    };
    // one-click precision toggle (bf16 -> mxfp8 -> fp32), hidden below the dtype tier
    const dtBtn = (id, x, y) => !this._ctl.dtype ? '' :
      `<foreignObject x="${x}" y="${y}" width="52" height="20">` +
      `<button xmlns="http://www.w3.org/1999/xhtml" class="st dtb" data-dt="${id}" style="color:${DT_STYLE[dt(id)]}" ` +
      `title="cycle precision: bf16 / mxfp8 / fp32">${dt(id)}</button></foreignObject>`;
    const modeBtn = (ids, x, y) => {
      if (!this._ctl.marks) return '';       // hidden below the marks tier
      const st = state(ids[0]);
      if (st === 'pin') return '';
      const redo = this.marks[ids[0]] === false;
      return `<foreignObject x="${x}" y="${y}" width="26" height="20">` +
        `<button xmlns="http://www.w3.org/1999/xhtml" class="st mode st-${redo ? 'redo' : 'save'}" ` +
        `data-mark="${ids.join(',')}" title="save output for backward vs recompute this op during backward">${redo ? '↻' : '💾'}</button></foreignObject>`;
    };
    const blockGrid = (bytes, x, y) => {
      const n = Math.max(1, Math.round(bytes / 1024 / 4)), per = 16;
      let s = '';
      for (let i = 0; i < n; i++)
        s += `<rect x="${x + (i % per) * 6}" y="${y + Math.floor(i / per) * 6}" width="5" height="5" fill="#eda100"/>`;
      return { svg: s, rows: Math.ceil(n / per) };
    };
    const fmtB = (bytes) => bytes >= 1024 ? (bytes / 1024).toFixed(1) + ' KiB' : bytes + ' B';
    // combined view: totals over the block column — layers × in-flight microbatches × 4096 tokens
    const M = this.view === 'combined' ? this.dispLayers * this.dispInflight * 4096 : 1;
    const fmtMem = (bytes) => {
      if (M === 1) return fmtB(bytes);
      const b = bytes * M;
      return b >= 2 ** 30 ? (b / 2 ** 30).toFixed(1) + ' GiB' : b >= 2 ** 20 ? (b / 2 ** 20).toFixed(0) + ' MiB' : (b / 1024).toFixed(0) + ' KiB';
    };
    // FLOP cost strip inside each op box, MFU-style: TIME at peak
    // (bf16-equivalent; mxfp8 counted half since its peak is 2x). Scaled so the
    // largest op in the transformer block fills exactly one row of 30 blocks;
    // the lm head takes however many rows it needs at the same scale.
    // Colored by the op's precision; vector ops get a muted fig-leaf block.
    const flopEq = (flopsTok, d) => flopsTok * (d === 'mxfp8' ? 0.5 : d === 'fp32' ? 2 : 1);
    const opDt = (id) => {
      const n = ana.byId[id];
      if (!n) return 'vector';
      if (n.opKind === 'matmul' || n.opKind === 'attn') return dt(id === 'gate_up' ? 'ffn_gate_up' : id);
      return 'vector';
    };
    // per-op FLOP formulas (per token) for the hover tooltips
    const FLOP_EXPR = {
      norm1: '≈ 8 · 7168 — bandwidth-bound vector op, compute precision unspecified',
      norm2: '≈ 8 · 7168 — bandwidth-bound vector op, compute precision unspecified',
      qkv_down: '2 · 7168 · (1536 + 576)',
      q_up: '2 · 1536 · 128·192', kv_up: '2 · 512 · 128·256',
      attn: '2 · 128 heads · (192 + 128) · 4096/2 (causal average context)',
      o_proj: '2 · 128·128 · 7168', router: '2 · 7168 · 256',
      ...(this.kind === 'dense' ? {
        gate_up: '2 · (2 · 7168 · 18432)', swiglu: '≈ 6 · 18432 — elementwise',
        ffn_down: '2 · (18432 · 7168)',
      } : {
        gate_up: '2 · (2 · 7168 · 2048) · 9 experts', swiglu: '≈ 6 · 2048 · 9 — elementwise',
        ffn_down: '2 · (2048 · 7168) · 9 experts',
      }),
      lm_head: '2 · 7168 · 129280',
      dispatch: 'a2a communication — no FLOPs', combine: 'a2a communication — no FLOPs',
    };
    const escAttr = (s) => esc(s).replace(/"/g, '&quot;').replace(/\n/g, '&#10;');
    const boxTip = (id, dimsNote, paramId = id) => {
      const n = ana.byId[id];
      const f = n?.flopsTok ? `${fmtNum(n.flopsTok)} FLOP/token = ${FLOP_EXPR[id] ?? ''}` : (FLOP_EXPR[id] ?? '');
      const pc = exactParam(paramId);
      return ` data-tip="${escAttr([f, dimsNote, pc == null ? '' : `parameters: ${pc.toLocaleString('en-US')}`].filter(Boolean).join('\n'))}"`;
    };
    // dtype the sim ascribes to a stashed tensor (the dtype of the matmul whose
    // backward reads it — a real degree of freedom, so we surface it)
    const dtOf = (n) => {
      const b = n.outBytes / n.elems;
      return b >= 3.5 ? 'fp32' : b >= 1.7 ? 'bf16' : 'mxfp8';
    };
    // 32/row: a power of two, so parallelism shards divide the strips cleanly
    // (EP64 on the ×58 MoE gate/up strip = 32·58/64 = 29 whole squares/rank),
    // and the byte unit lands exact: largestOp.moe/32 · 2 B = 448 MiB/square
    const FLOP_ROW = 32;
    const CHIP_ROW = 16;   // amber wire-chip squares wrap sooner (chips sit between columns)
    const FLOP_UNIT = Math.max(...['qkv_down', 'q_up', 'kv_up', 'attn', 'o_proj', 'router', 'gate_up', 'swiglu', 'ffn_down']
      .filter(id => ana.byId[id])            // dense blocks have no router
      .map(id => flopEq(ana.byId[id].flopsTok, opDt(id)))) / FLOP_ROW;
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
      if (!nParams || !(ABS || OPTIM)) return 0;
      return (Math.max(1, Math.ceil(stripCells(nParams, cls) / row)) - 1) * 6;
    };
    const paramBlocks = (x, y, nParams, cls, row = FLOP_ROW) => {
      if (!PBYTES || !nParams) return;
      const cells = compCells(nParams, cls);
      let g = '', i = 0;
      for (const { c, n } of cells)
        for (let k = 0; k < n; k++, i++)
          g += `<rect x="${x + (i % row) * 6}" y="${y + Math.floor(i / row) * 6}" width="5" height="4" fill="${c.color}"/>`;
      if (!i) {
        const top = cells.reduce((b2, r) => r.f > b2.f ? r : b2, { f: 0 });
        if (top.f > 0.02)
          g += `<rect x="${x + 0.4}" y="${y + 0.4}" width="4.2" height="3.2" fill="none" stroke="${top.c.color}" stroke-width="0.8"/>`;
      }
      P.push(g);
    };
    const flopBlocks = (x, y, flopsTok, dt2) => {
      if (!flopsTok || !this._ctl.quant) return 0;
      const n = Math.max(1, Math.round(flopEq(flopsTok, dt2) / FLOP_UNIT));
      const color = DT_STYLE[dt2] ?? '#c3c2b7';
      let s = '';
      for (let i = 0; i < n; i++)
        s += `<rect x="${x + (i % FLOP_ROW) * 6}" y="${y + Math.floor(i / FLOP_ROW) * 6}" width="5" height="4" fill="${color}"/>`;
      P.push(s);
      return Math.ceil(n / FLOP_ROW);
    };
    // who reads this saved tensor in backward: consumer below (↓), the
    // producer's own backward above (↑), or both (↕)
    const needDir = (ids) => {
      const by = new Set(ids.flatMap(i => [...(ana.neededBy.get(i) ?? [])]));
      const up = ids.some(i => by.has(i));
      const down = [...by].some(b => !ids.includes(b));
      return up && down ? '⇅' : up ? '↑' : '↓';
    };
    // ov (optional): display-split override for a chip that shows part of one
    // graph node — { name, tdims, frac } (bytes and grid scale by frac)
    const tensorChip = (ids, x, y, ov) => {
      const id = ids[0], st = state(id), n = ana.byId[id];
      const bytes = ids.reduce((t, i) => t + ana.byId[i].outBytes * (ana.dual.has(i) ? 2 : 1), 0) * (ov?.frac ?? 1);
      const dualTag = ids.some(i => ana.dual.has(i)) ? ' ᵀ×2' : '';
      const name0 = ov?.name ?? n.tensor;
      if (PONLY) {
        // consolidated: the saved activations live ON THE WIRES, like the AC
        // diagram — amber chips (name + bytes for one 4096-token microbatch)
        // with squares at the same global unit (mostly hollow: individually
        // sub-square is the honest picture). Gaps (chipSpace) unchanged.
        const m = CONS ? cmult('showActs') : 0;
        if (m && st === 'redo') {   // recomputed: named, no bytes — the AC feedback
          P.push(`<g opacity="${m.toFixed(3)}"><text class="tensor tredo" x="${x}" y="${y + 8}">↻ ${esc(name0.replace(' (checkpoint anchor)', ''))}</text></g>`);
          return 12;
        }
        if (!m || (st !== 'save' && st !== 'pin')) return 12;
        // cumulative: every block's stash is resident — chips follow the ×N
        // convention (labels snap, squares grow with the tween like the strips)
        const b4096 = bytes * 4096 * (LOCAL ? (CUM ? KMUL : 1) * IFN : CUM ? KMUL : 1);
        const chipF = LOCAL ? actsT : ABS ? stripMul : CUM ? KMUL : 1;
        // stash-knob tween: the squares pour between the OLD and NEW bytes
        const VA = this._vtween?.prev?.anaPrev;
        const bytesT = VA
          ? ids.reduce((t2, i2) => {
            const nb = ana.byId[i2].outBytes * (ana.dual.has(i2) ? 2 : 1);
            const pb2 = (VA.byId[i2]?.outBytes ?? nb / (ana.dual.has(i2) ? 2 : 1)) * (ana.dual.has(i2) ? 2 : 1);
            return t2 + pb2 + (nb - pb2) * this._vtween.t;
          }, 0) * (ov?.frac ?? 1)
          : bytes;
        const full = Math.round(bytesT * 4096 * chipF / (PB_UNIT * 2));
        const nsq = Math.round(full * m), hollow = !nsq && m >= 0.5 && chipF > 0;
        const name = esc(name0.replace(' (checkpoint anchor)', ''));
        const lock = st === 'pin' ? ' 🔒' : '';
        // narrow fork columns (ov.short) get two lines — one line would run
        // into the neighbouring column's spine at ×58 byte widths
        // squares wrap well before the strip width (chips sit between
        // columns); the wire gaps grow with the rows (chipSpaceA prices them)
        const CROW = ov?.short ? 8 : CHIP_ROW;
        const [sqX, sqY] = ov?.short ? [x + 60, y + 17] : [x, y + 12];
        let g = ov?.short
          ? `<text class="tensor tsave" x="${x}" y="${y + 8}">${needDir(ids)} ${name}${lock}</text>` +
            `<text class="tensor tsave" x="${x}" y="${y + 21}">${fmtBytes(b4096)}${facTxt('a')}</text>`
          : `<text class="tensor tsave" x="${x}" y="${y + 8}">${needDir(ids)} ${name} · ${fmtBytes(b4096)}${facTxt('a')}${lock}</text>`;
        if (hollow) g += `<rect x="${sqX + 0.4}" y="${sqY + 0.4}" width="4.2" height="3.2" fill="none" stroke="#eda100" stroke-width="0.8"/>`;
        else for (let i = 0; i < nsq; i++)
          g += `<rect x="${sqX + (i % CROW) * 6}" y="${sqY + Math.floor(i / CROW) * 6}" width="5" height="4" fill="#eda100"/>`;
        P.push(`<g opacity="${m.toFixed(3)}">${g}</g>`);
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
          : `<text class="tensor tsave" x="${x}" y="${y + 8}">${needDir(ids)} ${name} <tspan class="tdim">· ${sz}</tspan></text>`);
        return h;
      }
      if (st === 'save' || st === 'pin') {
        P.push(`<text class="tensor tsave" x="${x}" y="${y + 8}">${needDir(ids)} ${esc(name0)} · ${fmtMem(bytes)} ` +
          `<tspan fill="${DT_STYLE[dtOf(n)]}">${dtOf(n)}${dualTag}</tspan>${st === 'pin' ? ' 🔒' : ''}</text>`);
        const g = blockGrid(bytes, x, y + 12);
        P.push(g.svg);
        h = 12 + g.rows * 6 + 2;
      } else if (st === 'redo') {
        // ov.short: narrow fork columns drop the suffix (the ↻ glyph carries it)
        P.push(`<text class="tensor tredo" x="${x}" y="${y + 8}">↻ ${esc(name0)}${ov?.short ? '' : ' — recomputed'}</text>`);
      } else {
        P.push(`<text class="tensor tidle" x="${x}" y="${y + 8}">· ${esc(name0)}</text>`);
      }
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
        const b = ids.reduce((t, i) => t + anaX.byId[i].outBytes, 0) * 4096 * chipF;
        const rows = Math.max(1, Math.ceil(Math.round(b / (PB_UNIT * 2)) / CHIP_ROW));
        return Math.round(18 + (rows * 6 - 2) * mA);
      }
      // worst case per element: bf16 (2 B), or dual fp8 orientations (2 × 1.03 B)
      const perElem = this.transposed ? 2 * (1 + 1 / 32) : 2;
      const worst = ids.reduce((t, i) => t + anaX.byId[i].elems * perElem, 0);
      const rows = Math.ceil(Math.max(1, Math.round(worst / 1024 / 4)) / 16);
      return 12 + rows * 6 + 2;
    };
    const chipSpace = (ids) => chipSpaceA(ana, ids);
    // the MoE column's wire gaps, measured on the parallel MoE analysis —
    // the dense column advances by these to stay row-aligned across the flip
    const gapM = (ids) => Math.max(22, chipSpaceA(anaM, ids) + 10);
    const wireOut = (ids, sx, y, ov) => {
      tensorChip(ids, sx + 14, y + 4, ov);
      const gap = Math.max(22, chipSpace(ids) + 10);
      wire(sx, y, y + gap);
      return y + gap;
    };
    // aux backward artifact, exiting the box to the right
    const auxOut = (id, x, yMid) => {
      if (PONLY) return;
      const n = ana.byId[id];
      if (!n.aux) return;
      const replayed = ana.replayed.has(id); // a replay regenerates its aux
      // attention sits inside the MLA group: its lse label starts past the
      // group border so the border never cuts through the text
      const xt = (id === 'attn' && MLAGW) ? Math.max(x + W + 14, C1 - 10 + MLAGW + 8) : x + W + 14;
      P.push(`<line class="wire" x1="${x + W}" y1="${yMid}" x2="${xt - 4}" y2="${yMid}" marker-end="url(#arr)"/>` +
        (replayed
          ? `<text class="tensor tredo" x="${xt}" y="${yMid + 3}">↻ ${esc(n.aux.name)}</text>`
          : !this._ctl.quant
            ? `<text class="tensor tsave" x="${xt}" y="${yMid + 3}">← ${esc(n.aux.name)}</text>`
            : `<text class="tensor tsave" x="${xt}" y="${yMid + 3}">← ${esc(n.aux.name)} · ${fmtMem(n.aux.bytes)} ` +
              `<tspan fill="${DT_STYLE.fp32}">fp32</tspan></text>`));
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
    const mmBox = (ids, x, y, markIds, label, dims) => {
      const spec = MATMULS.find(m => m.id === ids[0]);
      const extra = stripExtra(sqParam(ids[0]), clsOf(ids[0]));
      P.push(`<g data-op="${ids[0]}"${boxTip((markIds ?? ids)[0], dims ? undefined : spec.dimsNote, ids[0])}>` +
        `<rect class="box" x="${x}" y="${y}" width="${W}" height="${BH + extra}" rx="4"/>` +
        (PBYTES && !this._ctl.dtype && exactParam(ids[0]) != null ? `<text class="dims" x="${x + W - 8}" y="${y + 13}" text-anchor="end">bf16</text>` : '') +
        `<text class="name" x="${x + 8}" y="${y + 13}">${label ?? spec.label}</text>` +
        `<text class="dims" x="${x + 8}" y="${y + 26}">${PONLY ? pstr(ids[0]).trim() : flatten(dims ?? spec.dims) + pstr(ids[0])}</text></g>`);
      P.push(modeBtn(markIds ?? ids, x + W - 86, y + 6));
      P.push(dtBtn(ids[0], x + W - 58, y + 6));
      auxOut((markIds ?? ids)[0], x, y + 19);
      flopBlocks(x + 8, y + 30, ana.byId[(markIds ?? ids)[0]]?.flopsTok, dt(ids[0]));
      paramBlocks(x + 8, y + 30, sqParam(ids[0]), clsOf(ids[0]));
      return y + BH + stripExtra(sqParam(ids[0]), clsOf(ids[0]));
    };
    const opNode = (id, label, x, y, cls = 'op', pc = '') => {
      const h2 = cls === 'comm' || !BQ ? 22 : 27;   // the extra 5px hold the fig-leaf strip
      P.push(`<g data-op="${id}"${boxTip(id)}>` +
        `<rect class="${cls}" x="${x}" y="${y}" width="${W}" height="${h2}" rx="6"/>` +
        `<text class="oplabel" x="${x + 10}" y="${y + 15}">${label}${pc ? `<tspan class="dims"> ${pc}</tspan>` : ''}</text></g>` +
        modeBtn([id], x + W - 30, y + 1));
      auxOut(id, x, y + Math.round(h2 / 2));
      if (cls !== 'comm') { flopBlocks(x + 10, y + 19, ana.byId[id]?.flopsTok, 'vector'); paramBlocks(x + 10, y + 19, sqParam(id)); }
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
      const RX = C1 + 150 + 22;
      // the down-projection is two separate GEMMs in production stacks
      // (wq_a | wkv_a in every production stack), so it is split at every tier:
      // fork norm1-out first
      P.push(`<circle cx="${SX1}" cy="${y - 10}" r="2.5" fill="#898781"/>` +
        `<path class="wire" d="M ${SX1} ${y - 10} L ${RX} ${y - 10} L ${RX} ${y}" marker-end="url(#arr)"/>`);
      const qFrac = DSV3.qRank / (DSV3.qRank + DSV3.kvRank + DSV3.qkRope);
      const HALF_ROW = 21;   // 140px half boxes: strip rows wrap inside the box
      const dhalf = (x, name, dims, tip, frac, withBtns, pc = '', nP = 0) => {
        const ex = stripExtra(nP, 'd', HALF_ROW);   // strip rows can outgrow the box (×N)
        P.push(`<g data-op="qkv_down" data-tip="${escAttr(tip)}">` +
          `<rect class="box" x="${x}" y="${y}" width="140" height="${HBH + ex}" rx="4"/>` +
          (PBYTES && !this._ctl.dtype ? `<text class="dims" x="${x + 134}" y="${y + 13}" text-anchor="end">bf16</text>` : '') +
          `<text class="name" x="${x + 6}" y="${y + 13}">${name}</text>` +
          `<text class="dims" x="${x + 6}" y="${y + 25}">${PONLY ? pc.trim() : flatten(dims) + pc}</text></g>` +
          (withBtns ? modeBtn(['qkv_down'], x + 140 - 86, y + 29) + dtBtn('qkv_down', x + 140 - 58, y + 29) : ''));
        flopBlocks(x + 6, y + 52, ana.byId.qkv_down.flopsTok * frac, dt('qkv_down'));
        paramBlocks(x + 6, y + 52, nP, 'd', HALF_ROW);
        return ex;
      };
      const pQ = pk(DSV3.hidden * DSV3.qRank), pKV = pk(DSV3.hidden * (DSV3.kvRank + DSV3.qkRope));
      const exQ = dhalf(C1, 'q down-proj', '7168 → 1536',
        `2 · 7168 · 1536 FLOP/token — wq_a; a separate GEMM from kv down-proj in production stacks\nparameters: ${(DSV3.hidden * DSV3.qRank).toLocaleString('en-US')}`, qFrac, true, pQ, DSV3.hidden * DSV3.qRank);
      const exKV = dhalf(C1 + 150, 'kv down-proj', '7168 → 512 + 64',
        `2 · 7168 · (512 + 64) FLOP/token — wkv_a; shares q down-proj’s mark and dtype (one graph node)\nparameters: ${(DSV3.hidden * (DSV3.kvRank + DSV3.qkRope)).toLocaleString('en-US')}`, 1 - qFrac, false, pKV, DSV3.hidden * (DSV3.kvRank + DSV3.qkRope));
      y += HBH + Math.max(exQ, exKV);   // the pair advances together
      // display-split of the one latents stash. What backward keeps is the
      // POST-norm latent (the up-proj's input), so in detail the chips sit
      // below the RMSNorm row. The kv down-proj box has TWO outputs: k_rope
      // (64) leaves from its bottom-right corner immediately and rides an
      // outer rail (clear of chip text) down to the kv-side RoPE.
      const latTot = DSV3.qRank + DSV3.kvRank;   // the k_rope dims are never stashed
      bypX = C1 + 358;                           // k_rope rail, clear of all chip text
      let bypTop = 0;
      if (DET) {
        const kx = C1 + 272;
        bypTop = y + 14;
        P.push(`<path class="wire" d="M ${kx} ${y} L ${kx} ${bypTop} L ${bypX} ${bypTop}"/>`);
        if (!PONLY) P.push(`<text class="tensor tidle" x="${kx + 6}" y="${bypTop - 4}">· k_rope · ${DSV3.qkRope}</text>`);
        else if (CONS) P.push(`<text class="tensor tidle" x="${kx + 6}" y="${bypTop - 4}">· k_rope</text>`);   // named, idle: never saved (RoPE bwd is a rotation)
        // pre-norm latent chips: real graph state (saved at no-AC — the latent
        // norms' backward input; the replay anchor under recompute presets)
        tensorChip(['qkv_down'], SX1 + 14, y + 24,
          { name: 'q latent', tdims: String(DSV3.qRank), frac: DSV3.qRank / latTot, short: true });
        tensorChip(['qkv_down'], RX + 14, y + 24,
          { name: 'kv latent', tdims: String(DSV3.kvRank), frac: DSV3.kvRank / latTot, short: true });
        wire(SX1, y, y + 48);
        P.push(`<path class="wire" d="M ${RX} ${y} L ${RX} ${y + 48}" marker-end="url(#arr)"/>`);
        y += 48;
        // MLA-internal RMSNorms (q_a_layernorm; kv_a_layernorm norms the 512 only)
        const normTip = 'input-form backward: reads its INPUT (pre-norm) + rstd — never its output. ' +
          'The pre-norm latent is not stashed; it is exactly recoverable from the post-norm stash, ' +
          '\u03b3, and rstd (x = y / (\u03b3\u00b7rstd)), which is why one latent copy suffices.';
        micro('RMSNorm', C1, y, 140, normTip, pk(DSV3.qRank).trim(), 'q_norm');
        micro('RMSNorm', C1 + 150, y, 140, normTip, pk(DSV3.kvRank).trim(), 'kv_norm');
        y += 18;
        // their rstd: exits the bottom, elbows right (\u2191 = read by the op
        // above, the norm's own backward); a replayed norm regenerates it
        if (!PONLY) for (const [nid, bx] of [['q_norm', C1], ['kv_norm', C1 + 150]]) {
          const rep = ana.replayed.has(nid);
          P.push(`<path class="wire" d="M ${bx + 112} ${y} L ${bx + 112} ${y + 7} L ${bx + 124} ${y + 7}" marker-end="url(#arr)"/>` +
            `<text class="tensor ${rep ? 'tredo' : 'tsave'}" x="${bx + 128}" y="${y + 10}">${rep ? '\u21bb' : '\u2191'} rstd</text>`);
        }
        y += 14;
      }
      if (DET) {
        // the normed latents are their own graph nodes (q_norm / kv_norm)
        tensorChip(['q_norm'], SX1 + 14, y + 4, { short: true });
        tensorChip(['kv_norm'], RX + 14, y + 4, { short: true });
      } else {
        tensorChip(['qkv_down'], SX1 + 14, y + 4,
          { name: 'q latent', tdims: String(DSV3.qRank), frac: DSV3.qRank / latTot, short: true });
        tensorChip(['qkv_down'], RX + 14, y + 4,
          { name: 'kv latent', tdims: String(DSV3.kvRank), frac: DSV3.kvRank / latTot, short: true });
      }
      // consolidated: the narrow fork chips are two lines tall (name / bytes)
      const latGap = Math.max(CONS ? 36 : 26, chipSpace(['qkv_down']) + 8);
      const wireTop = DET ? y - 14 : y;          // span the rstd band too — no spine gap
      wire(SX1, wireTop, y + latGap);
      P.push(`<path class="wire" d="M ${RX} ${wireTop} L ${RX} ${y + latGap}" marker-end="url(#arr)"/>`);
      y += latGap;
      const halfBox = (id, x) => {
        const m = MATMULS.find(mm2 => mm2.id === id);
        const ex = stripExtra(sqParam(id), 'd', 21);   // strip rows can outgrow the box (×N; 140px box row)
        P.push(`<g data-op="${id}"${boxTip(id, m.dimsNote)}>` +
          `<rect class="box" x="${x}" y="${y}" width="140" height="${HBH + ex}" rx="4"/>` +
          (PBYTES && !this._ctl.dtype ? `<text class="dims" x="${x + 134}" y="${y + 13}" text-anchor="end">bf16</text>` : '') +
          `<text class="name" x="${x + 6}" y="${y + 13}">${m.label}</text>` +
          `<text class="dims" x="${x + 6}" y="${y + 25}">${PONLY ? pstr(id).trim() : flatten(m.dims) + pstr(id)}</text></g>` +
          modeBtn([id], x + 140 - 86, y + 29) + dtBtn(id, x + 140 - 58, y + 29));
        flopBlocks(x + 6, y + 52, ana.byId[id]?.flopsTok, dt(id));
        paramBlocks(x + 6, y + 52, sqParam(id), 'd', 21);
        return ex;
      };
      y += HBH + Math.max(halfBox('q_up', C1), halfBox('kv_up', C1 + 150));   // the pair advances together
      if (DET) {
        // the up-proj outputs get names before RoPE; then two separate RoPE
        // kernels (Megatron: apply_mla_rope_for_q / _for_kv) — the kv one is
        // rope plus a little extra (split, broadcast, assemble K and V) — feed
        // q and k,v directly into attention. The k_rope rail lands here.
        if (!PONLY) P.push(`<text class="tensor tidle" x="${SX1 + 14}" y="${y + 12}">q_heads · ${flatten('128×192')}</text>` +
          `<text class="tensor tidle" x="${RX + 14}" y="${y + 12}">kv_heads · ${flatten('128×(128+128)')}</text>`);
        wire(SX1, y, y + 16);
        P.push(`<path class="wire" d="M ${RX} ${y} L ${RX} ${y + 16}" marker-end="url(#arr)"/>`);
        y += 16;
        micro('RoPE', C1, y, 140,
          'fused_apply_mla_rope_for_q — rotate the 64 rope dims of every q head (fp32), make Q contiguous');
        micro('RoPE + build K,V', C1 + 150, y, 140,
          'fused_apply_mla_rope_for_kv — split kv_heads into k_nope and V, rotate k_rope, broadcast it across the 128 heads, concat K = [k_nope | k_rope], make K and V contiguous');
        P.push(`<path class="wire" d="M ${bypX} ${bypTop} L ${bypX} ${y + 9} L ${C1 + W + 1} ${y + 9}" marker-end="url(#arr)"/>`);
        y += 18;
      }
      tensorChip(['q_up'], SX1 + 14, y + 4);
      tensorChip(['kv_up'], RX + 14, y + 4);
      const gap = Math.max(24, Math.max(chipSpace(['q_up']), chipSpace(['kv_up'])) + 12);
      wire(SX1, y, y + gap);
      P.push(`<path class="wire" d="M ${RX} ${y} L ${RX} ${y + gap - 14} L ${SX1 + 3} ${y + gap - 14}"/>` +
        `<circle cx="${SX1}" cy="${y + gap - 14}" r="2.5" fill="#898781"/>`);
      y += gap;
    }
    MLAGW = DET ? bypX - C1 + 22 : 336;   // before the attn row: its lse label clears this edge
    y = mmBox(['attn'], C1, y);
    y = wireOut(['attn'], SX1, y);
    y = mmBox(['o_proj'], C1, y);
    grp(C1, g1, y + 5, CUM ? `MLA · ${fmtPV(PARAMS.mla * KMUL)}${facTxt('d')}`
      : `MLA ×${DSV3.layers} · ${fmtPV(PARAMS.mla)}${facTxt('d')}`, MLAGW);
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
            ? `<path d="${shape} Z" fill="#fcfcfb" stroke="none"/>` +
              `<path d="${shape}" fill="none" stroke="#c3c2b7"/>`
            : `<path d="${shape} Z" fill="#eeede7" stroke="#d8d6cb"/>`) +
          `<text x="${x + 10}" y="${y0 + 17}" style="font:600 11px system-ui" fill="${on ? '#0b0b0b' : '#898781'}">${label}` +
          `<tspan style="font:10px system-ui" fill="${on ? '#898781' : '#a8a69e'}"> ${sub}</tspan></text></g>`;
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
        + (TABS ? 36 : 0);
      if (TABS) { drawTabs(z + g0 - 46); encTop = z + g0 - 20; }
      z += g0 + BH + (DET ? 18 : 0) + gapM(['router']) + 22 + gapM(['dispatch']);
      const gTop = z + 3; z += 21;
      wire(SX2, spineFrom, gTop - 3);   // arrow stops above the group, like the MoE rows
      z = mmBox(['ffn_gate_up'], C2, z, ['gate_up'], 'ffn gate/up', `7168 → 2×${DSV3.denseInter}`);
      z = wireOut(['gate_up'], SX2, z);
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
      }
    } else {
    let shBot = 0, shTop = 0;
    const SHX = C2 + 320, shMid = SHX + 22;        // shared-expert mini column; spine down its LEFT, like every column
    const shBox = (name, dims, tip, yy, pc = '') => {
      const n = name.includes('gate/up') ? 2 * DSV3.hidden * DSV3.moeInter : DSV3.hidden * DSV3.moeInter;
      P.push(`<g data-op="shared" data-tip="${escAttr(`${tip}\nparameters: ${n.toLocaleString('en-US')}`)}">` +
      `<rect class="box" x="${SHX}" y="${yy}" width="140" height="34" rx="4"/>` +
      `<text class="name" x="${SHX + 6}" y="${yy + 14}">${name}</text>` +
      `<text class="dims" x="${SHX + 6}" y="${yy + 27}">${PONLY ? pc.trim() : flatten(dims) + pc}</text></g>`);
    };
    if (!DET) {
      const g0 = Math.max(22, chipSpace(['norm2']) + 10) + (TABS ? 36 : 0);
      tensorChip(['norm2'], SX2 + 14, z + 4);
      wire(SX2, z, z + g0);
      if (TABS) { drawTabs(z + g0 - 46); encTop = z + g0 - 20; }
      z += g0;
    } else {
      // the shared expert runs on EVERY token as its own plain GEMMs — fork
      // norm2-out here; its boxes are drawn row-aligned with the routed ones
      tensorChip(['norm2'], SX2 + 14, z + 4);
      const nGap = Math.max(38, chipSpace(['norm2']) + 20) + (TABS ? 36 : 0);
      wire(SX2, z, z + nGap);
      if (TABS) { drawTabs(z + nGap - 46); encTop = z + nGap - 20; }
      shTop = z + nGap - 10;
      P.push(`<circle cx="${SX2}" cy="${shTop}" r="2.5" fill="#898781"/>` +
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
      z = micro('sigmoid · +bias · group top-k · scale', C2, z, W,
        `the learned e_score_correction_bias affects expert selection but not the gating weights\nparameters: ${PARAMS.routerBias.toLocaleString('en-US')}`,
        pk(PARAMS.routerBias).trim(), 'router_bias');
      P.push(`<path class="wire" d="M ${C2 + W} ${gateTop} L ${gateX} ${gateTop}"/>` +
        (PONLY ? '' : `<text class="tensor tidle" x="${C2 + 198}" y="${z + 11}">top-k weights · 8</text>`));
    }
    z = wireOut(['router'], SX2, z, DET ? { name: 'router state' } : undefined);
    const dispTop = z;
    z = opNode('dispatch', DET ? 'a2a dispatch (permute + comm) → EP group' : 'a2a dispatch → EP group', C2, z, 'comm');
    // the top-k weights are dispatched too: the rail enters the a2a alongside
    // the tokens and re-emerges as the per-expert weights (Megatron: probs in,
    // expert_probs out of hybridep_dispatch)
    if (DET) P.push(
      `<path class="wire" d="M ${gateX} ${gateTop} L ${gateX} ${dispTop + 7} L ${C2 + W + 1} ${dispTop + 7}" marker-end="url(#arr)"/>` +
      `<path class="wire" d="M ${C2 + W} ${dispTop + 16} L ${gateX} ${dispTop + 16}"/>`);
    z = wireOut(['dispatch'], SX2, z);
    const g2 = z + 3; z += 21;
    const rowG = z;
    z = mmBox(['ffn_gate_up'], C2, z, ['gate_up'], DET ? 'ffn gate/up (grouped ×8)' : undefined);
    if (DET) {
      P.push(`<path class="wire" d="M ${shMid} ${shTop} L ${shMid} ${rowG - 18}" marker-end="url(#arr)"/>` +
        `<text class="grplabel" x="${SHX}" y="${rowG - 6}">shared expert (every token)</text>`);
      shBox('shared gate/up', '7168 → 2×2048',
        'one plain GEMM per token — follows the ffn gate/up mark and dtype (its FLOPs are counted in the grouped strip)', rowG,
        pk(DSV3.hidden * 2 * DSV3.moeInter));
      tensorChip(['gate_up'], shMid + 14, z + 4, { name: 'gate, up (sh)', tdims: '2×2048', frac: 1 / nExp });
    }
    z = wireOut(['gate_up'], SX2, z, DET ? { name: 'gate, up (routed)', tdims: `${DSV3.topk}×2×2048`, frac: DSV3.topk / nExp } : undefined);
    if (DET) wire(shMid, rowG + 34, z);
    // gate-at-swiglu, not gate-at-combine: by linearity the router weights can
    // multiply the swiglu output before the down-proj (one fused kernel,
    // a fused swiglu-and-scale kernel) — this is what makes the expert outputs a pure
    // intermediate instead of a stash for the combine's backward
    if (DET) P.push(`<path class="wire" d="M ${gateX} ${dispTop + 16} L ${gateX} ${z + 13} L ${C2 + W + 1} ${z + 13}" marker-end="url(#arr)"/>`);
    const rowS = z;
    z = opNode('swiglu', DET ? 'SwiGLU · × top-k weight (one fused kernel)' : 'SwiGLU', C2, z);
    if (DET) {
      micro('SwiGLU (ungated)', SHX, rowS, 140);
      tensorChip(['swiglu'], shMid + 14, z + 4, { name: 'swiglu out (sh)', tdims: '2048', frac: 1 / nExp });
    }
    z = wireOut(['swiglu'], SX2, z, DET ? { name: 'swiglu out (routed)', tdims: `${DSV3.topk}×2048`, frac: DSV3.topk / nExp } : undefined);
    if (DET) wire(shMid, rowS + 18, z);
    const rowD = z;
    z = mmBox(['ffn_down'], C2, z, undefined, DET ? 'ffn down (grouped ×8)' : undefined);
    if (DET) {
      shBox('shared down', '2048 → 7168',
        'one plain GEMM per token — follows the ffn down mark and dtype; its output joins the routed sum', rowD,
        pk(DSV3.moeInter * DSV3.hidden));
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
        'routed + shared expert outputs — Megatron fuses this with the residual add (add_shared_and_residual); split here for clarity', 178);
      encBot = zA + 14;   // the routed+shared add is MoE-internal — inside the box
      if (ONLY === 'ffn') {
        z = zA;
      } else {
      const zB = Math.max(zA + 34, col1End - 4);
      wire(SX2, zA + 9, zB - 11);
      plus(SX2, zB);
      addBox(SX2 + 26, zB, 'residual add', 'residual add — x1 + the ffn output');
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
      P.push(`<circle cx="${midX}" cy="${z}" r="2.5" fill="#898781"/>` +
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
        // active tab footprint: the outline leaves a gap there instead of an
        // opaque eraser (which bleeds the border through when the tab fades)
        const [tx, tw] = this.kind === 'dense' ? [C2 + 42, 148] : [C2 + 198, 168];
        P[P.indexOf('__ENC__')] = HIDT >= 1 ? '' :
          `<g opacity="${(1 - HIDT).toFixed(3)}">` +
          `<rect x="${x0}" y="${y0}" width="${x1 - x0}" height="${y1 - y0}" rx="${r}" fill="#fcfcfb" stroke="none"/>` +
          `<path d="M ${tx + tw} ${y0} L ${x1 - r} ${y0} Q ${x1} ${y0} ${x1} ${y0 + r} L ${x1} ${y1 - r} ` +
          `Q ${x1} ${y1} ${x1 - r} ${y1} L ${x0 + r} ${y1} Q ${x0} ${y1} ${x0} ${y1 - r} ` +
          `L ${x0} ${y0 + r} Q ${x0} ${y0} ${x0 + r} ${y0} L ${tx} ${y0}" fill="none" stroke="#c3c2b7"/></g>`;
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
      if (CONS) this._segTotals.push(ana.savedBytes * 4096 * m2);
    }
    if (LOCAL) {   // the fit bar renders in its own row under the controls (this._barHtml)
      const cap = 80 * 2 ** 30;
      const moeExp = PARAMS.expert * DSV3.routedExperts;
      const stageParts = (S) => {   // this rank's params by class: experts / non-expert blocks / emb+lm head (+final norm)
        const g = ppStage(Math.min(S.stage, S.pp - 1), S.pp, S.vpp, S.fold);
        return {
          e: g.moe * moeExp / S.ep,
          d: g.dense * PARAMS.denseBlock + g.moe * (PARAMS.moeBlock - moeExp),
          v: ((g.emb ? 1 : 0) + (g.head ? 1 : 0)) * PARAMS.embed
            + (g.head ? DSV3.hidden : 0),
          layers: g.layers,
        };
      };
      const shardOf = (S, c, cls) =>
        (S.zero ?? 1) >= c.zthresh ? c.bpp / (cls === 'e' ? (S.world ?? LOCAL_PAR.world) / S.pp / S.ep : (S.world ?? LOCAL_PAR.world) / S.pp) : c.bpp;
      // per component: [experts, non-expert blocks, emb+lm head] bytes — the solo
      // breakdown and its pin factors ride these
      const partsFor = (S) => {
        const q = stageParts(S);
        const actM = 4096 * q.layers * inflightOf(S.sched ?? '1f1b', S.stage, S.pp, S.vpp, S.fold);
        const actP = actBucketsOf(ana).map((b) => b * actM);
        return COMPS.map((c) => [q.e * shardOf(S, c, 'e'), q.d * shardOf(S, c, 'd'), q.v * shardOf(S, c, 'd')])
          .concat([actP]);
      };
      const segB = (S) => {
        const q = stageParts(S);
        return COMPS.map((c) => (q.d + q.v) * shardOf(S, c, 'd') + q.e * shardOf(S, c, 'e'))
          .concat([ana.savedBytes * 4096 * q.layers * inflightOf(S.sched ?? '1f1b', S.stage, S.pp, S.vpp, S.fold)]);
      };
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
      const colors = [...COMPS.map((c) => c.color), '#eda100'];
      const IF2 = inflightOf(SCHED, STG, PPn, VPPn, FOLD);
      const names = ['weights', 'gradients (fp32)', 'optimizer states', `activations ×${IF2}mb`];
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
      const partIdxsOf = (j) => (this._segParts[j] ?? [])
        .map((b, k) => b > 0 || (pin?.parts?.[j]?.[k] ?? 0) > 0
          || (ALLPARTS && !(j === 3 && k === ACT_BUCKETS.length - 1)) ? k : -1)
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
          prop: i < COMPS.length ? COMPS[i].prop : 'showActs', abs,
          segs: [
            { key: 'base', x0, x1: grey ? px(grey) : x0, color: '#c3c2b7', op: 1 },
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
          name: 'total', color: '#52514e', prop: null, abs: totalN,
          segs: [{ key: 'base', x0, x1: px(stacked ? totalN - topSum : totalN),
            color: stacked ? '#c3c2b7' : '#52514e', op: 1,
            bar: stacked ? null : 'total', true: stacked ? null : totalN }, ...tips],
          ghost: ghostOf(totalN, pinTotal, '#898781'),
          val: { x: px(totalN) + 5, op: 1, text: `${fmtBytes(totalN)}${badge(totalN, pinTotal)}`,
            true: totalN, pin: pinTotal || '' } });
      }
      const SHOWLBL = pin && !(SNAP2 && this.getAttribute('knobs'));
      const L1 = {
        // header: PP1 needs no locus (the prose owns the framing); a
        // pipelined chart names whose bytes these are — one GPU's stage
        hdr: PPn === 1 ? 'logarithmic:' : `one GPU, stage ${STG} of PP${PPn} (logarithmic):`,
        axisY, HB: axisY + 38, capPx: px(cap),   // +38: the distances legend band (shared with the save label)
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
      const lmRows = this._ctl.quant
        ? Math.ceil(Math.max(1, Math.round(flopEq(lmFlops, dt('lm_head')) / FLOP_UNIT)) / FLOP_ROW) : 0;
      lmH = (BQ ? 38 : 34) + lmRows * 6;   // lm-head text sits 2px lower than mmBox text
      P.push(`<g data-op="lm_head" data-tip="${escAttr(`${fmtNum(lmFlops)} FLOP/token = ${FLOP_EXPR.lm_head}\n${lm.dimsNote}\nparameters: ${PARAMS.embed.toLocaleString('en-US')}`)}">` +
        `<rect class="box" x="${C1 + 184}" y="${h}" width="240" height="${lmH}" rx="4"/>` +
        `<text class="name" x="${C1 + 192}" y="${h + 14}">${lm.label}</text>` +
        `<text class="dims" x="${C1 + 192}" y="${h + 28}">${PONLY ? pstr('lm_head').trim() : flatten(lm.dims) + pstr('lm_head')}</text></g>` + dtBtn('lm_head', C1 + 184 + 240 - 58, h + 7));
      flopBlocks(C1 + 192, h + 33, lmFlops, dt('lm_head'));
      P.push(`<line class="wire" x1="${C1 + 424}" y1="${h + 17}" x2="${C1 + 454}" y2="${h + 17}" marker-end="url(#arr)"/>`);
      P.push(`<rect class="op" x="${C1 + 458}" y="${h + 6}" width="140" height="22" rx="11"/>` +
        `<text class="oplabel" x="${C1 + 470}" y="${h + 21}">softmax / loss</text>`);
    }

    // ---- per-layer FLOP tally: fwd + bwd + recompute replay, same block scale.
    // Rendered as its own small SVG, floated to the right of the caption.
    const T = [];
    const eq = (n) => flopEq(n.flopsTok, opDt(n.id));
    const fwdOps = Object.values(ana.byId).filter(n => n.flopsTok > 0);
    const fwdEq = fwdOps.reduce((t, n) => t + eq(n), 0);
    const replayOps = fwdOps.filter(n => ana.replayed.has(n.id));
    const replayEq = replayOps.reduce((t, n) => t + eq(n), 0);
    let ty = 10;
    T.push(`<text class="grplabel" x="0" y="${ty}">per-layer FLOPs as time at peak (same block scale):</text>`);
    ty += 8;
    const DT_ORDER = { bf16: 0, mxfp8: 1, fp32: 2 };
    const tallyStrip = (label, list, mult, num) => {
      T.push(`<text class="dims" x="0" y="${ty + 9}">${label}</text>`);
      let i = 0;
      // group same-dtype ops so each color is one contiguous run
      for (const n of [...list].sort((p, q) => (DT_ORDER[opDt(p.id)] ?? 3) - (DT_ORDER[opDt(q.id)] ?? 3))) {
        const k = Math.round(eq(n) * mult / FLOP_UNIT);
        const color = DT_STYLE[opDt(n.id)] ?? '#c3c2b7';
        for (let j = 0; j < k; j++, i++)
          T.push(`<rect x="${44 + (i % FLOP_ROW) * 6}" y="${ty + Math.floor(i / FLOP_ROW) * 6}" width="5" height="4" fill="${color}"/>`);
      }
      T.push(`<text class="dims" x="${44 + FLOP_ROW * 6 + 12}" y="${ty + 9}">${num}</text>`);
      ty += Math.max(1, Math.ceil(i / FLOP_ROW)) * 6 + 6;
    };
    tallyStrip('fwd', fwdOps, 1, '1.00×');
    tallyStrip('bwd', fwdOps, 2, '2.00× (dgrad + wgrad)');
    tallyStrip('replay', replayOps, 1, `+${(replayEq / fwdEq).toFixed(2)}×`
      + (ana.replayComm.length ? ' + a2a ' + ana.replayComm.join('+') : ''));
    T.push(`<text class="dims" x="44" y="${ty + 8}">= ${(3 + replayEq / fwdEq).toFixed(2)}× fwd per training step</text>`);
    const tallyEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    tallyEl.setAttribute('width', 370); tallyEl.setAttribute('height', ty + 16);
    tallyEl.setAttribute('viewBox', `0 0 370 ${ty + 16}`);
    tallyEl.innerHTML = T.join('');
    this._tallySvg = tallyEl;

    const H = h + lmH + 14;

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
      `<path d="M 0 0 L 8 4 L 0 8 z" fill="#898781"/></marker></defs>` + P.join('');
    for (const b of svgEl.querySelectorAll('button[data-dt]')) {
      b.onclick = () => {
        const mutate = () => {
          const cycle = { bf16: 'mxfp8', mxfp8: 'fp32', fp32: 'bf16' };
          this.matmuls[b.dataset.dt] = cycle[this.matmuls[b.dataset.dt]] ?? 'bf16';
        };
        if (this.hasAttribute('local') && this.getAttribute('lens') === 'param-bytes') {
          const prev = this._snapLocal(); mutate(); this._tweenLocal(prev);
        } else { mutate(); this.render(); this.changed(); }
      };
    }
    for (const b of svgEl.querySelectorAll('button[data-mark]')) {
      b.onclick = () => this.toggleMark(b.dataset.mark.split(','));
    }
    for (const b of svgEl.querySelectorAll('[data-kind]')) {
      b.onclick = () => {
        if (this.kind === b.dataset.kind) return;
        this.kind = b.dataset.kind; this.render(); this.changed(true);
      };
    }
    return svgEl;
  }
  // instant tooltips; click a tipped element (not a button) to pin
  attachTip(root) {
    const tip = el('div', 'lv-tip');
    root.append(tip);
    let pinned = false;
    const place = (ev) => {
      const r = root.getBoundingClientRect();
      tip.style.left = Math.min(ev.clientX - r.left + 14, r.width - 280) + 'px';
      tip.style.top = (ev.clientY - r.top + 14) + 'px';
    };
    root.addEventListener('mousemove', (ev) => {
      if (pinned) return;
      const t = ev.target.closest?.('[data-tip]');
      if (t) { tip.textContent = t.dataset.tip; tip.style.display = 'block'; place(ev); }
      else tip.style.display = 'none';
    });
    root.addEventListener('click', (ev) => {
      if (pinned) { pinned = false; tip.classList.remove('pinned'); tip.style.display = 'none'; return; }
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
.pps { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: #0b0b0b;
  border: 1px solid #e1e0d9; border-radius: 6px; background: #fcfcfb; padding: 8px 10px; }
.pps .top { display: flex; align-items: flex-start; gap: 12px; padding-bottom: 8px; flex-wrap: wrap; }
${knobCss('.pps .top')}
.pps .hd { color: #52514e; font-size: 11.5px; align-self: center; }
.pps .stghit { cursor: pointer; }
.pps g.lane { cursor: pointer; }
.pps g.lane.pin rect[data-stash] { stroke-width: 1.6; }
.pps .stghit:hover { opacity: 0.6; }
.pps .scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; overscroll-behavior-x: none; }
.pps svg { display: block; }
`;
class Dsv3PpSchedule extends HTMLElement {
  connectedCallback() {
    this._sig = '';
    this._m = this.getAttribute('mb') === 'auto' ? 'auto' : +(this.getAttribute('mb') ?? 64);
    const style = document.createElement('style'); style.textContent = PPS_CSS;
    this._root = el('div', 'pps');
    this._top = el('div', 'top');
    this._ctl = el('span');            // the replicated pipeline knob group
    this._hd = el('div', 'hd');
    this._top.append(this._ctl, this._hd);
    this._scr = el('div', 'scroll');
    // the sX axis IS the stage picker: click a row's gutter to select it;
    // once clicked the strip holds focus, so ↑/↓ walk the stages
    this._scr.tabIndex = -1;
    this._scr.style.outline = 'none';
    this._scr.addEventListener('click', (e) => {
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
      if (!t || !this._layer) return;
      this._scr.focus({ preventScroll: true });
      const l = this._layer, v = +t.dataset.stage;
      if (l.stage !== v) l.setLocal(() => { l.stage = v; });
    });
    this._scr.addEventListener('keydown', (e) => {
      const d = e.key === 'ArrowDown' ? 1 : e.key === 'ArrowUp' ? -1 : 0;
      const l = this._layer;
      if (!d || !l) return;
      e.preventDefault();
      const v = Math.min(l.pp - 1, Math.max(0, l.stage + d));
      if (l.stage !== v) l.setLocal(() => { l.stage = v; });
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
  controls(pp, sched, stage, vpp, fold) {
    const l = this._layer;
    this._ctl.innerHTML = '';
    if (!l) return;
    const g = el('span', 'pargrp');
    const lab = el('div', 'parlab'); lab.textContent = 'pipeline'; g.append(lab);
    const row = (...kids) => { const d = el('div', 'parrow'); d.append(...kids); return d; };
    const txt = (t) => { const sp = el('span'); sp.style.cssText = 'color:#52514e;font-size:11px;'; sp.textContent = t; return sp; };
    const mkstp = (opts, cur, set, max = Infinity) => {
      const w2 = el('span', 'stp');
      const val = document.createElement('select'); val.className = 'v';
      for (const o of opts.filter((o) => o <= max)) val.append(new Option(o, o));
      val.value = String(cur);
      val.onchange = () => set(+val.value);
      const i = opts.indexOf(cur);
      const btn = (t, j) => {
        const b = document.createElement('button');
        b.textContent = t; b.type = 'button';
        b.disabled = j < 0 || j >= opts.length || opts[j] > max;
        if (!b.disabled) b.onclick = () => set(opts[j]);
        return b;
      };
      w2.append(btn('−', i - 1), val, btn('+', i + 1));
      return w2;
    };
    const stp = mkstp([1, 2, 4, 8, 16, 32, 64], pp, (v) => l.setLocal(() => l._setPP(v)));
    const seg = (opts, cur, set) => {
      const w2 = el('span', 'stp');
      for (const [k, t] of opts) {
        const b = document.createElement('button');
        b.textContent = t; b.type = 'button';
        if (cur === k) b.classList.add('on');
        else b.onclick = () => set(k);
        w2.append(b);
      }
      return w2;
    };
    const world = l.world ?? LOCAL_PAR.world;
    const sw = seg([['1f1b', '1F1B'], ['one', '×1 mb']], sched, (k) => l.setLocal(() => { l.sched = k; }));
    const vw = mkstp([1, 2, 4], vpp, (v) => l.setLocal(() => {
      l.vpp = v; l.stage = peakStage(l.pp, l.ep, l.zero ?? 1, world, l.sched, v, l.fold);
    }));
    const fw = seg([['wrap', 'wrap'], ['reflect', 'V']], fold, (k) => l.setLocal(() => {
      l.fold = k; l.stage = peakStage(l.pp, l.ep, l.zero ?? 1, world, l.sched, l.vpp ?? 1, k);
    }));
    // microbatches DRAWN — a strip-local knob (the memory model needs no m;
    // its 1F1B law assumes m ≥ pp): a real step's worth by default, 'auto'
    // = just enough to reach steady state (depth + 4)
    const msel = document.createElement('select'); msel.className = 'v';
    msel.style.borderLeft = msel.style.borderRight = '1px solid #c3c2b7';
    msel.style.borderRadius = '4px';
    for (const o of ['auto', 4, 8, 16, 32, 64, 128]) msel.append(new Option(o, o));
    msel.value = String(this._m);
    msel.onchange = () => { this._m = msel.value === 'auto' ? 'auto' : +msel.value; this._sig = ''; this.sync(); };
    for (const [n, e2] of [['pp', stp], ['sched', sw], ['vpp', vw], ['fold', fw], ['mb', msel]])
      e2.dataset.knob = n;
    g.append(row(txt('PP'), stp),
      row(txt('sched'), sw, txt('VPP'), vw, fw, txt('mb'), msel));
    this._ctl.append(g);
  }
  cfg() {
    const l = this._layer;
    return {
      pp: l?.pp ?? +(this.getAttribute('pp') ?? LOCAL_PAR.pp),
      sched: l?.sched ?? (this.getAttribute('sched') ?? '1f1b'),
      stage: l?.stage ?? +(this.getAttribute('stage') ?? 0),
      vpp: l?.vpp ?? +(this.getAttribute('vpp') ?? 1),
      fold: l?.fold ?? (this.getAttribute('fold') ?? 'reflect'),
    };
  }
  sync() {
    const { pp, sched, stage, vpp, fold } = this.cfg();
    const sig = `${pp}|${sched}|${stage}|${vpp}|${fold}|${this._m}`;
    if (sig === this._sig) return;   // the layer's tween fires 'recipe' every frame
    const grew = this._sig.split('|').slice(0, 2).join('|') !== `${pp}|${sched}`;
    const prevH = this._sig && grew ? this._scr.getBoundingClientRect().height : 0;
    this._sig = sig;
    this.controls(pp, sched, stage, vpp, fold);
    this.draw(pp, sched, stage, vpp, fold);
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
    // resolve timing: rank-sequential, F waits on F@v−1, B on B@v+1 (its own
    // F at the deepest stage), W only on rank order; fused blocks wait on both
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
            rec = [{ s: r, v: o.v, mb: o.mb, ph: o.ph, t0, t1, chunk: o.v >= pp ? 1 : 0 }];
            done.set(`${o.ph}${o.mb}@${o.v}`, t1);
            rankT[r] = t1;
          } else if (o.ph === 'W') {
            const t0 = rankT[r];
            rec = [{ s: r, v: o.v, mb: o.mb, ph: 'W', t0, t1: t0 + 1, chunk: o.v >= pp ? 1 : 0 }];
            rankT[r] = t0 + 1;
          } else {   // FB: fused forward+backward, 3 slots
            const dF = depF(o.vF, o.mbF), dB = depB(o.vB, o.mbB);
            if (dF === undefined || dB === undefined) break;
            const t0 = Math.max(rankT[r], dF, dB);
            rec = [
              { s: r, v: o.vF, mb: o.mbF, ph: 'F', t0, t1: t0 + 1, fuse: 1, chunk: o.vF >= pp ? 1 : 0 },
              { s: r, v: o.vB, mb: o.mbB, ph: 'B', t0: t0 + 1, t1: t0 + 3, fuse: 2, chunk: o.vB >= pp ? 1 : 0 },
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
    return cells;
  }
  draw(pp, sched, stage, vpp = 1, fold = 'reflect') {
    this._pinHl = null;
    // one chain of VIRTUAL stages with 1F1B admission per stage, vpp·pp deep;
    // placement per vstagesOf (wrap = Megatron interleaving, reflect = the
    // V/DualPipeV zigzag). Each rank interleaves its chunk queues greedily,
    // one op at a time (the official DualPipeV also overlaps F+B blocks;
    // this doesn't — but the residency it draws IS the modeled law).
    // F_mb@v waits on F_mb@(v−1); B_mb@v on B_mb@(v+1), or its own forward
    // on the deepest stage. Durations: F = 1 slot, B = 2 (~2× the FLOPs).
    const D = vpp * pp;
    const m = sched === 'one' ? 1 : this._m === 'auto' ? D + 4 : this._m;
    const OFFICIAL = sched === '1f1b' && vpp === 2 && fold === 'reflect' && pp > 1 && m >= 2 * pp;
    const qs = OFFICIAL ? [] : Array.from({ length: D }, (_, v) => {
      const wu = Math.min(D - 1 - v, m); const items = [];
      for (let j = 0; j < wu; j++) items.push(['F', j]);
      for (let j = wu; j < m; j++) items.push(['F', j], ['B', j - wu]);
      for (let j = Math.max(m - wu, 0); j < m; j++) items.push(['B', j]);
      return { items, i: 0 };
    });
    const done = new Map(); const cells = OFFICIAL ? this._officialDPV(pp, m) : [];
    const rankT = Array(pp).fill(0);
    const stagesOf = Array.from({ length: pp }, (_, r) => vstagesOf(r, pp, vpp, fold));
    let progress = !OFFICIAL;
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
    const T = Math.max(...cells.map(c => c.t1));
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
    const schedH = pp * (RH + GAP) - GAP;
    const laneY0 = schedH + 10 + HDR;
    const H = laneY0 + laneEnd.length * (RH2 + GAP) - GAP + 4;
    const W = GUT + T * U + 1;
    const rowY = (s) => s * (RH + GAP);
    const laneY = (ln) => laneY0 + ln * (RH2 + GAP);
    const P = [`<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family="system-ui">`];
    if (pp > 1) P.push(`<rect class="stghl" x="0" y="${rowY(stage)}" width="${W}" height="${RH}" fill="#fff3d1"/>`);
    for (let s = 0; s < pp; s++) {
      P.push(`<text x="${GUT - 6}" y="${rowY(s) + RH - 4}" text-anchor="end" font-size="9.5"`
        + ` font-weight="${s === stage ? 600 : 400}" fill="${s === stage ? '#0b0b0b' : '#898781'}">s${s}</text>`);
      P.push(`<rect class="stghit" data-stage="${s}" x="0" y="${rowY(s)}" width="${GUT - 2}" height="${RH}" fill="#fff3d1" opacity="0"/>`);
    }
    // later chunks wear progressively deeper shades of the same hues
    // (VPP2 reflect: light down pass, dark up pass); W = deferred weight
    // grads, the pale dashed cells of the zero-bubble split
    const STY = {
      F: [['#fdeab5', '#eda100', '#7a5200'], ['#f6cd74', '#c98800', '#5c3d00'],
        ['#eab04a', '#a86e00', '#4a3000'], ['#d69432', '#875600', '#3a2500']],
      B: [['#fbd4c0', '#eb6834', '#7a2f12'], ['#f3ac8b', '#c74e1d', '#5c2410'],
        ['#e58a63', '#a63c12', '#471b09'], ['#d16b42', '#88300c', '#361406']],
      W: [['#fdefe8', '#eb6834', '#7a2f12'], ['#f9ded0', '#c74e1d', '#5c2410'],
        ['#f3c8b3', '#a63c12', '#471b09'], ['#ecb298', '#88300c', '#361406']],
    };
    for (const c of cells) {
      const [fill, stroke, ink] = STY[c.ph][c.chunk];
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
    P.push(`<text x="0" y="${laneY0 - 7}" font-size="10" fill="#52514e">in flight on s${Math.min(stage, pp - 1)}`
      + ` — each bar: the F that stashes a microbatch, held (amber) until the B that frees it.`
      + ` The peak is what the memory bars charge</text>`);
    for (const e of stash) {
      const [f, fs2, fi] = STY.F[e.chunk], [bf, bs, bi] = STY.B[e.chunk];
      const y = laneY(e.lane);
      P.push(`<g class="lane" data-mb="${e.mb}" data-v="${e.v}">`);
      // hitbox: the whole row band over the stash's span, not just the marks
      P.push(`<rect x="${GUT + e.f0 * U}" y="${y - GAP / 2}" width="${(e.b1 - e.f0) * U}" height="${RH2 + GAP}" fill="transparent"/>`);
      P.push(`<rect x="${GUT + e.f1 * U}" y="${y + RH2 / 2 - 2}" width="${(e.b0 - e.f1) * U}" height="4" fill="#fdeab5" data-stash-tail="1"/>`);
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
    P.push(`<path d="M ${bx - 4} ${by0} h 8 M ${bx} ${by0} V ${by1} M ${bx - 4} ${by1} h 8" stroke="#0b0b0b" stroke-width="1.2" fill="none" pointer-events="none"/>`);
    P.push(`<text data-peak="${IFm}" x="${bx + 7}" y="${(by0 + by1) / 2 + 3.5}" font-size="10" font-weight="600" fill="#0b0b0b" stroke="#fcfcfb" stroke-width="3" paint-order="stroke" pointer-events="none">${IFm} mb in flight (peak)${vpp > 1 ? ` = ${peakN} chunks` : ''}</text>`);
    P.push('</svg>');
    const ppTag = this._layer ? '' : `PP${pp} · `;   // the knob group already names PP
    const vppTag = OFFICIAL
      ? 'DualPipeV (official program) · down-pass chunk light, up-pass dark · F and B drawn touching = one'
        + ' overlapped F&B block · pale dashed W = deferred weight grads (B alone = input grads)'
      : vpp > 1
        ? `VPP${vpp} ${fold === 'wrap' ? 'wrap (Megatron interleaving)' : 'reflect'}`
          + ` · each rank runs ${vpp} chunks, later passes darker · chunks scheduled 1F1B`
          + (fold === 'reflect' && vpp === 2 ? ' (official DualPipeV needs ≥ 2·PP microbatches)' : '')
        : '';
    this._hd.textContent = sched === 'one'
      ? `${ppTag}one microbatch at a time — an F wave down the stages, then a B wave back up`
      : vpp > 1
        ? `${ppTag}${m} microbatches shown · ${vppTag}`
        : `${ppTag}1F1B · ${m} microbatches shown — F = forward (one slot), B = backward (two: ~2× the FLOPs)`;
    this._scr.innerHTML = P.join('');
  }
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
.deck-nav button { font: 12px ui-monospace, monospace; padding: 2px 12px; border: 1px solid #c3c2b7;
  border-radius: 4px; background: #fff; cursor: pointer; }
.deck-nav button:hover:not(:disabled) { background: #f3f2ee; }
.deck-nav button:disabled { color: #dedcd3; cursor: default; }
.deck-step { font: 11px ui-monospace, monospace; color: #52514e; }
.deck-hyp { font: italic 11px system-ui; color: #898781; }
.deck-mod { font: italic 11px system-ui; color: #b05f00; }
.deck-nav button.deck-rst { color: #b05f00; border-color: #b05f00; padding: 1px 8px; }
.deck-cap { max-width: 760px; font-size: 13.5px; color: #1c1c1a; line-height: 1.5; }
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
    const stg = c.stage ?? peakStage(c.pp, c.ep, c.zero ?? 1, c.world, c.sched, c.vpp, c.fold);
    return l.pp !== c.pp || l.ep !== c.ep || (l.zero ?? 0) !== c.zero || l.world !== c.world
      || (l.sched ?? '1f1b') !== c.sched || (l.vpp ?? 1) !== c.vpp
      || (l.fold ?? 'reflect') !== c.fold || l.stage !== stg;
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
