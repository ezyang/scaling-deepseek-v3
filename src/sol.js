// The stupidest step-time model: no timeline, no overlap, no communication.
// Every op of the 61-layer stack (+ head) is priced at ONE rate — the GPU's
// FP8 tensor peak (the ops that really run bf16/fp32, attention core and
// router, are a later negotiation) — forward + backward (2× forward) +
// whatever the recompute policy replays, and the pieces are SUMMED. Compute is conserved
// under any parallelism, so no parallelism knob exists here: the per-GPU
// step is the whole batch's compute divided by the cluster. <dsv3-sol> draws
// the sum as stacked rows (forward / backward / recompute / total) by op
// group, against the step the vendor-reported throughput implies.
import { DSV3, HARDWARE, HEAD_OPS } from './model.js';
import { blockGraph, analyze, RECOMPUTE_PRESETS } from './blockgraph.js';
import { C } from './theme.js';
import { knobCss } from './ui.js';

export const SOL_DEFAULTS = { hw: 'h800', recompute: 'dsv3', gpus: 2048, gbs: 15360, seq: 4096 };

// op groups: the stack's columns, in the order the bar stacks them
export const GROUPS = [
  { id: 'attn', label: 'attention core', c: '#c74e1d' },
  { id: 'mla', label: 'MLA projections', c: '#f3ac8b' },
  { id: 'routed', label: 'routed experts', c: '#2a78d6' },
  { id: 'shared', label: 'shared expert', c: '#bcd8f3' },
  { id: 'dense', label: 'dense FFN', c: '#1baf7a' },
  { id: 'router', label: 'router', c: '#8a3324' },
  { id: 'vector', label: 'norms · RoPE · SwiGLU', c: '#c3c2b7' },
  { id: 'head', label: 'lm head · loss', c: '#eda100' },
];
export const PASSES = [
  { id: 'fwd', label: 'forward' }, { id: 'bwd', label: 'backward' }, { id: 'replay', label: 'recompute' },
];
// vendor-reported throughput, tok/s/GPU, that the "reported step" line
// divides the batch by: H800 = implied by DeepSeek's 2.788M GPU-hours for
// 14.8T tokens. Nothing published for H100.
export const REPORTED = {
  h800: { rate: 14.8e12 / (2.788e6 * 3600), src: 'implied by DeepSeek’s GPU-hours' },
};
// the tech sheet: what the GPU knob shows (per GPU, per direction, data-sheet peaks)
export const SHEET = (hw) => [
  ['FP8', `${(hw.flops.fp8 / 1e12).toLocaleString('en-US')} TFLOP/s`],
  ['NVLink', `${hw.nvl / 1e9} GB/s`],
  ['InfiniBand', `${hw.nic / 1e9} GB/s`],
];

const GROUP_OF = { attn: 'attn', qkv_down: 'mla', q_up: 'mla', kv_up: 'mla', o_proj: 'mla', router: 'router' };
// every op is a matmul at the FP8 rate here; blockGraph wants a dtype table
// only to size stashes, which this model never reads
const ALL_FP8 = new Proxy({}, { get: () => 'e4m3' });

// FLOP/token per (pass, group)
export function solFlops(cfg) {
  cfg = { ...SOL_DEFAULTS, ...cfg };
  const a = DSV3;
  const marks = RECOMPUTE_PRESETS[cfg.recompute];
  const cells = {};   // `${pass}·${group}` → { flops }
  const add = (pass, group, flops) => {
    if (!flops) return;
    const k = pass + '·' + group;
    const c = cells[k] ??= { pass, group, flops: 0 };
    c.flops += flops;
  };
  for (const [kind, count] of [['dense', a.denseLayers], ['moe', a.layers - a.denseLayers]]) {
    const nodes = blockGraph(kind, a, ALL_FP8, cfg.seq);
    const ana = analyze(nodes, marks);
    for (const n of nodes) {
      if (!n.flopsTok) continue;
      const f = n.flopsTok * count;
      // the FFN GEMMs count all experts a token visits (top-k routed + shared) in one node
      const pieces = n.id === 'gate_up' || n.id === 'ffn_down'
        ? kind === 'dense' ? [['dense', 1]] : [['routed', a.topk / (a.topk + a.sharedExperts)], ['shared', a.sharedExperts / (a.topk + a.sharedExperts)]]
        : [[GROUP_OF[n.id] ?? 'vector', 1]];
      for (const [g, frac] of pieces) {
        add('fwd', g, f * frac);
        add('bwd', g, 2 * f * frac);
        if (ana.replayed.has(n.id)) add('replay', g, f * frac);
      }
    }
  }
  for (const op of HEAD_OPS(a)) {   // the loss is forward-only
    add('fwd', 'head', op.ftok);
    add('bwd', 'head', op.ftok * (op.bwdMult ?? 2));
  }
  return { cfg, cells: Object.values(cells) };
}

// seconds per step on one GPU: tokens per GPU × Σ FLOP/token ÷ the FP8 peak
export function solStep(cfg) {
  const { cfg: c, cells } = solFlops(cfg);
  const hw = HARDWARE[c.hw];
  const tokens = c.gbs * c.seq / c.gpus;
  const rows = {};
  let total = 0, flopsTok = 0;
  for (const cell of cells) {
    cell.s = tokens * cell.flops / hw.flops.fp8;
    rows[cell.pass] = (rows[cell.pass] ?? 0) + cell.s;
    total += cell.s; flopsTok += cell.flops;
  }
  const rep = REPORTED[c.hw];
  return { cfg: c, hw, tokens, cells, rows, total, flopsTok,
    reported: rep ? tokens / rep.rate : null, reportedSrc: rep?.src ?? null };
}

// ---- the widget ------------------------------------------------------------
const GPUS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
const GBSS = [1920, 3840, 7680, 15360, 30720];
const SEQS = [2048, 4096, 8192, 16384, 32768];
const RECOMP_UI = [['none', 'none'], ['dsv3', 'DeepSeek'], ['attn-replay', 'attention replay'], ['full', 'full']];
const HW_UI = [['h800', 'H800'], ['h100', 'H100']];
const W = 900, LAB = 74, PW = W - LAB - 60, ROWH = 22, TOP = 32, BARH = 12;
const AXY = TOP + 4 * ROWH + 6;
const HB = AXY + 26;
const fmtS = (s) => (s >= 100 ? s.toFixed(0) : s >= 10 ? s.toFixed(1) : s >= 1 ? s.toFixed(2) : (s * 1e3).toFixed(0) + ' m') + (s >= 1 ? ' s' : 's');
const fmtF = (f) => f >= 1e9 ? (f / 1e9).toFixed(1) + ' GFLOP' : (f / 1e6).toFixed(1) + ' MFLOP';
const fitEase = (p) => 1 - (1 - p) ** 3;
const lerp = (a, b, t) => a + (b - a) * t;

const CSS = `
dsv3-sol { display: block; margin: 14px 0 26px; }
.so { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-0b0b0b);
  border: 1px solid var(--c-e1e0d9); border-radius: 6px; background: var(--c-fcfcfb); padding: 8px 10px;
  width: max-content; max-width: 100%; box-sizing: border-box; }
.so .top { display: flex; align-items: stretch; gap: 10px; padding-bottom: 8px; flex-wrap: wrap; }
${knobCss('.so .top')}
.so svg { display: block; }
.so .dims { font: 9px system-ui; fill: var(--c-898781); }
.so .rlab { font: 11px system-ui; fill: var(--c-52514e); }
.so .ro { font-size: 11.5px; color: var(--c-52514e); min-height: 50px; margin-top: 4px; max-width: ${W}px; line-height: 1.45; }
.so .ro b { color: var(--c-0b0b0b); font-weight: 600; }
.so .sheet { display: grid; grid-template-columns: auto auto; column-gap: 8px; row-gap: 1px; font: 10.5px ui-monospace, Menlo, monospace;
  color: var(--c-52514e); padding: 3px 0 0; margin-top: 3px; border-top: 1px solid var(--c-e1e0d9); }
.so .sheet span:nth-child(odd) { color: var(--c-898781); font: 10px system-ui; }
.so .sheet span:nth-child(even) { text-align: right; min-width: 12ch; }
`;

class Dsv3Sol extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    const A = (k) => this.getAttribute(k);
    const st = this.id ? readState('s:' + this.id) : null;
    this.cfg = { ...SOL_DEFAULTS };
    for (const k of Object.keys(SOL_DEFAULTS)) {
      const v = st?.[k] ?? A(k);
      if (v != null) this.cfg[k] = typeof SOL_DEFAULTS[k] === 'number' ? +v : v;
    }
    const style = document.createElement('style'); style.textContent = CSS;
    this._root = el('div', 'so');
    this._top = el('div', 'top');
    this._chart = el('div');
    this._ro = el('div', 'ro');
    this._root.append(this._top, this._chart, this._ro);
    this.append(style, this._root);
    this._buildKnobs();
    this._L = null;
    this.render();
    addEventListener('dsv3-theme', () => this.render());
  }
  _buildKnobs() {
    const grp = (label) => { const g = el('span', 'pargrp'); const l = el('div', 'parlab'); l.textContent = label; g.append(l); return g; };
    const row = (...kids) => { const r = el('div', 'parrow'); r.append(...kids); return r; };
    const txt = (s) => { const t = el('span'); t.style.cssText = 'color:var(--c-52514e);font-size:11px;'; t.textContent = s; return t; };
    const stepper = (key, opts) => {
      const eg = el('span', 'stp'); eg.dataset.knob = key;
      const sel = document.createElement('select'); sel.className = 'v';
      for (const o of opts) sel.append(new Option(o.toLocaleString('en-US'), o));
      sel.value = this.cfg[key];
      const b = (t, di) => { const x = document.createElement('button'); x.type = 'button'; x.textContent = t; x.dataset.dir = di;
        x.onclick = () => { const j = opts.indexOf(this.cfg[key]) + di; if (opts[j] != null) this._set(key, opts[j]); }; return x; };
      sel.onchange = () => this._set(key, +sel.value);
      eg.append(b('−', -1), sel, b('+', +1));
      this._ui[key] = { sel, eg, opts };
      return eg;
    };
    const seg = (key, opts) => {
      const eg = el('span', 'stp'); eg.dataset.knob = key;
      for (const [v, lab] of opts) {
        const x = document.createElement('button'); x.type = 'button'; x.textContent = lab; x.dataset.v = v;
        x.onclick = () => this._set(key, v);
        eg.append(x);
      }
      this._ui[key] = { eg };
      return eg;
    };
    this._ui = {};
    // the GPU group carries its tech sheet: the three data-sheet numbers the whole model runs on
    this._sheet = el('div', 'sheet');
    const g1 = grp('GPU'); g1.append(row(seg('hw', HW_UI)), this._sheet);
    const g3 = grp('recompute'); g3.append(row(seg('recompute', RECOMP_UI)));
    const g4 = grp('cluster'); g4.append(row(txt('GPUs'), stepper('gpus', GPUS)));
    const g5 = grp('batch'); g5.append(row(txt('seqs/step'), stepper('gbs', GBSS), txt('× seq'), stepper('seq', SEQS)));
    this._top.append(g1, g3, g4, g5);
    this._syncKnobs();
  }
  _syncKnobs() {
    this._sheet.innerHTML = SHEET(HARDWARE[this.cfg.hw]).map(([k, v]) => `<span>${k}</span><span data-sheet="${k}">${v}</span>`).join('');
    for (const [k, u] of Object.entries(this._ui)) {
      if (u.sel) {
        u.sel.value = this.cfg[k];
        const j = u.opts.indexOf(this.cfg[k]);
        for (const b of u.eg.querySelectorAll('button')) b.disabled = u.opts[j + +b.dataset.dir] == null;
      } else for (const b of u.eg.querySelectorAll('button')) b.classList.toggle('on', b.dataset.v == this.cfg[k]);
    }
  }
  _set(k, v) {
    if (this.cfg[k] === v) return;
    this.cfg[k] = v;
    this._syncKnobs();
    if (this.id) writeState('s:' + this.id, this.cfg);
    this._animateTo(this._layout());
  }
  // pixel-space layout: the axis autoscales to whatever is longest (the
  // total, its untreated ghost, or the reported step), so a knob flip that
  // changes the scale slides every bar at once
  _layout() {
    const S = solStep(this.cfg);
    // the untreated anchor: nothing recomputed, same tokens
    const anchor = solStep({ ...this.cfg, recompute: 'none' }).total;
    const span = Math.max(S.total, anchor, S.reported ?? 0) * 1.06;
    const px = (s) => LAB + s / span * PW;
    const rows = [];
    for (const [i, p] of [...PASSES, { id: 'total', label: 'total' }].entries()) {
      const y = TOP + i * ROWH;
      let cum = 0; const segs = [];
      for (const g of GROUPS) {
        const cs = S.cells.filter((c) => c.group === g.id && (p.id === 'total' || c.pass === p.id));
        const s = cs.reduce((t, c) => t + c.s, 0);
        if (s > 0) segs.push({ k: g.id, x0: px(cum), x1: px(cum + s), c: g.c, v: s, flops: cs.reduce((t, c) => t + c.flops, 0) });
        cum += s;
      }
      rows.push({ id: p.id, label: p.label, y, segs, v: cum });
    }
    return { S, rows, span, px, anchorX: px(anchor), anchor, totalX: px(S.total), repX: S.reported ? px(S.reported) : null };
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
  render() { this._L = this._layout(); this._draw(this._L); }
  _draw(L) {
    const f1 = (v) => v.toFixed(1);
    const B = [];
    const dims = 'class="dims"';
    // legend
    let lx = LAB;
    for (const g of GROUPS) {
      B.push(`<rect x="${lx}" y="6" width="8" height="8" fill="${C(g.c)}"/><text ${dims} x="${lx + 11}" y="13">${g.label}</text>`);
      lx += 11 + g.label.length * 5 + 14;
    }
    // axis: nice ticks in seconds over the current span (labels snap, positions ride the tween)
    const step = niceStep(L.span / 8);
    for (let s = 0; s <= L.span; s += step) {
      const x = L.px(s);
      B.push(`<line x1="${f1(x)}" y1="${TOP - 6}" x2="${f1(x)}" y2="${AXY}" stroke="${C('#e1e0d9')}" stroke-width="1"/>`);
      B.push(`<text ${dims} x="${f1(x)}" y="${AXY + 11}" text-anchor="middle">${+s.toPrecision(3)}</text>`);
    }
    B.push(`<text ${dims} x="${LAB + PW + 8}" y="${AXY + 11}">s per step</text>`);
    for (const r of L.rows) {
      B.push(`<g data-row="${r.id}">`);
      B.push(`<text class="rlab" x="${LAB - 8}" y="${f1(r.y + BARH - 2)}" text-anchor="end">${r.label}</text>`);
      for (const s of r.segs) if (s.x1 - s.x0 > 0.3)
        B.push(`<rect data-seg="${s.k}" data-true="${s.v}" x="${f1(s.x0)}" y="${r.y}" width="${f1(Math.max(0.5, s.x1 - s.x0))}" height="${BARH}" fill="${C(s.c)}"/>`);
      const end = r.segs.length ? r.segs[r.segs.length - 1].x1 : LAB;
      B.push(`<text ${dims} data-rowval="${r.v}" x="${f1(end + 5)}" y="${f1(r.y + BARH - 2)}">${r.v ? fmtS(r.v) : '—'}</text>`);
      B.push(`<rect data-hit="${r.id}" x="0" y="${r.y - 4}" width="${W}" height="${ROWH}" fill="transparent"/>`);
      B.push('</g>');
    }
    // the no-recompute tick on the total row: where the step would end with nothing replayed
    const tr = L.rows[3];
    if (L.anchorX < L.totalX - 1)
      B.push(`<line data-ghost x1="${f1(L.anchorX)}" y1="${tr.y - 3}" x2="${f1(L.anchorX)}" y2="${tr.y + BARH + 3}" stroke="${C('#898781')}" stroke-dasharray="2 2"/>`);
    if (L.repX != null) {
      B.push(`<line data-rep="${L.S.reported}" x1="${f1(L.repX)}" y1="${TOP - 6}" x2="${f1(L.repX)}" y2="${AXY}" stroke="${C('#0b0b0b')}" stroke-dasharray="4 3"/>`);
      B.push(`<text ${dims} x="${f1(L.repX)}" y="${TOP - 9}" text-anchor="middle">reported step ${fmtS(L.S.reported)}</text>`);
    }
    this._chart.innerHTML = `<svg width="${W}" height="${HB}" viewBox="0 0 ${W} ${HB}">${B.join('')}</svg>`;
    const svg = this._chart.firstChild;
    svg.onmouseover = (e) => { const g = e.target.closest('g[data-row]'); if (g) this._readout(L, g.dataset.row, e.target.dataset.seg ?? null); };
    svg.onmouseleave = () => this._readout(L, null, null);
    this._readout(L, this._hover?.row ?? null, this._hover?.seg ?? null);
  }
  _readout(L, row, seg) {
    this._hover = row ? { row, seg } : null;
    const S = L.S, tok = S.tokens.toLocaleString('en-US');
    const peak = `${(S.hw.flops.fp8 / 1e12).toLocaleString('en-US')} TFLOP/s`;
    if (row && seg) {
      const r = L.rows.find((q) => q.id === row), s = r.segs.find((q) => q.k === seg);
      const g = GROUPS.find((q) => q.id === seg);
      this._ro.innerHTML = `<b>${g.label} · ${r.label}</b>: ${fmtF(s.flops)}/token × ${tok} tokens ÷ ${peak} = <b>${fmtS(s.v)}</b> (${(s.v / S.total * 100).toFixed(1)}% of the step).`;
      return;
    }
    if (row) {
      const r = L.rows.find((q) => q.id === row);
      const parts = r.segs.map((s) => `${GROUPS.find((q) => q.id === s.k).label} ${fmtS(s.v)}`).join(' + ');
      this._ro.innerHTML = `<b>${r.label}</b> = ${parts || 'nothing'}${r.v ? ` = <b>${fmtS(r.v)}</b>` : ''}.`;
      return;
    }
    const rep = S.reported ? ` The dashed line is the ${fmtS(S.reported)} step ${S.reportedSrc} (${Math.round(REPORTED[S.cfg.hw].rate).toLocaleString('en-US')} tok/s/GPU): ×${(S.reported / S.total).toFixed(1)} the sum.` : ' No reported throughput for this GPU.';
    const ghost = L.anchor < S.total * 0.999 ? ` The dashed tick is the same step with nothing recomputed, ${fmtS(L.anchor)}.` : '';
    this._ro.innerHTML = `<b>${fmtS(S.total)} per step</b> = forward ${fmtS(S.rows.fwd)} + backward ${fmtS(S.rows.bwd)}` +
      (S.rows.replay ? ` + recompute ${fmtS(S.rows.replay)}` : '') +
      `: ${tok} tokens per GPU (${S.cfg.gbs.toLocaleString('en-US')} × ${S.cfg.seq.toLocaleString('en-US')} ÷ ${S.cfg.gpus.toLocaleString('en-US')}) ` +
      `× ${fmtF(S.flopsTok)}/token ÷ ${peak}, summed.${ghost}${rep}`;
  }
}
function niceStep(raw) {
  const p = 10 ** Math.floor(Math.log10(raw)), m = raw / p;
  return p * (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10);
}
function blend(A, B, t) {
  if (!A || t >= 1) return B;
  const seg = (a, b) => ({ ...b, x0: lerp(a.x0, b.x0, t), x1: lerp(a.x1, b.x1, t) });
  const rows = B.rows.map((b, i) => {
    const a = A.rows[i];
    return { ...b, segs: b.segs.map((s) => { const as = a.segs.find((q) => q.k === s.k) ?? { x0: s.x0, x1: s.x0 }; return seg(as, s); }) };
  });
  const span = lerp(A.span, B.span, t);
  return { ...B, rows, span, px: (s) => LAB + s / span * PW,
    anchorX: lerp(A.anchorX, B.anchorX, t), totalX: lerp(A.totalX, B.totalX, t),
    repX: B.repX == null ? null : lerp(A.repX ?? B.repX, B.repX, t) };
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
if (typeof customElements !== 'undefined' && !customElements.get('dsv3-sol')) customElements.define('dsv3-sol', Dsv3Sol);
