// The first picture: what a fixed slice of time buys on one GPU. In one
// microsecond the tensor cores finish so many FLOP, the CUDA cores so many
// fp32 ops, and HBM moves so many bytes — which is so many NUMBERS once a
// dtype fixes the bytes per number. Then the simplest kernel there is, an
// elementwise op over one tensor, priced on both meters: its arithmetic on
// the CUDA cores and its bytes (read the inputs, write the result) through
// HBM, on one time axis. <dsv3-clock> draws the two rows; the dtype knob
// scales the bytes and leaves the arithmetic alone, the size knob scales both.
import { HARDWARE } from './model.js';
import { C } from './theme.js';
import { knobCss } from './ui.js';

export const CLOCK_DEFAULTS = { hw: 'h800', dtype: 'bf16', tokens: 4096 };
// storage dtypes: bytes per number, the precision-family color
export const DTYPES = { fp32: { b: 4, c: '#8a3324' }, bf16: { b: 2, c: '#0b0b0b' }, fp8: { b: 1, c: '#d6408b' } };
// the kernel is x × 2: read the tensor once, write it once, one multiply per
// number; the tensor is tokens × 7,168 hidden states
export const HIDDEN = 7168;
export const TOKENS = [512, 1024, 2048, 4096, 8192, 16384, 32768];
const US = 1e-6;

// every number the picture shows, for one config
export function clock(cfg) {
  cfg = { ...CLOCK_DEFAULTS, ...cfg };
  const hw = HARDWARE[cfg.hw], { b } = DTYPES[cfg.dtype], N = cfg.tokens * HIDDEN;
  // fp32 CUDA-core peak counts an FMA as two FLOP; a lone multiply or add
  // fills one FMA slot, so numbers per second is half the FLOP rate
  const opsRate = hw.flops.fp32 / 2;
  const window = { tensor: hw.flops.fp8 * US, cuda: hw.flops.fp32 * US, hbmBytes: hw.hbm * US, hbmNums: hw.hbm * US / b };
  const bytes = N * b;
  const compute = N / opsRate, read = bytes / hw.hbm, write = bytes / hw.hbm;
  return { cfg, hw, b, N, opsRate, window, bytes, compute, read, write, mem: read + write, ratio: (read + write) / compute };
}

// ---- the widget ------------------------------------------------------------
const W = 900, LAB = 96, PW = W - LAB - 96, TOP = 30, ROWH = 26, BARH = 12;
const AXY = TOP + 2 * ROWH + 4, HB = AXY + 26;
const n = (x, d = 0) => x.toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d });
const fmtUs = (s) => `${n(s / US, s / US >= 10 ? 1 : 2)} µs`;
const fmtB = (bytes, d = 1) => bytes >= 1e6 ? `${n(bytes / 1e6, d)} MB` : `${n(bytes / 1e3, 1)} KB`;
const fmtM = (x) => x >= 1e9 ? `${n(x / 1e9, 2)} G` : `${n(x / 1e6, x >= 1e8 ? 0 : x >= 1e7 ? 1 : 2)} M`;
const fitEase = (p) => 1 - (1 - p) ** 3;
const lerp = (a, b, t) => a + (b - a) * t;

const CSS = `
dsv3-clock { display: block; margin: 14px 0 26px; }
.ck { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-0b0b0b);
  border: 1px solid var(--c-e1e0d9); border-radius: 6px; background: var(--c-fcfcfb); padding: 8px 10px;
  width: max-content; max-width: 100%; box-sizing: border-box; }
.ck .top { display: flex; align-items: stretch; gap: 10px; padding-bottom: 8px; flex-wrap: wrap; }
${knobCss('.ck .top')}
.ck svg { display: block; }
.ck .dims { font: 9px system-ui; fill: var(--c-898781); }
.ck .rlab { font: 11px system-ui; fill: var(--c-52514e); }
.ck .ro { font-size: 11.5px; color: var(--c-52514e); min-height: 50px; margin-top: 4px; max-width: ${W}px; line-height: 1.45; }
.ck .ro b { color: var(--c-0b0b0b); font-weight: 600; }
.ck .sheet { display: grid; grid-template-columns: auto auto; column-gap: 8px; row-gap: 1px; font: 10.5px ui-monospace, Menlo, monospace;
  color: var(--c-52514e); padding: 3px 0 0; margin-top: 3px; border-top: 1px solid var(--c-e1e0d9); }
.ck .sheet span:nth-child(odd) { color: var(--c-898781); font: 10px system-ui; }
.ck .sheet span:nth-child(even) { text-align: right; }
.ck .note { font-size: 11px; color: var(--c-52514e); min-width: 12ch; }
`;

class Dsv3Clock extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    const st = this.id ? readState('c:' + this.id) : null;
    this.cfg = { ...CLOCK_DEFAULTS };
    for (const k of Object.keys(CLOCK_DEFAULTS)) {
      const v = st?.[k] ?? this.getAttribute(k);
      if (v != null) this.cfg[k] = typeof CLOCK_DEFAULTS[k] === 'number' ? +v : v;
    }
    const style = document.createElement('style'); style.textContent = CSS;
    this._root = el('div', 'ck');
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
    const hw = HARDWARE[this.cfg.hw];
    // the GPU group carries what one microsecond buys on each meter
    this._sheet = el('div', 'sheet');
    const g1 = grp(`${hw.label.split(' ')[0]} · in 1 µs`); g1.append(this._sheet);
    const g2 = grp('numbers stored as'); g2.append(row(seg('dtype', Object.keys(DTYPES).map((k) => [k, k]))));
    const g3 = grp('the kernel: x × 2 over the hidden states of'); this._note = el('div', 'note');
    g3.append(row(stepper('tokens', TOKENS), txt('tokens × 7,168 ='), this._note));
    this._top.append(g1, g2, g3);
    this._syncKnobs();
  }
  _syncKnobs() {
    const K = clock(this.cfg);
    this._sheet.innerHTML = [
      ['tensor cores, FP8 matmul', `${fmtM(K.window.tensor)}FLOP`],
      ['CUDA cores, fp32', `${fmtM(K.window.cuda)}FLOP = ${fmtM(K.window.cuda / 2)} FMA`],
      ['HBM', `${fmtB(K.window.hbmBytes, 2)} = ${fmtM(K.window.hbmNums)} ${K.cfg.dtype} numbers`],
    ].map(([k, v]) => `<span>${k}</span><span data-sheet="${k.split(',')[0]}">${v}</span>`).join('');
    this._note.innerHTML = `<b>${fmtM(K.N)}</b> numbers`;
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
    if (this.id) writeState('c:' + this.id, this.cfg);
    this._animateTo(this._layout());
  }
  // the axis spans the fp32 case at the current size, so a dtype flip visibly
  // halves the byte bars while the arithmetic bar holds; a size change
  // rescales the axis (labels snap, positions tween)
  _layout() {
    const K = clock(this.cfg);
    const worst = clock({ ...this.cfg, dtype: 'fp32' });
    const span = worst.mem * 1.06;
    const px = (s) => LAB + s / span * PW;
    return { K, span, px, computeX: px(K.compute), readX: px(K.read), memX: px(K.mem), c: DTYPES[K.cfg.dtype].c };
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
    const f1 = (v) => v.toFixed(1), K = L.K;
    const B = [], dims = 'class="dims"';
    // the one-microsecond window, shaded, at the origin of the axis
    B.push(`<rect data-window x="${LAB}" y="${TOP - 6}" width="${f1(L.px(US) - LAB)}" height="${AXY - TOP + 6}" fill="${C('#f3f2ee')}"/>`);
    B.push(`<text ${dims} x="${LAB}" y="${TOP - 10}">1 µs</text>`);
    // axis: nice ticks in µs
    const step = niceStep(L.span / US / 8) * US;
    for (let s = 0; s <= L.span; s += step) {
      const x = L.px(s);
      B.push(`<line x1="${f1(x)}" y1="${TOP - 6}" x2="${f1(x)}" y2="${AXY}" stroke="${C('#e1e0d9')}" stroke-width="1"/>`);
      B.push(`<text ${dims} x="${f1(x)}" y="${AXY + 11}" text-anchor="middle">${+(s / US).toPrecision(3)}</text>`);
    }
    B.push(`<text ${dims} x="${LAB + PW + 8}" y="${AXY + 11}">µs</text>`);
    // row 1: the arithmetic on the CUDA cores
    const y1 = TOP, y2 = TOP + ROWH;
    B.push(`<text class="rlab" x="${LAB - 8}" y="${f1(y1 + BARH - 2)}" text-anchor="end">CUDA cores</text>`);
    B.push(`<rect data-bar="compute" data-true="${K.compute}" x="${LAB}" y="${y1}" width="${f1(Math.max(1.5, L.computeX - LAB))}" height="${BARH}" fill="${C('#898781')}"/>`);
    B.push(`<text ${dims} data-val="compute" x="${f1(Math.max(L.computeX, LAB + 1.5) + 5)}" y="${f1(y1 + BARH - 2)}">${fmtM(K.N)} multiplies · ${fmtUs(K.compute)}</text>`);
    // row 2: the bytes through HBM — reads, then the write, end to end (one pipe)
    B.push(`<text class="rlab" x="${LAB - 8}" y="${f1(y2 + BARH - 2)}" text-anchor="end">HBM</text>`);
    B.push(`<rect data-bar="read" data-true="${K.read}" x="${LAB}" y="${y2}" width="${f1(L.readX - LAB)}" height="${BARH}" fill="${C(L.c)}"/>`);
    B.push(`<rect data-bar="write" data-true="${K.write}" x="${f1(L.readX)}" y="${y2}" width="${f1(L.memX - L.readX)}" height="${BARH}" fill="${C(L.c)}" opacity="0.45"/>`);
    B.push(`<text ${dims} data-val="mem" x="${f1(L.memX + 5)}" y="${f1(y2 + BARH - 2)}">read ${fmtB(K.bytes)} + write ${fmtB(K.bytes)} · ${fmtUs(K.mem)}</text>`);
    this._chart.innerHTML = `<svg width="${W}" height="${HB}" viewBox="0 0 ${W} ${HB}">${B.join('')}</svg>`;
    this._readout(K);
  }
  _readout(K) {
    const t = K.cfg.dtype;
    this._ro.innerHTML = `<b>x × 2</b> over ${fmtM(K.N)} ${t} numbers: the multiplies are ${fmtM(K.N)} ÷ ${n(K.opsRate / 1e12, 1)} T/s = <b>${fmtUs(K.compute)}</b> of CUDA-core time ` +
      `(${n(K.hw.flops.fp32 / 1e12)} TFLOP/s counts an FMA as two FLOP; a lone multiply fills one slot). ` +
      `Moving the numbers is ${fmtB(K.bytes)} in + ${fmtB(K.bytes)} out = ${fmtB(2 * K.bytes)} ÷ ${n(K.hw.hbm / 1e12, 2)} TB/s = <b>${fmtUs(K.mem)}</b> of HBM time. ` +
      `The kernel cannot finish before its bytes do: <b>${fmtUs(K.mem)}</b>, ×${n(K.ratio)} the arithmetic. ` +
      `The 1 µs window is ${fmtM(K.window.hbmNums)} ${t} numbers through HBM against ${fmtM(K.window.cuda / 2)} FMA slots.`;
  }
}
function niceStep(raw) {
  const p = 10 ** Math.floor(Math.log10(raw)), m = raw / p;
  return p * (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10);
}
function blend(A, B, t) {
  if (!A || t >= 1) return B;
  return { ...B, computeX: lerp(A.computeX, B.computeX, t), readX: lerp(A.readX, B.readX, t), memX: lerp(A.memX, B.memX, t) };
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
if (typeof customElements !== 'undefined' && !customElements.get('dsv3-clock')) customElements.define('dsv3-clock', Dsv3Clock);
