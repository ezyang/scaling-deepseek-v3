// The first picture: one matmul on both meters. A GEMM Y = X · W with X of
// M tokens × K and W of K × N costs 2·M·K·N FLOP on the tensor cores and
// moves X + W + Y through HBM; each is a time on its own meter, and the
// kernel can finish no sooner than the longer. <dsv3-gemm> draws the two on
// one µs axis for one of DeepSeek-V3's own projections, with the token count
// M as the knob that walks a shape from weight-read-bound (a few tokens per
// expert) to compute-bound (a whole microbatch through a dense projection),
// and the dtype knob that halves the input bytes and doubles the peak at once.
import { DSV3, HARDWARE } from './model.js';
import { C } from './theme.js';
import { knobCss } from './ui.js';

export const GEMM_DEFAULTS = { hw: 'h800', dtype: 'fp8', shape: 'expert_up', tokens: 4096 };
// GEMM inputs at the dtype (bytes per number, tensor peak key, the
// precision-family color); the output is always written bf16
export const DTYPES = { bf16: { b: 2, peak: 'bf16', c: '#0b0b0b' }, fp8: { b: 1, peak: 'fp8', c: '#d6408b' } };
const OUT_B = 2;
// the model's projections, K → N
const a = DSV3;
export const SHAPES = {
  q_down: { label: 'MLA q down', K: a.hidden, N: a.qRank },
  kv_down: { label: 'MLA kv down', K: a.hidden, N: a.kvRank + a.qkRope },
  q_up: { label: 'MLA q up', K: a.qRank, N: a.heads * (a.qkNope + a.qkRope) },
  kv_up: { label: 'MLA kv up', K: a.kvRank, N: a.heads * (a.qkNope + a.vHead) },
  o_proj: { label: 'MLA output', K: a.heads * a.vHead, N: a.hidden },
  expert_up: { label: 'expert gate + up', K: a.hidden, N: 2 * a.moeInter },
  expert_down: { label: 'expert down', K: a.moeInter, N: a.hidden },
  dense_up: { label: 'dense FFN gate + up', K: a.hidden, N: 2 * a.denseInter },
  head: { label: 'lm head', K: a.hidden, N: a.vocab },
};
export const TOKENS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768];
const US = 1e-6;

// every number the picture shows, for one config
export function gemm(cfg) {
  cfg = { ...GEMM_DEFAULTS, ...cfg };
  const hw = HARDWARE[cfg.hw], d = DTYPES[cfg.dtype], sh = SHAPES[cfg.shape], M = cfg.tokens, { K, N } = sh;
  const peak = hw.flops[d.peak];
  const flops = 2 * M * K * N;
  const bytes = { X: M * K * d.b, W: K * N * d.b, Y: M * N * OUT_B };
  const total = bytes.X + bytes.W + bytes.Y;
  const compute = flops / peak, mem = total / hw.hbm;
  return { cfg, hw, d, sh, M, K, N, peak, flops, bytes, total, compute, mem,
    intensity: flops / total, ridge: peak / hw.hbm, bound: compute >= mem ? 'compute' : 'memory',
    window: { flop: peak * US, hbmBytes: hw.hbm * US, hbmNums: hw.hbm * US / d.b } };
}

// ---- the widget ------------------------------------------------------------
const W = 900, LAB = 96, PW = W - LAB - 150, TOP = 30, ROWH = 26, BARH = 12;
const AXY = TOP + 2 * ROWH + 4, HB = AXY + 26;
const n = (x, d = 0) => x.toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d });
const fmtUs = (s) => s >= 1e-3 ? `${n(s * 1e3, 2)} ms` : `${n(s / US, s / US >= 10 ? 1 : 2)} µs`;
const fmtB = (b, d = 1) => b >= 1e9 ? `${n(b / 1e9, 2)} GB` : b >= 1e6 ? `${n(b / 1e6, d)} MB` : `${n(b / 1e3, 1)} KB`;
const fmtF = (f) => f >= 1e12 ? `${n(f / 1e12, 2)} TFLOP` : f >= 1e9 ? `${n(f / 1e9, f >= 1e10 ? 1 : 2)} GFLOP` : `${n(f / 1e6)} MFLOP`;
const fmtM = (x) => x >= 1e9 ? `${n(x / 1e9, 2)} G` : `${n(x / 1e6, x >= 1e8 ? 0 : x >= 1e7 ? 1 : 2)} M`;
const fitEase = (p) => 1 - (1 - p) ** 3;
const lerp = (a, b, t) => a + (b - a) * t;

const CSS = `
dsv3-gemm { display: block; margin: 14px 0 26px; }
.gm { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-0b0b0b);
  border: 1px solid var(--c-e1e0d9); border-radius: 6px; background: var(--c-fcfcfb); padding: 8px 10px;
  width: max-content; max-width: 100%; box-sizing: border-box; }
.gm .top { display: flex; align-items: stretch; gap: 10px; padding-bottom: 8px; flex-wrap: wrap; }
${knobCss('.gm .top')}
.gm svg { display: block; }
.gm .dims { font: 9px system-ui; fill: var(--c-898781); }
.gm .rlab { font: 11px system-ui; fill: var(--c-52514e); }
.gm .ro { font-size: 11.5px; color: var(--c-52514e); min-height: 66px; margin-top: 4px; max-width: ${W}px; line-height: 1.45; }
.gm .ro b { color: var(--c-0b0b0b); font-weight: 600; }
.gm .sheet { display: grid; grid-template-columns: auto auto; column-gap: 8px; row-gap: 1px; font: 10.5px ui-monospace, Menlo, monospace;
  color: var(--c-52514e); padding: 3px 0 0; margin-top: 3px; border-top: 1px solid var(--c-e1e0d9); }
.gm .sheet span:nth-child(odd) { color: var(--c-898781); font: 10px system-ui; }
.gm .sheet span:nth-child(even) { text-align: right; }
.gm .note { font: 10.5px ui-monospace, Menlo, monospace; color: var(--c-52514e); min-width: 24ch; }
`;

class Dsv3Gemm extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    const st = this.id ? readState('g:' + this.id) : null;
    this.cfg = { ...GEMM_DEFAULTS };
    for (const k of Object.keys(GEMM_DEFAULTS)) {
      const v = st?.[k] ?? this.getAttribute(k);
      if (v != null) this.cfg[k] = typeof GEMM_DEFAULTS[k] === 'number' ? +v : v;
    }
    const style = document.createElement('style'); style.textContent = CSS;
    this._root = el('div', 'gm');
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
    const select = (key, opts) => {
      const sel = document.createElement('select'); sel.dataset.knob = key;
      for (const [v, lab] of opts) sel.append(new Option(lab, v));
      sel.value = this.cfg[key];
      sel.onchange = () => this._set(key, sel.value);
      this._ui[key] = { sel, eg: sel, opts: opts.map(([v]) => v), plain: true };
      return sel;
    };
    this._ui = {};
    const hw = HARDWARE[this.cfg.hw];
    // the GPU group carries what one microsecond buys on each meter
    this._sheet = el('div', 'sheet');
    const g1 = grp(`${hw.label.split(' ')[0]} · in 1 µs`); g1.append(this._sheet);
    const g2 = grp('inputs stored as'); g2.append(row(seg('dtype', Object.keys(DTYPES).map((k) => [k, k]))));
    const g3 = grp('the matmul: X · W, with W ='); this._note = el('div', 'note');
    g3.append(row(select('shape', Object.entries(SHAPES).map(([k, s]) => [k, s.label])), txt('and X ='), stepper('tokens', TOKENS), txt('tokens:'), this._note));
    this._top.append(g1, g2, g3);
    this._syncKnobs();
  }
  _syncKnobs() {
    const G = gemm(this.cfg);
    this._sheet.innerHTML = [
      [`tensor cores, ${G.cfg.dtype} matmul`, fmtF(G.window.flop)],
      ['HBM', `${fmtB(G.window.hbmBytes, 2)} = ${fmtM(G.window.hbmNums)} ${G.cfg.dtype} numbers`],
      ['FLOP per HBM byte', n(G.ridge)],
    ].map(([k, v]) => `<span>${k}</span><span data-sheet="${k.split(',')[0]}">${v}</span>`).join('');
    this._note.innerHTML = `${n(G.M)} × ${n(G.K)} @ ${n(G.K)} × ${n(G.N)}`;
    for (const [k, u] of Object.entries(this._ui)) {
      if (u.plain) u.sel.value = this.cfg[k];
      else if (u.sel) {
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
    if (this.id) writeState('g:' + this.id, this.cfg);
    this._animateTo(this._layout());
  }
  // the axis autoscales to the longer meter (labels snap, positions tween)
  _layout() {
    const G = gemm(this.cfg);
    const span = Math.max(G.compute, G.mem) * 1.06;
    const px = (s) => LAB + s / span * PW;
    const bw = G.hw.hbm;
    return { G, span, px, computeX: px(G.compute), xX: px(G.bytes.X / bw), wX: px((G.bytes.X + G.bytes.W) / bw), memX: px(G.mem), c: G.d.c };
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
    const f1 = (v) => v.toFixed(1), G = L.G;
    const B = [], dims = 'class="dims"';
    // the one-microsecond window, shaded, at the origin of the axis
    B.push(`<rect data-window x="${LAB}" y="${TOP - 6}" width="${f1(Math.max(0, L.px(US) - LAB))}" height="${AXY - TOP + 6}" fill="${C('#f3f2ee')}"/>`);
    B.push(`<text ${dims} x="${LAB}" y="${TOP - 10}">1 µs</text>`);
    // axis: nice ticks in µs
    const step = niceStep(L.span / US / 8) * US;
    for (let s = 0; s <= L.span; s += step) {
      const x = L.px(s);
      B.push(`<line x1="${f1(x)}" y1="${TOP - 6}" x2="${f1(x)}" y2="${AXY}" stroke="${C('#e1e0d9')}" stroke-width="1"/>`);
      B.push(`<text ${dims} x="${f1(x)}" y="${AXY + 11}" text-anchor="middle">${+(s / US).toPrecision(3)}</text>`);
    }
    B.push(`<text ${dims} x="${LAB + PW + 8}" y="${AXY + 11}">µs</text>`);
    // row 1: the FLOPs on the tensor cores
    const y1 = TOP, y2 = TOP + ROWH;
    B.push(`<text class="rlab" x="${LAB - 8}" y="${f1(y1 + BARH - 2)}" text-anchor="end">tensor cores</text>`);
    B.push(`<rect data-bar="compute" data-true="${G.compute}" x="${LAB}" y="${y1}" width="${f1(Math.max(1.5, L.computeX - LAB))}" height="${BARH}" fill="${C('#898781')}"/>`);
    B.push(`<text ${dims} data-val="compute" x="${f1(Math.max(L.computeX, LAB + 1.5) + 5)}" y="${f1(y1 + BARH - 2)}">${fmtF(G.flops)} · ${fmtUs(G.compute)}</text>`);
    // row 2: the bytes through HBM — X, W, then Y, end to end (one pipe)
    B.push(`<text class="rlab" x="${LAB - 8}" y="${f1(y2 + BARH - 2)}" text-anchor="end">HBM</text>`);
    B.push(`<rect data-bar="X" data-true="${G.bytes.X}" x="${LAB}" y="${y2}" width="${f1(L.xX - LAB)}" height="${BARH}" fill="${C(L.c)}"/>`);
    B.push(`<rect data-bar="W" data-true="${G.bytes.W}" x="${f1(L.xX)}" y="${y2}" width="${f1(L.wX - L.xX)}" height="${BARH}" fill="${C(L.c)}" opacity="0.55"/>`);
    B.push(`<rect data-bar="Y" data-true="${G.bytes.Y}" x="${f1(L.wX)}" y="${y2}" width="${f1(L.memX - L.wX)}" height="${BARH}" fill="${C('#0b0b0b')}" opacity="0.3"/>`);
    B.push(`<text ${dims} data-val="mem" x="${f1(L.memX + 5)}" y="${f1(y2 + BARH - 2)}">X ${fmtB(G.bytes.X)} + W ${fmtB(G.bytes.W)} + Y ${fmtB(G.bytes.Y)} · ${fmtUs(G.mem)}</text>`);
    this._chart.innerHTML = `<svg width="${W}" height="${HB}" viewBox="0 0 ${W} ${HB}">${B.join('')}</svg>`;
    this._readout(G);
  }
  _readout(G) {
    const t = G.cfg.dtype, r = G.compute / G.mem;
    this._ro.innerHTML = `<b>${G.sh.label}</b>, ${n(G.M)} tokens: 2 × ${n(G.M)} × ${n(G.K)} × ${n(G.N)} = ${fmtF(G.flops)} ÷ ${n(G.peak / 1e12)} TFLOP/s = <b>${fmtUs(G.compute)}</b> of tensor-core time. ` +
      `X ${n(G.M)} × ${n(G.K)} × ${G.d.b} B + W ${n(G.K)} × ${n(G.N)} × ${G.d.b} B + Y ${n(G.M)} × ${n(G.N)} × ${OUT_B} B (bf16 out) = ${fmtB(G.total)} ÷ ${n(G.hw.hbm / 1e12, 2)} TB/s = <b>${fmtUs(G.mem)}</b> of HBM time. ` +
      `${n(G.intensity)} FLOP per byte against the ${n(G.ridge)} the tensor cores finish per HBM byte: ` +
      (G.bound === 'compute' ? `<b>compute-bound</b>, ×${n(r, r < 10 ? 1 : 0)} — the bytes hide under the FLOPs.` : `<b>memory-bound</b>, ×${n(1 / r, 1 / r < 10 ? 1 : 0)} — the tensor cores idle while W streams in.`) +
      ` In 1 µs: ${fmtF(G.window.flop)} against ${fmtB(G.window.hbmBytes, 2)}.`;
  }
}
function niceStep(raw) {
  const p = 10 ** Math.floor(Math.log10(raw)), m = raw / p;
  return p * (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10);
}
function blend(A, B, t) {
  if (!A || t >= 1) return B;
  const span = lerp(A.span, B.span, t);
  return { ...B, span, px: (s) => LAB + s / span * PW, computeX: lerp(A.computeX, B.computeX, t),
    xX: lerp(A.xX, B.xX, t), wX: lerp(A.wX, B.wX, t), memX: lerp(A.memX, B.memX, t) };
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
if (typeof customElements !== 'undefined' && !customElements.get('dsv3-gemm')) customElements.define('dsv3-gemm', Dsv3Gemm);
