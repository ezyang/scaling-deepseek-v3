// The scaling ladder: DeepSeek-V3 as one rung of a family, every rung spending
// its compute budget C the way DSv3 does. Held fixed: tokens per activated
// parameter (14.8T ÷ 36.6B), aspect ratio (layers ∝ width), the attention:FFN
// split (every width dim — hidden, heads, latent ranks, FFN widths — scales
// by one factor s; per-head dims don't), the MoE shape (256 experts, top-8,
// 1 shared) and the vocabulary. Dims scale CONTINUOUSLY (61·s layers, no
// rounding), so DSv3 is exactly s = 1. <dsv3-ladder> draws each op class's
// share of the counted FLOPs, against the share 6ND covers, in two linked
// slices of the (C, S) plane: across C at the cursor's S, and across S at its C.
import { DSV3, HEAD_OPS } from './model.js';
import { blockGraph } from './blockgraph.js';
import { C } from './theme.js';
import { knobCss } from './ui.js';

export const TOKENS = 14.8e12;                      // DSv3 pretraining tokens
const WIDTH = ['hidden', 'qRank', 'kvRank', 'denseInter', 'moeInter', 'heads'];
export function ladderArch(s) {
  const a = { ...DSV3, layers: DSV3.layers * s, denseLayers: DSV3.denseLayers * s };
  for (const k of WIDTH) a[k] = DSV3[k] * s;
  return a;
}
// activated params, DeepSeek's accounting (params.js): no input embedding,
// the head in full, top-k + shared experts
export function activeParams(a) {
  const h = a.hidden, qk = a.qkNope + a.qkRope;
  const mla = h * (a.qRank + a.kvRank + a.qkRope) + a.qRank * a.heads * qk
    + a.kvRank * a.heads * (a.qkNope + a.vHead) + a.heads * a.vHead * h + h + a.qRank + a.kvRank;
  return a.denseLayers * (mla + 3 * h * a.denseInter + h)
    + (a.layers - a.denseLayers) * (mla + (a.topk + a.sharedExperts) * 3 * h * a.moeInter + (h + 1) * a.routedExperts + h)
    + h * a.vocab + h;
}
export const N_DSV3 = activeParams(DSV3);
export const TPP = TOKENS / N_DSV3;
export const LOG_C_DSV3 = Math.log10(6 * N_DSV3 * TOKENS);

// op classes, bottom of the stack first; `in6N` = its weights are counted in N
export const CLASSES = [
  { id: 'proj', label: 'learned projections', scale: 'L·d²', c: '#2a78d6', in6N: true },
  { id: 'router', label: 'router', scale: 'L·d', c: '#8a3324', in6N: true },
  { id: 'head', label: 'lm head', scale: 'd·V', c: '#eda100', in6N: true },
  { id: 'attn', label: 'attention core', scale: 'L·d·S', c: '#c74e1d' },
  { id: 'vector', label: 'non-matmul (rough)', scale: 'L·d', c: '#c3c2b7' },
];
const BF16 = new Proxy({}, { get: () => 'bf16' });   // blockGraph sizes stashes by dtype; unused here

// FLOP/token (forward + backward = 3× forward; the loss is forward-only) by op
// class, for any arch. Model FLOPs: no recompute.
export function classFlops(a, seq) {
  const f = { proj: 0, router: 0, head: 0, attn: 0, vector: 0 };
  for (const [kind, n] of [['dense', a.denseLayers], ['moe', a.layers - a.denseLayers]])
    for (const x of blockGraph(kind, a, BF16, seq)) {
      if (!x.flopsTok) continue;
      const k = x.opKind === 'matmul' ? (x.id === 'router' ? 'router' : 'proj') : x.opKind === 'attn' ? 'attn' : 'vector';
      f[k] += 3 * n * x.flopsTok;
    }
  for (const op of HEAD_OPS(a)) f[op.name === 'lm_head' ? 'head' : 'vector'] += op.ftok * (1 + (op.bwdMult ?? 2));
  return f;
}

// the rung a budget buys: C = 6·N·D with D = TPP·N → N = √(C / 6·TPP); solve N(s)
export function ladderPoint(logC, seq) {
  const N = logC === LOG_C_DSV3 ? N_DSV3 : Math.sqrt(10 ** logC / (6 * TPP));
  let lo = 0, hi = 100;
  if (N === N_DSV3) lo = hi = 1;
  else for (let i = 0; i < 200 && hi - lo > 1e-15; i++) { const m = (lo + hi) / 2; if (activeParams(ladderArch(m)) < N) lo = m; else hi = m; }
  const s = (lo + hi) / 2, a = ladderArch(s);
  const f = classFlops(a, seq);
  const total = Object.values(f).reduce((x, y) => x + y, 0);
  return { s, a, N, D: TPP * N, f, total, sixN: 6 * N };
}

// ---- the widget ------------------------------------------------------------
const LO = 19, HI = 27, K = 161;                    // C panel's x domain (log10 FLOP) and samples
const SLO = 9, SHI = 20;                            // S panel's x domain (log2 tokens)
const SEQS = Array.from({ length: SHI - SLO + 1 }, (_, i) => 2 ** (SLO + i));   // 512 … 1M
const W = 738, X0 = 60, PW = 316, XS = X0 + PW + 30, Y0 = 18, PH = 200, AX = Y0 + PH;   // W: the prose column, so margin notes sit beside it
const LY = AX + 60, LW = 300, H = LY + 3 * 20 + 6;  // legend below the plots: 2 columns × 3 rows
const px = (lc) => X0 + (lc - LO) / (HI - LO) * PW;
const lcOf = (x) => Math.max(LO, Math.min(HI, LO + (x - X0) / PW * (HI - LO)));
const pxs = (ls) => XS + (ls - SLO) / (SHI - SLO) * PW;
const seqOf = (x) => { const v = Math.max(SLO, Math.min(SHI, SLO + (x - XS) / PW * (SHI - SLO)));   // continuous, snapping onto powers of two
  return Math.abs(v - Math.round(v)) < SSNAP ? 2 ** Math.round(v) : Math.round(2 ** v); };
const fmtS = (s) => s >= 2 ** 20 ? s / 2 ** 20 + 'M' : s >= 1024 ? s / 1024 + 'K' : String(s);
const py = (u) => Y0 + (1 - u) * PH;
const YLO = 6, YHI = 14;                            // log mode: y domain (log10 FLOP/token)
const pyl = (lf) => Y0 + (1 - (Math.max(YLO, lf) - YLO) / (YHI - YLO)) * PH;
const YS = ['share', 'log', 'lin'];                 // y-axis modes
const oneHot = (y) => Object.fromEntries(YS.map((k) => [k, +(k === y)]));
const HEAD = 1.05;                                  // linear mode: the ceiling sits this far above the tallest stack
// linear gridlines under a continuous ceiling: each 1·2·5 × 10ᵏ step is a level, opaque
// at ≤ 5 lines and faded out by 10, so a rescale slides and crossfades lines, never pops
// them; a line also fades in through the headroom as the ceiling rises past it
export function linTicks(top) {
  const T = new Map();
  for (let e = Math.floor(Math.log10(top)) - 1; e <= Math.floor(Math.log10(top)); e++)
    for (const m of [1, 2, 5]) {
      const st = m * 10 ** e, o = Math.min(1, Math.max(0, (10 - top / st) / 5));
      for (let k = 1; o > 0 && k * st <= top; k++) {
        const v = +(k * st).toPrecision(6), f = o * Math.min(1, (top - v) / (top * (1 - 1 / HEAD)));
        if (f > (T.get(v) ?? 0)) T.set(v, f);
      }
    }
  return [...T];
}
const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹';
const sup = (n) => String(n).split('').map((d) => SUP[d]).join('');
const fmtC = (lc) => { const e = Math.floor(lc + 1e-9); return `${(10 ** (lc - e)).toFixed(2)}×10${sup(e)}`; };
const fmtN = (n) => n >= 1e12 ? +(n / 1e12).toPrecision(3) + 'T' : n >= 1e9 ? +(n / 1e9).toPrecision(3) + 'B' : +(n / 1e6).toPrecision(3) + 'M';
const fmtF = (f) => f >= 1e12 ? `${+(f / 1e12).toPrecision(3)} TFLOP` : f >= 1e9 ? `${+(f / 1e9).toPrecision(3)} GFLOP` : `${+(f / 1e6).toPrecision(3)} MFLOP`;
const fmtP = (u) => (u * 100).toFixed(u < 0.001 && u > 0 ? 2 : 1) + '%';
const lerp = (a, b, t) => a + (b - a) * t;
const ease = (p) => 1 - (1 - p) ** 3;
const SNAP = 0.03;                                  // decades: the slider snaps onto DSv3
const SSNAP = 0.04;                                 // octaves: an S drag snaps onto the stepper's powers of two

const CSS = `
dsv3-ladder { display: block; margin: 14px 0 26px; }
.ld { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-0b0b0b);
  border: 1px solid var(--c-e1e0d9); border-radius: 6px; background: var(--c-fcfcfb); padding: 8px 10px;
  width: ${W + 22}px; max-width: 100%; box-sizing: border-box; }   /* svg + padding + border = the 760px prose column */
.ld .top { display: flex; flex-wrap: wrap; align-items: stretch; gap: 8px 10px; padding-bottom: 8px; }
${knobCss('.ld .top')}
.ld .stp button { white-space: nowrap; }
.ld .pre { margin-left: auto; }
.ld input[type=range] { width: 156px; margin: 0; accent-color: var(--c-52514e); }
.ld .cv { font: 11px ui-monospace, Menlo, monospace; min-width: 10ch; }
.ld .fix { font-size: 11px; color: var(--c-52514e); white-space: nowrap; }
.ld svg { display: block; touch-action: none; }
.ld .dims { font: 9px system-ui; fill: var(--c-898781); }
.ld .lg { font: 11px system-ui; fill: var(--c-52514e); }
.ld .sc { font: 10.5px system-ui; fill: var(--c-898781); }
.ld .lgv { font: 11px ui-monospace, Menlo, monospace; fill: var(--c-0b0b0b); }
`;

class Dsv3Ladder extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    const st = this.id ? readState('l:' + this.id) : null;
    this.cfg = { c: LOG_C_DSV3, seq: +(this.getAttribute('seq') ?? 4096), y: 'share' };
    this._def = JSON.stringify(this.cfg);
    if (st?.c != null) this.cfg.c = +st.c;
    if (Number.isInteger(st?.seq) && st.seq >= SEQS[0] && st.seq <= SEQS.at(-1)) this.cfg.seq = st.seq;
    if (YS.includes(st?.y)) this.cfg.y = st.y;
    this._w = oneHot(this.cfg.y);                   // each view's opacity; a y flip tweens between
    const style = document.createElement('style'); style.textContent = CSS;
    this._root = el('div', 'ld');
    this._top = el('div', 'top');
    this._chart = el('div');
    this._root.append(this._top, this._chart);
    this.append(style, this._root);
    this._buildKnobs();
    // drag on a panel sets its axis: C (continuous) or S (snaps to the powers of two)
    // (handlers on the persistent container: the svg re-renders under the pointer)
    const xOf = (e) => { const r = this._chart.firstChild.getBoundingClientRect(); return (e.clientX - r.left) * W / r.width; };
    const atC = (e) => { const v = lcOf(xOf(e)); this._set({ c: Math.abs(v - LOG_C_DSV3) < SNAP ? LOG_C_DSV3 : +v.toFixed(2) }, false); };
    const atS = (e) => this._set({ seq: seqOf(xOf(e)) }, false);
    this._chart.onpointerdown = (e) => { const k = e.target.dataset?.hit; if (!k) return;
      const at = k === 's' ? atS : atC;
      this._chart.setPointerCapture(e.pointerId); at(e); this._chart.onpointermove = at; };
    this._chart.onpointerup = this._chart.onpointercancel = () => { this._chart.onpointermove = null; };
    this.render();
    addEventListener('dsv3-theme', () => this.render());
  }
  _buildKnobs() {
    const grp = (label) => { const g = el('span', 'pargrp'); const l = el('div', 'parlab'); l.textContent = label; g.append(l); return g; };
    const row = (...kids) => { const r = el('div', 'parrow'); r.append(...kids); return r; };
    // compute: a log slider (snaps onto DSv3)
    const r = this._range = document.createElement('input');
    Object.assign(r, { type: 'range', min: LO, max: HI, step: 0.01 });
    r.dataset.knob = 'c';
    r.oninput = () => { let v = +r.value; if (Math.abs(v - LOG_C_DSV3) < SNAP) v = LOG_C_DSV3; this._set({ c: v }, false); };
    this._cv = el('span', 'cv');
    const g1 = grp('compute budget C (FLOP)'); g1.append(row(r, this._cv));
    // the DSv3 preset, top right: its rung and its 4K pretraining S
    const pre = el('span', 'stp'); pre.dataset.knob = 'preset';
    this._pre = document.createElement('button'); this._pre.type = 'button'; this._pre.textContent = 'DeepSeek-V3';
    this._pre.onclick = () => this._set({ c: LOG_C_DSV3, seq: 4096 }, true);
    pre.append(this._pre);
    const g5 = grp('preset'); g5.classList.add('pre'); g5.append(row(pre));
    // sequence length stepper: the powers of two (an S-panel drag lands between them; ± steps to the next one)
    const eg = el('span', 'stp'); eg.dataset.knob = 'seq';
    const sel = this._sel = document.createElement('select'); sel.className = 'v';
    for (const o of SEQS) sel.append(new Option(o.toLocaleString('en-US'), o));
    sel.onchange = () => this._set({ seq: +sel.value }, true);
    const b = (t, di) => { const x = document.createElement('button'); x.type = 'button'; x.textContent = t; x.dataset.dir = di;
      x.onclick = () => { const v = this._step(di); if (v) this._set({ seq: v }, true); }; return x; };
    eg.append(b('−', -1), sel, b('+', +1));
    const g2 = grp('sequence length S'); g2.append(row(eg));
    const fix = el('span', 'fix');
    fix.textContent = `D/N = ${TPP.toFixed(0)} · d/L = ${(DSV3.hidden / DSV3.layers).toFixed(1)} · top-${DSV3.topk} of ${DSV3.routedExperts} · V = ${DSV3.vocab.toLocaleString('en-US')}`;
    const g3 = grp('held fixed'); g3.append(row(fix));
    // y axis: 100%-stacked shares, each class's FLOP/token as a line on a log axis, or stacked FLOP/token on a linear one
    const yb = el('span', 'stp'); yb.dataset.knob = 'y';
    for (const [v, t] of [['share', '%'], ['log', 'log'], ['lin', 'linear']]) {
      const x = document.createElement('button'); x.type = 'button'; x.textContent = t; x.dataset.y = v;
      x.onclick = () => this._set({ y: v }, true); yb.append(x);
    }
    const g4 = grp('y axis'); g4.append(row(yb));
    this._top.append(g1, g2, g4, g5, g3);
    this._sync();
  }
  _sync() {
    this._range.value = this.cfg.c;
    this._cv.textContent = fmtC(this.cfg.c);
    this._pre.classList.toggle('on', this.cfg.c === LOG_C_DSV3 && this.cfg.seq === 4096);
    // an off-grid S (from a drag) shows as a transient option in its sorted place
    this._sel.querySelector('option[data-off]')?.remove();
    if (!SEQS.includes(this.cfg.seq)) {
      const o = new Option(this.cfg.seq.toLocaleString('en-US'), this.cfg.seq); o.dataset.off = '';
      this._sel.insertBefore(o, this._sel.options[SEQS.findIndex((v) => v > this.cfg.seq)]);
    }
    this._sel.value = this.cfg.seq;
    for (const x of this._top.querySelectorAll('[data-knob="y"] button')) x.classList.toggle('on', x.dataset.y === this.cfg.y);
    for (const x of this._top.querySelectorAll('[data-knob="seq"] button')) x.disabled = !this._step(+x.dataset.dir);
  }
  _step(di) { return di > 0 ? SEQS.find((v) => v > this.cfg.seq) : SEQS.findLast((v) => v < this.cfg.seq); }
  _set(patch, animate) {
    const next = { ...this.cfg, ...patch };
    if (next.c === this.cfg.c && next.seq === this.cfg.seq && next.y === this.cfg.y) return;
    const flip = next.y !== this.cfg.y;
    this.cfg = next;
    this._sync();
    if (this.id) writeState('l:' + this.id, this.cfg, this._def);
    if (flip) return this._animateMix(oneHot(next.y));
    this._w = oneHot(next.y);                       // a C/S change lands any crossfade in flight
    if (animate) this._animateTo(this._layout()); else { this._gen = (this._gen ?? 0) + 1; this.render(); }
  }
  // both panels' series, one shared linear ceiling, the cursor in each
  _layout() {
    const P = ladderPoint(this.cfg.c, this.cfg.seq), Ns = [];
    const c = series((i) => ladderPoint(LO + (HI - LO) * i / (K - 1), this.cfg.seq));
    const s = series((i) => { const f = classFlops(P.a, 2 ** (SLO + (SHI - SLO) * i / (K - 1)));   // the cursor's rung: only the attention core moves
      return { f, total: Object.values(f).reduce((x, y) => x + y, 0), sixN: P.sixN }; });
    for (let e = LO; e <= HI; e++) Ns.push(ladderPoint(e, 4096).N);
    return { c, s, top: HEAD * Math.max(c.mx, s.mx), Ns, cx: px(this.cfg.c), sx: pxs(Math.log2(this.cfg.seq)), P };
  }
  // y-axis flip: crossfade the two views over the same frames as the S tween
  _animateMix(to) {
    const from = this._w, N = 12; let f = 0;
    const gen = this._gen = (this._gen ?? 0) + 1;
    this._L = this._layout();
    const step = () => {
      if (this._gen !== gen) return;
      f++; const t = ease(Math.min(1, f / N));
      this._w = Object.fromEntries(YS.map((k) => [k, lerp(from[k], to[k], t)]));
      this._draw(this._L);
      if (f < N) setTimeout(step, 16);
    };
    setTimeout(step, 16);
  }
  _animateTo(B) {
    const A = this._L, N = 12; let f = 0;
    const gen = this._gen = (this._gen ?? 0) + 1;
    const step = () => {
      if (this._gen !== gen) return;
      f++; const t = ease(Math.min(1, f / N));
      this._draw(blend(A, B, t));
      if (f < N) setTimeout(step, 16); else { this._L = B; this._draw(B); }
    };
    setTimeout(step, 16);
  }
  render() { this._L = this._layout(); this._draw(this._L); }
  _draw(L) {
    const f1 = (v) => v.toFixed(1), B = [], dims = 'class="dims"';
    const xs = [X0, XS].map((x0) => Array.from({ length: K }, (_, i) => f1(x0 + PW * i / (K - 1))));
    const panels = [['c', L.c, xs[0]], ['s', L.s, xs[1]]];
    const w = this._w, ab = w.log + w.lin, op = (o) => `opacity="${+o.toFixed(3)}"`;
    const grid = (y, label, o = 1) => {             // a gridline across both panels, labelled once at the left
      const a = o < 1 ? ` ${op(o)}` : '';
      for (const x0 of [X0, XS]) B.push(`<line x1="${x0}" y1="${f1(y)}" x2="${x0 + PW}" y2="${f1(y)}" stroke="${C('#e1e0d9')}"${a}/>`);
      B.push(`<text ${dims} x="${X0 - 5}" y="${f1(y + 3)}" text-anchor="end"${a}>${label}</text>`);
    };
    const stack = (attr, Y, cum) => panels.forEach(([id, S, xx]) => {
      B.push(`<g data-x="${id}">`);
      CLASSES.forEach((k, j) => {
        const top = xx.map((x, i) => `${x},${f1(Y(S[cum][j][i]))}`);
        const bot = xx.map((x, i) => `${x},${f1(Y(j ? S[cum][j - 1][i] : 0))}`).reverse();
        B.push(`<polygon ${attr}="${k.id}" points="${top.join(' ')} ${bot.join(' ')}" fill="${C(k.c)}"/>`);
      });
      B.push('</g>');
    });
    const dashed = (attr, Y, key) => panels.forEach(([id, S, xx]) =>
      B.push(`<polyline ${attr} data-x="${id}" points="${xx.map((x, i) => `${x},${f1(Y(S[key][i]))}`).join(' ')}" fill="none" stroke="${C('#0b0b0b')}" stroke-width="1.25" stroke-dasharray="5 3"/>`));
    // share view: % grid, 100%-stacked bands, the share 6ND covers
    if (w.share > 0) {
      B.push(`<g data-view="share" ${op(w.share)}>`);
      for (const u of [0, 0.25, 0.5, 0.75, 1]) grid(py(u), `${u * 100}%`);
      stack('data-band', py, 'cum');
      dashed('data-six', (v) => py(Math.min(1, v)), 'six');
      B.push('</g>');
    }
    // log view: decade grid, each class's FLOP/token as a line, 6N dashed
    if (w.log > 0) {
      B.push(`<g data-view="log" ${op(w.log)}>`);
      for (let e = YLO; e <= YHI; e++) grid(pyl(e), `10${sup(e)}`);
      B.push(`<text ${dims} x="${X0}" y="${Y0 - 7}">FLOP per token, log scale</text>`);
      panels.forEach(([id, S, xx]) => CLASSES.forEach((k, j) => B.push(`<polyline data-line="${k.id}" data-x="${id}" points="${xx.map((x, i) => `${x},${f1(pyl(S.lf[j][i]))}`).join(' ')}" fill="none" stroke="${C(k.c)}" stroke-width="2"/>`)));
      dashed('data-six-abs', pyl, 'l6');
      B.push('</g>');
    }
    // linear view: stacked FLOP/token under a continuous ceiling over both panels' ranges, 6N dashed
    if (w.lin > 0) {
      const pa = (v) => Y0 + (1 - v / L.top) * PH;
      B.push(`<g data-view="lin" data-top="${L.top}" ${op(w.lin)}>`);
      grid(pa(0), 0);
      for (const [v, o] of linTicks(L.top)) if (o > 0.005) grid(pa(v), fmtF(v), o);
      B.push(`<text ${dims} x="${X0}" y="${Y0 - 7}">FLOP per token, linear scale</text>`);
      stack('data-area', pa, 'ca');
      dashed('data-six-lin', pa, 'a6');
      B.push('</g>');
    }
    for (let e = LO; e <= HI; e++) {
      const x = f1(px(e));
      B.push(`<line x1="${x}" y1="${AX}" x2="${x}" y2="${AX + 3}" stroke="${C('#898781')}"/>`);
      B.push(`<text ${dims} x="${x}" y="${AX + 13}" text-anchor="middle">10${sup(e)}</text>`);
      B.push(`<text ${dims} x="${x}" y="${AX + 25}" text-anchor="middle">${fmtN(L.Ns[e - LO])}</text>`);
    }
    for (const s of SEQS) {
      const x = f1(pxs(Math.log2(s)));
      B.push(`<line x1="${x}" y1="${AX}" x2="${x}" y2="${AX + 3}" stroke="${C('#898781')}"/>`);
      B.push(`<text ${dims} x="${x}" y="${AX + 13}" text-anchor="middle">${fmtS(s)}</text>`);
    }
    // each panel's caption names its axis and the slice it holds fixed
    B.push(`<text ${dims} x="${X0 + PW / 2}" y="${AX + 40}" text-anchor="middle">compute budget C, FLOP (N, params) · at S = ${this.cfg.seq.toLocaleString('en-US')}</text>`);
    B.push(`<text ${dims} x="${XS + PW / 2}" y="${AX + 40}" text-anchor="middle">sequence length S, tokens · at C = ${fmtC(this.cfg.c)}</text>`);
    // DSv3 markers (its rung; its pretraining S) + the cursor in each panel
    for (const [x, t] of [[px(LOG_C_DSV3), 'DeepSeek-V3'], [pxs(12), 'DeepSeek-V3 pretraining']]) {
      B.push(`<line x1="${f1(x)}" y1="${Y0 - 4}" x2="${f1(x)}" y2="${AX}" stroke="${C('#0b0b0b')}" stroke-width="0.75" stroke-dasharray="2 2"/>`);
      B.push(`<text ${dims} x="${f1(x)}" y="${Y0 - 7}" text-anchor="middle">${t}</text>`);
    }
    for (const [k, x, v] of [['cursor', L.cx, this.cfg.c], ['cursor-s', L.sx, this.cfg.seq]]) {
      B.push(`<line data-${k}="${v}" x1="${f1(x)}" y1="${Y0}" x2="${f1(x)}" y2="${AX}" stroke="${C('#0b0b0b')}" stroke-width="1.5"/>`);
      B.push(`<circle cx="${f1(x)}" cy="${AX}" r="3.5" fill="${C('#0b0b0b')}"/>`);
    }
    for (const [k, x0] of [['c', X0], ['s', XS]])
      B.push(`<rect data-hit="${k}" x="${x0}" y="${Y0}" width="${PW}" height="${PH + 6}" fill="transparent" style="cursor:ew-resize"/>`);
    // legend = the cursor's breakdown, below the plot: the bands' top-to-bottom order down column 1 (outside N, then 6ND), then column 2 (in N)
    const P = L.P, rows = [...CLASSES].reverse().map((k) => ({ k, v: P.f[k.id] / P.total, f: P.f[k.id] }));
    rows.splice(2, 0, { six: true, v: P.sixN / P.total, f: P.sixN });
    if (w.share > 0) B.push(`<text ${dims} ${op(w.share)} x="${X0}" y="${LY}">share of forward + backward FLOPs at the cursor</text>`);
    if (ab > 0) B.push(`<text ${dims} ${op(ab)} x="${X0}" y="${LY}">forward + backward FLOPs per token at the cursor</text>`);
    rows.forEach((r, i) => {
      const x = X0 + (i / 3 | 0) * LW, y = LY + 18 + (i % 3) * 20, xr = x + LW - 36;
      if (r.six) B.push(`<line x1="${x}" y1="${y - 4}" x2="${x + 10}" y2="${y - 4}" stroke="${C('#0b0b0b')}" stroke-width="1.25" stroke-dasharray="3 2"/>` +
        `<text class="lg" x="${x + 16}" y="${y}">6ND counts</text>`);
      else B.push(`<rect x="${x}" y="${y - 9}" width="10" height="10" fill="${C(r.k.c)}"/>` +
        `<text class="lg" x="${x + 16}" y="${y}">${r.k.label}</text><text class="sc" x="${x + 128}" y="${y}"><tspan font-size="13">∝</tspan> ${r.k.scale}</text>`);
      if (w.share > 0) B.push(`<text class="lgv" ${op(w.share)} data-share="${r.six ? 'six' : r.k.id}" data-true="${r.v}" x="${xr}" y="${y}" text-anchor="end">${fmtP(r.v)}</text>`);
      if (ab > 0) B.push(`<text class="lgv" ${op(ab)} data-abs="${r.six ? 'six' : r.k.id}" data-true="${r.f}" x="${xr}" y="${y}" text-anchor="end">${fmtF(r.f)}</text>`);
    });
    this._chart.innerHTML = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">${B.join('')}</svg>`;
  }
}
// one panel's curves from K sample points: cumulative share boundaries per class, the 6N share,
// log10 and cumulative FLOP/token, and the tallest total (for the linear ceiling)
function series(pt) {
  const cum = CLASSES.map(() => new Float64Array(K)), six = new Float64Array(K);
  const lf = CLASSES.map(() => new Float64Array(K)), l6 = new Float64Array(K);   // log10 FLOP/token
  const ca = CLASSES.map(() => new Float64Array(K)), a6 = new Float64Array(K);   // cumulative FLOP/token
  let mx = 0;
  for (let i = 0; i < K; i++) {
    const p = pt(i);
    let u = 0, a = 0;
    CLASSES.forEach((k, j) => { u += p.f[k.id] / p.total; cum[j][i] = u; lf[j][i] = Math.log10(p.f[k.id]); ca[j][i] = a += p.f[k.id]; });
    six[i] = p.sixN / p.total; l6[i] = Math.log10(p.sixN); a6[i] = p.sixN; mx = Math.max(mx, p.total);
  }
  return { cum, six, lf, l6, ca, a6, mx };
}
function blendSeries(A, B, t) {
  const m = (k) => B[k].map((c, j) => c.map((v, i) => lerp(A[k][j][i], v, t))), v = (k) => B[k].map((x, i) => lerp(A[k][i], x, t));
  return { ...B, cum: m('cum'), lf: m('lf'), ca: m('ca'), six: v('six'), l6: v('l6'), a6: v('a6') };
}
function blend(A, B, t) {
  if (!A || t >= 1) return B;
  return { ...B, c: blendSeries(A.c, B.c, t), s: blendSeries(A.s, B.s, t),
    cx: lerp(A.cx, B.cx, t), sx: lerp(A.sx, B.sx, t), top: lerp(A.top, B.top, t) };
}
function el(tag, cls) { const e = document.createElement(tag); if (cls) e.className = cls; return e; }
function readState(key) {
  try { const v = new URLSearchParams(location.hash.slice(1)).get(key); return v ? JSON.parse(v) : null; } catch { return null; }
}
function writeState(key, obj, def) {
  const p = new URLSearchParams(location.hash.slice(1)), s = JSON.stringify(obj);
  if (s === def) p.delete(key); else p.set(key, s);   // the hash carries only departures from the defaults
  history.replaceState(null, '', p.size ? '#' + p : location.pathname + location.search);
}
if (typeof customElements !== 'undefined' && !customElements.get('dsv3-ladder')) customElements.define('dsv3-ladder', Dsv3Ladder);
