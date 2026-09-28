// The scaling ladder: DeepSeek-V3 as one rung of a family, every rung spending
// its compute budget C the way DSv3 does. Held fixed: tokens per activated
// parameter (14.8T ÷ 36.6B), aspect ratio (layers ∝ width), the attention:FFN
// split (every width dim — hidden, heads, latent ranks, FFN widths — scales
// by one factor s; per-head dims don't), the MoE shape (256 experts, top-8,
// 1 shared) and the vocabulary. Dims scale CONTINUOUSLY (61·s layers, no
// rounding), so DSv3 is exactly s = 1. <dsv3-ladder> draws each op class's
// share of the counted FLOPs across C, against the share 6ND covers.
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
const LO = 19, HI = 27, K = 161;                    // x domain (log10 FLOP) and samples
const SEQS = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
const W = 738, X0 = 60, PW = 570, Y0 = 18, PH = 200, AX = Y0 + PH;   // W: the prose column, so margin notes sit beside it
const LY = AX + 48, LW = (PW + 30) / 2, H = LY + 3 * 20 + 6;           // legend below the plot, spanning it: 2 columns × 3 rows
const px = (lc) => X0 + (lc - LO) / (HI - LO) * PW;
const lcOf = (x) => Math.max(LO, Math.min(HI, LO + (x - X0) / PW * (HI - LO)));
const py = (u) => Y0 + (1 - u) * PH;
const YLO = 6, YHI = 13;                            // log mode: y domain (log10 FLOP/token)
const pyl = (lf) => Y0 + (1 - (Math.max(YLO, lf) - YLO) / (YHI - YLO)) * PH;
const YS = ['share', 'log', 'lin'];                 // y-axis modes
const oneHot = (y) => Object.fromEntries(YS.map((k) => [k, +(k === y)]));
const niceTop = (v) => { const e = 10 ** Math.floor(Math.log10(v)); const m = [1, 2, 2.5, 5, 10].find((x) => x * e >= v); return { top: m * e, nt: m === 2 ? 4 : 5 }; };
const SUP = '⁰¹²³⁴⁵⁶⁷⁸⁹';
const sup = (n) => String(n).split('').map((d) => SUP[d]).join('');
const fmtC = (lc) => { const e = Math.floor(lc + 1e-9); return `${(10 ** (lc - e)).toFixed(2)}×10${sup(e)}`; };
const fmtN = (n) => n >= 1e12 ? +(n / 1e12).toPrecision(3) + 'T' : n >= 1e9 ? +(n / 1e9).toPrecision(3) + 'B' : +(n / 1e6).toPrecision(3) + 'M';
const fmtF = (f) => f >= 1e12 ? `${+(f / 1e12).toPrecision(3)} TFLOP` : f >= 1e9 ? `${+(f / 1e9).toPrecision(3)} GFLOP` : `${+(f / 1e6).toPrecision(3)} MFLOP`;
const fmtP = (u) => (u * 100).toFixed(u < 0.001 && u > 0 ? 2 : 1) + '%';
const lerp = (a, b, t) => a + (b - a) * t;
const ease = (p) => 1 - (1 - p) ** 3;
const SNAP = 0.03;                                  // decades: the slider snaps onto DSv3

const CSS = `
dsv3-ladder { display: block; margin: 14px 0 26px; }
.ld { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-0b0b0b);
  border: 1px solid var(--c-e1e0d9); border-radius: 6px; background: var(--c-fcfcfb); padding: 8px 10px;
  width: ${W + 22}px; max-width: 100%; box-sizing: border-box; }   /* svg + padding + border = the 760px prose column */
.ld .top { display: flex; flex-wrap: wrap; align-items: stretch; gap: 8px 10px; padding-bottom: 8px; }
${knobCss('.ld .top')}
.ld .stp button { white-space: nowrap; }
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
    this.cfg = { c: LOG_C_DSV3, seq: +(this.getAttribute('seq') ?? 4096) };
    if (st?.c != null) this.cfg.c = +st.c;
    if (st?.seq != null && SEQS.includes(+st.seq)) this.cfg.seq = +st.seq;
    this.cfg.y = YS.includes(st?.y) ? st.y : 'share';
    this._w = oneHot(this.cfg.y);                   // each view's opacity; a y flip tweens between
    const style = document.createElement('style'); style.textContent = CSS;
    this._root = el('div', 'ld');
    this._top = el('div', 'top');
    this._chart = el('div');
    this._root.append(this._top, this._chart);
    this.append(style, this._root);
    this._buildKnobs();
    // drag on the plot sets C (handlers on the persistent container: the svg re-renders under the pointer)
    const at = (e) => { const r = this._chart.firstChild.getBoundingClientRect(), v = lcOf((e.clientX - r.left) * W / r.width);
      this._set({ c: Math.abs(v - LOG_C_DSV3) < SNAP ? LOG_C_DSV3 : +v.toFixed(2) }, false); };
    this._chart.onpointerdown = (e) => { if (!e.target.matches('rect[data-hit]')) return;
      this._chart.setPointerCapture(e.pointerId); at(e); this._chart.onpointermove = at; };
    this._chart.onpointerup = this._chart.onpointercancel = () => { this._chart.onpointermove = null; };
    this.render();
    addEventListener('dsv3-theme', () => this.render());
  }
  _buildKnobs() {
    const grp = (label) => { const g = el('span', 'pargrp'); const l = el('div', 'parlab'); l.textContent = label; g.append(l); return g; };
    const row = (...kids) => { const r = el('div', 'parrow'); r.append(...kids); return r; };
    // compute: a log slider (snaps onto DSv3) + the DSv3 preset
    const r = this._range = document.createElement('input');
    Object.assign(r, { type: 'range', min: LO, max: HI, step: 0.01 });
    r.dataset.knob = 'c';
    r.oninput = () => { let v = +r.value; if (Math.abs(v - LOG_C_DSV3) < SNAP) v = LOG_C_DSV3; this._set({ c: v }, false); };
    this._cv = el('span', 'cv');
    const pre = el('span', 'stp'); pre.dataset.knob = 'preset';
    this._pre = document.createElement('button'); this._pre.type = 'button'; this._pre.textContent = 'DeepSeek-V3';
    this._pre.onclick = () => this._set({ c: LOG_C_DSV3 }, true);
    pre.append(this._pre);
    const g1 = grp('compute budget C (FLOP)'); g1.append(row(r, this._cv, pre));
    // sequence length stepper
    const eg = el('span', 'stp'); eg.dataset.knob = 'seq';
    const sel = this._sel = document.createElement('select'); sel.className = 'v';
    for (const o of SEQS) sel.append(new Option(o.toLocaleString('en-US'), o));
    sel.onchange = () => this._set({ seq: +sel.value }, true);
    const b = (t, di) => { const x = document.createElement('button'); x.type = 'button'; x.textContent = t; x.dataset.dir = di;
      x.onclick = () => { const j = SEQS.indexOf(this.cfg.seq) + di; if (SEQS[j]) this._set({ seq: SEQS[j] }, true); }; return x; };
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
    this._top.append(g1, g2, g4, g3);
    this._sync();
  }
  _sync() {
    this._range.value = this.cfg.c;
    this._cv.textContent = fmtC(this.cfg.c);
    this._pre.classList.toggle('on', this.cfg.c === LOG_C_DSV3);
    this._sel.value = this.cfg.seq;
    for (const x of this._top.querySelectorAll('[data-knob="y"] button')) x.classList.toggle('on', x.dataset.y === this.cfg.y);
    for (const x of this._top.querySelectorAll('[data-knob="seq"] button')) x.disabled = !SEQS[SEQS.indexOf(this.cfg.seq) + +x.dataset.dir];
  }
  _set(patch, animate) {
    const next = { ...this.cfg, ...patch };
    if (next.c === this.cfg.c && next.seq === this.cfg.seq && next.y === this.cfg.y) return;
    const flip = next.y !== this.cfg.y;
    this.cfg = next;
    this._sync();
    if (this.id) writeState('l:' + this.id, this.cfg);
    if (flip) return this._animateMix(oneHot(next.y));
    this._w = oneHot(next.y);                       // a C/S change lands any crossfade in flight
    if (animate) this._animateTo(this._layout()); else { this._gen = (this._gen ?? 0) + 1; this.render(); }
  }
  // cumulative share boundaries per class at every sample, the 6N share, the cursor
  _layout() {
    const cum = CLASSES.map(() => new Float64Array(K)), six = new Float64Array(K), Ns = [];
    const lf = CLASSES.map(() => new Float64Array(K)), l6 = new Float64Array(K);   // log10 FLOP/token
    const ca = CLASSES.map(() => new Float64Array(K)), a6 = new Float64Array(K);   // cumulative FLOP/token
    let mx = 0;
    for (let i = 0; i < K; i++) {
      const p = ladderPoint(LO + (HI - LO) * i / (K - 1), this.cfg.seq);
      let u = 0, a = 0;
      CLASSES.forEach((k, j) => { u += p.f[k.id] / p.total; cum[j][i] = u; lf[j][i] = Math.log10(p.f[k.id]); ca[j][i] = a += p.f[k.id]; });
      six[i] = p.sixN / p.total; l6[i] = Math.log10(p.sixN); a6[i] = p.sixN; mx = Math.max(mx, p.total);
    }
    for (let e = LO; e <= HI; e++) Ns.push(ladderPoint(e, 4096).N);
    const { top, nt } = niceTop(mx);
    return { cum, six, lf, l6, ca, a6, top, nt, Ns, cx: px(this.cfg.c), P: ladderPoint(this.cfg.c, this.cfg.seq) };
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
    const xs = Array.from({ length: K }, (_, i) => f1(X0 + PW * i / (K - 1)));
    const w = this._w, ab = w.log + w.lin, op = (o) => `opacity="${+o.toFixed(3)}"`;
    // share view: % grid, 100%-stacked bands, the share 6ND covers
    if (w.share > 0) {
      B.push(`<g data-view="share" ${op(w.share)}>`);
      for (const u of [0, 0.25, 0.5, 0.75, 1]) {
        B.push(`<line x1="${X0}" y1="${f1(py(u))}" x2="${X0 + PW}" y2="${f1(py(u))}" stroke="${C('#e1e0d9')}"/>`);
        B.push(`<text ${dims} x="${X0 - 5}" y="${f1(py(u) + 3)}" text-anchor="end">${u * 100}%</text>`);
      }
      CLASSES.forEach((k, j) => {
        const top = xs.map((x, i) => `${x},${f1(py(L.cum[j][i]))}`);
        const bot = xs.map((x, i) => `${x},${f1(py(j ? L.cum[j - 1][i] : 0))}`).reverse();
        B.push(`<polygon data-band="${k.id}" points="${top.join(' ')} ${bot.join(' ')}" fill="${C(k.c)}"/>`);
      });
      B.push(`<polyline data-six points="${xs.map((x, i) => `${x},${f1(py(Math.min(1, L.six[i])))}`).join(' ')}" fill="none" stroke="${C('#0b0b0b')}" stroke-width="1.25" stroke-dasharray="5 3"/>`);
      B.push('</g>');
    }
    // log view: decade grid, each class's FLOP/token as a line, 6N dashed
    if (w.log > 0) {
      B.push(`<g data-view="log" ${op(w.log)}>`);
      for (let e = YLO; e <= YHI; e++) {
        B.push(`<line x1="${X0}" y1="${f1(pyl(e))}" x2="${X0 + PW}" y2="${f1(pyl(e))}" stroke="${C('#e1e0d9')}"/>`);
        B.push(`<text ${dims} x="${X0 - 5}" y="${f1(pyl(e) + 3)}" text-anchor="end">10${sup(e)}</text>`);
      }
      B.push(`<text ${dims} x="${X0}" y="${Y0 - 7}">FLOP per token, log scale</text>`);
      CLASSES.forEach((k, j) => B.push(`<polyline data-line="${k.id}" points="${xs.map((x, i) => `${x},${f1(pyl(L.lf[j][i]))}`).join(' ')}" fill="none" stroke="${C(k.c)}" stroke-width="2"/>`));
      B.push(`<polyline data-six-abs points="${xs.map((x, i) => `${x},${f1(pyl(L.l6[i]))}`).join(' ')}" fill="none" stroke="${C('#0b0b0b')}" stroke-width="1.25" stroke-dasharray="5 3"/>`);
      B.push('</g>');
    }
    // linear view: stacked FLOP/token up to a round ceiling over the whole range, 6N dashed
    if (w.lin > 0) {
      const pa = (v) => Y0 + (1 - v / L.top) * PH;
      B.push(`<g data-view="lin" ${op(w.lin)}>`);
      for (let t = 0; t <= L.nt; t++) {
        const v = L.top * t / L.nt;
        B.push(`<line x1="${X0}" y1="${f1(pa(v))}" x2="${X0 + PW}" y2="${f1(pa(v))}" stroke="${C('#e1e0d9')}"/>`);
        B.push(`<text ${dims} x="${X0 - 5}" y="${f1(pa(v) + 3)}" text-anchor="end">${t ? fmtF(v) : 0}</text>`);
      }
      B.push(`<text ${dims} x="${X0}" y="${Y0 - 7}">FLOP per token, linear scale</text>`);
      CLASSES.forEach((k, j) => {
        const top = xs.map((x, i) => `${x},${f1(pa(L.ca[j][i]))}`);
        const bot = xs.map((x, i) => `${x},${f1(pa(j ? L.ca[j - 1][i] : 0))}`).reverse();
        B.push(`<polygon data-area="${k.id}" points="${top.join(' ')} ${bot.join(' ')}" fill="${C(k.c)}"/>`);
      });
      B.push(`<polyline data-six-lin points="${xs.map((x, i) => `${x},${f1(pa(L.a6[i]))}`).join(' ')}" fill="none" stroke="${C('#0b0b0b')}" stroke-width="1.25" stroke-dasharray="5 3"/>`);
      B.push('</g>');
    }
    for (let e = LO; e <= HI; e++) {
      const x = f1(px(e));
      B.push(`<line x1="${x}" y1="${AX}" x2="${x}" y2="${AX + 3}" stroke="${C('#898781')}"/>`);
      B.push(`<text ${dims} x="${x}" y="${AX + 13}" text-anchor="middle">10${sup(e)}</text>`);
      B.push(`<text ${dims} x="${x}" y="${AX + 25}" text-anchor="middle">${fmtN(L.Ns[e - LO])}</text>`);
    }
    B.push(`<text ${dims} x="${X0 + PW + 24}" y="${AX + 13}">C, FLOP</text><text ${dims} x="${X0 + PW + 24}" y="${AX + 25}">N, params</text>`);
    // DSv3 marker + cursor
    const dx = f1(px(LOG_C_DSV3));
    B.push(`<line x1="${dx}" y1="${Y0 - 4}" x2="${dx}" y2="${AX}" stroke="${C('#0b0b0b')}" stroke-width="0.75" stroke-dasharray="2 2"/>`);
    B.push(`<text ${dims} x="${dx}" y="${Y0 - 7}" text-anchor="middle">DeepSeek-V3</text>`);
    B.push(`<line data-cursor="${this.cfg.c}" x1="${f1(L.cx)}" y1="${Y0}" x2="${f1(L.cx)}" y2="${AX}" stroke="${C('#0b0b0b')}" stroke-width="1.5"/>`);
    B.push(`<circle cx="${f1(L.cx)}" cy="${AX}" r="3.5" fill="${C('#0b0b0b')}"/>`);
    B.push(`<rect data-hit x="${X0}" y="${Y0}" width="${PW}" height="${PH + 6}" fill="transparent" style="cursor:ew-resize"/>`);
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
function blend(A, B, t) {
  if (!A || t >= 1) return B;
  return { ...B, cum: B.cum.map((c, j) => c.map((v, i) => lerp(A.cum[j][i], v, t))),
    six: B.six.map((v, i) => lerp(A.six[i], v, t)), cx: lerp(A.cx, B.cx, t),
    lf: B.lf.map((c, j) => c.map((v, i) => lerp(A.lf[j][i], v, t))), l6: B.l6.map((v, i) => lerp(A.l6[i], v, t)),
    ca: B.ca.map((c, j) => c.map((v, i) => lerp(A.ca[j][i], v, t))), a6: B.a6.map((v, i) => lerp(A.a6[i], v, t)), top: lerp(A.top, B.top, t) };
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
if (typeof customElements !== 'undefined' && !customElements.get('dsv3-ladder')) customElements.define('dsv3-ladder', Dsv3Ladder);
