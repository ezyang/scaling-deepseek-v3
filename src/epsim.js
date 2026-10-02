// <dsv3-epsim>: route tokens through DeepSeek-V3's EP64 group one at a time
// and count the copies each one costs: one Infiniband copy per remote node
// hosting any of its 8 experts (deduped), landing on the GPU with our local
// index, then one NVLink copy per other GPU on that node that needs it. The
// running mean of IB copies is the sheet's K2 (studies/03-roofline.html).
// Two routings: the sheet's (exactly 4 of the 8 nodes, uniformly, the 8
// experts scattered uniformly over those nodes' 128, redrawn until every node
// holds one: IB → 3.5, NVLink → EXACT.nv) and DeepSeek-V3's node-limited
// top-8 (top-4 nodes by each node's two best affinities, then the top 8
// experts among them) with iid uniform scores, which can touch fewer than 4
// nodes and has no closed form: that's what the sim is for.
import { C } from './theme.js';
import { knobCss } from './ui.js';

export const NODES = 8, GPUS = 8, EPG = 4;         // EP group: nodes, GPUs per node, experts per GPU
const EPN = GPUS * EPG;                            // experts per node (32)
// the sheet routing's exact means: IB 4 × 7/8; NVLink 4 nodes × 7 non-landing GPUs × P(a given
// GPU holds one of the 8 | all 4 nodes hold one), by inclusion–exclusion
export const EXACT = { ib: 3.5, nv: 61477381 / 9417862 };
export const MAX = 100000, REST = 1000;   // REST: the default, where autoplay lands
const STEPS = [1, 10, 100, 1000, 10000];
const MODES = ['sheet', 'dsv3'];

// one generator per (token, routing): any prefix of the run replays from the hash
function rng(t, mode) {
  let s = Math.imul(t + 1, 0x9E3779B1) ^ (mode === 'dsv3' ? 0x85EBCA6B : 0);
  return () => {
    s = s + 0x6D2B79F5 | 0; let x = Math.imul(s ^ s >>> 15, 1 | s);
    x = x + Math.imul(x ^ x >>> 7, 61 | x) ^ x;
    return ((x ^ x >>> 14) >>> 0) / 4294967296;
  };
}
const PK = new Int32Array(NODES * EPN);   // k of 0 … n−1: a partial Fisher–Yates in scratch
const pick = (r, n, k) => { const a = PK; for (let i = 0; i < n; i++) a[i] = i; for (let i = 0; i < k; i++) { const j = i + Math.floor(r() * (n - i)), x = a[i]; a[i] = a[j]; a[j] = x; } return Array.from(a.subarray(0, k)); };

const SC = new Float64Array(NODES * EPN);   // DeepSeek-V3 routing's affinity scratch
// token t's 8 experts (0 … 255; expert e lives on node e >> 5, GPU (e >> 2) & 7)
export function route(t, mode = 'sheet') {
  const r = rng(t, mode);
  if (mode === 'sheet') {
    const nodes = pick(r, NODES, 4);
    for (;;) {
      const ex = pick(r, 4 * EPN, 8);
      if (new Set(ex.map((i) => i >> 5)).size === 4) return ex.map((i) => nodes[i >> 5] * EPN + (i & 31)).sort((a, b) => a - b);
    }
  }
  const s = SC, top2 = [];
  for (let i = 0; i < NODES * EPN; i++) s[i] = r();
  for (let n = 0; n < NODES; n++) {
    let a = 0, b = 0;
    for (let i = n * EPN; i < (n + 1) * EPN; i++) if (s[i] > a) { b = a; a = s[i]; } else if (s[i] > b) b = s[i];
    top2.push(a + b);
  }
  const top = [];   // the 8 best experts on the 4 best nodes, best first
  for (const n of [...top2.keys()].sort((a, b) => top2[b] - top2[a]).slice(0, 4)) {
    for (let e = n * EPN; e < (n + 1) * EPN; e++) {
      if (top.length === 8 && s[e] <= s[top[7]]) continue;
      let j = Math.min(top.length, 7);
      while (j > 0 && s[top[j - 1]] < s[e]) { top[j] = top[j - 1]; j--; }
      top[j] = e;
    }
  }
  return top.sort((a, b) => a - b);
}
// what it costs: per node, the GPUs holding its experts; IB copies to remote
// nodes; NVLink copies from the landing GPU (local index 0, like ours) to the rest
export function copies(experts) {
  const nodes = new Map();
  for (const e of experts) { const n = e >> 5; if (!nodes.has(n)) nodes.set(n, new Set()); nodes.get(n).add((e >> 2) & 7); }
  let ib = 0, nv = 0;
  for (const [n, g] of nodes) { if (n) ib++; nv += g.size - g.has(0); }
  return { nodes, ib, nv };
}
// the tallies after n tokens (goldens; the widget extends them incrementally)
export function simulate(n, mode = 'sheet') {
  const T = { n: 0, ib: 0, nv: 0, hist: [0, 0, 0, 0, 0], trace: [] };
  for (let t = 1; t <= n; t++) add(T, route(t, mode));
  return T;
}
function add(T, experts) {
  const c = copies(experts);
  T.n++; T.ib += c.ib; T.nv += c.nv; T.hist[c.ib]++;
  if (T.n % Math.max(1, 10 ** (Math.floor(Math.log10(T.n)) - 1)) === 0) T.trace.push([T.n, T.ib / T.n]);   // ~90 points a decade
  return experts;
}

// layout: the EP group on the left (a row per node, 8 GPU cells of 2 × 2
// experts; IB copies bend through the left gutter into the landing GPU,
// NVLink copies arc over the row), the tallies on the right
const W = 738, TOP = 40, P = 40, CX = 108, CP = 30, CW = 26, CH = 20;
const H = TOP + NODES * P + 18;
const cellX = (g) => CX + g * CP, rowY = (n) => TOP + n * P, TX = CX - 16;   // TX: the IB trunk
const RX = 404, HB = 166, HH = 62;                 // right panel: histogram baseline, 100% height
const MX0 = RX + 22, MX1 = W - 16, MY0 = 232, MY1 = 322, YLO = 2, YHI = 4.5;   // running-mean chart
const mx = (t) => MX0 + Math.log10(t) / 5 * (MX1 - MX0);
const my = (v) => MY1 - (Math.max(YLO, Math.min(YHI, v)) - YLO) / (YHI - YLO) * (MY1 - MY0);
const fmt = (n) => n.toLocaleString('en-US');
const lerp = (a, b, t) => a + (b - a) * t;
const ease = (p) => 1 - (1 - p) ** 3;
const mix = (x, y, w) => '#' + [1, 3, 5].map((i) => Math.round(lerp(parseInt(x.slice(i, i + 2), 16), parseInt(y.slice(i, i + 2), 16), w)).toString(16).padStart(2, '0')).join('');

const CSS = `
dsv3-epsim { display: block; margin: 14px 0 22px; }
.es { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-0b0b0b);
  border: 1px solid var(--c-e1e0d9); border-radius: 6px; background: var(--c-fcfcfb); padding: 8px 10px;
  width: ${W + 22}px; max-width: 100%; box-sizing: border-box; }
.es .top { display: flex; flex-wrap: wrap; align-items: stretch; gap: 8px 10px; padding-bottom: 6px; }
${knobCss('.es .top')}
.es .stp button { white-space: nowrap; }
.es svg { display: block; }
.es text { font: 11px system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--c-52514e); }
.es .dims { font-size: 9.5px; fill: var(--c-898781); }
.es .hd { font-weight: 600; fill: var(--c-1c1c1a); }
.es .num { font: 11px ui-monospace, Menlo, monospace; fill: var(--c-0b0b0b); }
.es g[data-seek] { cursor: pointer; }
.es g[data-seek]:hover .hit { fill: var(--c-f3f2ee); }
`;

class Dsv3Epsim extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    const st = this.id ? readState('e:' + this.id) : null;
    this.mode = MODES.includes(st?.m) ? st.m : 'sheet';
    // a fresh load (no state in the hash) samples from token 1 to the resting 1,000 on its first
    // full view; that resting state is the default, so the run leaves the hash (and the page's ↺) clean
    const still = typeof IntersectionObserver === 'undefined' || matchMedia('(prefers-reduced-motion: reduce)').matches;
    this._armed = !st && !still;
    this._replay(st ? Math.max(1, Math.min(MAX, Math.floor(+st.n || 1))) : this._armed ? 1 : REST);
    const style = document.createElement('style'); style.textContent = CSS;
    this._root = el('div', 'es');
    this._top = el('div', 'top');
    this._chart = el('div');
    this._chart.onclick = (e) => { const g = e.target.closest('[data-seek]'); if (g) this.seek(+g.dataset.seek); };
    this._root.append(this._top, this._chart);
    this.append(style, this._root);
    this._buildKnobs();
    this.render();
    addEventListener('dsv3-theme', () => this._auto || this.render());   // autoplay redraws every frame anyway
    if (this._armed)
      (this._io = new IntersectionObserver((es) => this._view(es.at(-1).intersectionRatio), { threshold: 0.99 })).observe(this);
  }
  _view(ratio) { if (ratio >= 0.99 && this._armed) this.autoplay(); }   // tests drive this (headless IO is frame-starved and late)
  _replay(n) {
    this.T = { n: 0, ib: 0, nv: 0, hist: [0, 0, 0, 0, 0], trace: [] };
    for (let t = 1; t <= n; t++) this.cur = add(this.T, route(t, this.mode));
  }
  _buildKnobs() {
    const grp = (label, knob, items) => {
      const g = el('span', 'pargrp'), l = el('div', 'parlab'), r = el('div', 'parrow'), s = el('span', 'stp');
      l.textContent = label; s.dataset.knob = knob;
      for (const [v, t, fn] of items) { const b = document.createElement('button'); b.type = 'button'; b.textContent = t; b.dataset.v = v; b.onclick = fn; s.append(b); }
      r.append(s); g.append(l, r); return g;
    };
    this._top.append(
      grp('route more tokens', 'step', [...STEPS.map((k) => [k, '+' + fmt(k), () => this.step(k)]), ['reset', 'reset', () => this._set(this.mode, 1)]]),
      grp('\u00a0', 'anim', [['redo', 'redo animation', () => this.redo()]]),
      grp('routing', 'mode', [['sheet', 'the sheet', () => this._set('sheet')], ['dsv3', 'DeepSeek-V3', () => this._set('dsv3')]]));
    this._top.querySelector('[data-v="sheet"]').title = 'exactly 4 of the 8 nodes, uniformly; the 8 experts uniform over their 128, each node holding at least one';
    this._top.querySelector('[data-v="dsv3"]').title = 'node-limited top-8: the top 4 of 8 nodes by the sum of each node\'s two best affinities, then the top 8 experts among those nodes; affinities iid uniform (the load-balancing bias ignored)';
    this._sync();
  }
  _sync() {
    for (const b of this._top.querySelectorAll('[data-knob="step"] button')) b.disabled = b.dataset.v === 'reset' ? this.T.n === 1 : this.T.n + +b.dataset.v > MAX;
    for (const b of this._top.querySelectorAll('[data-knob="mode"] button')) b.classList.toggle('on', b.dataset.v === this.mode);
    if (this.id && !this._armed && !this._auto) writeState('e:' + this.id, { n: this.T.n, m: this.mode }, `{"n":${REST},"m":"sheet"}`);
  }
  step(k) {
    if (this.T.n + k > MAX) return;
    const from = this._snap();
    this._msg = null;
    for (let i = 0; i < k; i++) this.cur = add(this.T, route(this.T.n + 1, this.mode));
    this._sync();
    this._animate(from);
  }
  // a routing flip replays the same token count under the other routing; reset goes back to one token
  _set(mode, n = this.T.n) {
    if (mode === this.mode && n === this.T.n) return;
    const from = this._snap();
    this._msg = null;
    this.mode = mode;
    this._replay(n);
    this._sync();
    this._animate(from);
  }
  // a histogram click: keep routing tokens until one costs k IB copies (or the cap), and say how many it took
  seek(k) {
    const cp = k === 1 ? 'copy' : 'copies';
    if (this.mode === 'sheet' && k < 3) this._msg = `impossible here: the sheet's tokens touch exactly 4 nodes`;
    else if (this.T.n === MAX) this._msg = 'the sim stops at 100K tokens: reset to sample more';
    else {
      const from = this._snap();
      let i = 0, hit = false;
      while (!hit && this.T.n < MAX) { this.cur = add(this.T, route(this.T.n + 1, this.mode)); i++; hit = copies(this.cur).ib === k; }
      this._msg = hit ? (i === 1 ? `the next token had ${k} IB ${cp}` : `sampled ${fmt(i)} tokens until one had ${k} IB ${cp}`)
        : `no ${k} in the next ${fmt(i)} tokens: the sim stops at 100K`;
      this._sync();
      this._animate(from);
      return;
    }
    if (this._auto) { this._snap(); this._sync(); }
    this.render();
  }
  // ~3 s from token 1 to 1,000, n = 1000^(f/N) so the dot glides a decade a second along the log
  // axis. Token 1 stays drawn throughout as the example routing (crossfading in first if the run
  // started from another token, as on redo) and crossfades to token 1,000 at the end: nothing
  // token-specific flickers mid-run. The bars ease toward their shares (a fifth of the way a frame)
  // so the early tokens' big swings (100% → 50% → 33% …) slide instead of jumping. Any knob interrupts it.
  autoplay(s = this._snap()) {
    const N = 180, FADE = 12, gen = this._gen = (this._gen ?? 0) + 1;
    const first = this.cur, A = this._auto = { n: 1, bars: s.share, op: 1 };   // the token drawn, its ink's opacity
    const from = s.cur.join() === first.join() ? null : { cur: s.cur, a: s.a };
    let f = 0, last = null;
    this._msg = null;
    const tick = () => {
      if (this._gen !== gen) return;
      f++;
      while (this.T.n < Math.min(REST, Math.round(REST ** (f / N)))) last = add(this.T, route(this.T.n + 1, this.mode));
      if (f === N) { this.cur = last; A.n = this.T.n; }
      const fd = f < N ? (from && f < FADE ? from : null) : f < N + FADE ? { cur: first } : null;   // the crossfade underway
      A.op = fd ? ease((f < N ? f : f - N) / FADE) : 1;
      let gap = 0;
      this.T.hist.forEach((h, k) => { A.bars[k] += (h / this.T.n - A.bars[k]) * 0.2; gap = Math.max(gap, Math.abs(h / this.T.n - A.bars[k])); });
      if (f >= N + FADE && gap < 5e-4) { this._auto = null; this._sync(); }   // settled: the last frame draws the exact shares
      this._draw(fd, A.op, this._auto && A.bars);
      if (this._auto) setTimeout(tick, 16);
    };
    setTimeout(tick, 16);
  }
  redo() { const s = this._snap(); this._replay(1); this.autoplay(s); }
  // what's drawn now, for a tween to start from; it stops an autoplay (and disarms the first-view one)
  _snap() {
    const A = this._auto, s = { cur: this.cur, n: A?.n ?? this.T.n, a: A?.op ?? 1, share: A?.bars.slice() ?? this.T.hist.map((h) => h / this.T.n) };
    this._auto = null; this._armed = false;
    return s;
  }
  // ~200 ms: the old token's copies fade out as the new one's fade in, the bars slide
  _animate(from) {
    this._auto = null;
    const N = 12; let f = 0;
    const gen = this._gen = (this._gen ?? 0) + 1;
    const tick = () => {
      if (this._gen !== gen) return;
      f++; const t = ease(Math.min(1, f / N));
      this._draw(from, t);
      if (f < N) setTimeout(tick, 16);
    };
    setTimeout(tick, 16);
  }
  render() { this._auto = null; this._gen = (this._gen ?? 0) + 1; this._draw(null, 1); }
  // bars: the histogram's drawn shares, when not lerped by t; a: the token's ink opacity (autoplay)
  _draw(from, t, bars, a = from ? t : 1) {
    const B = [], T = this.T, f1 = (v) => v.toFixed(1);
    const mk = (id, c, m) => `<marker id="${id}" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="${m}" markerHeight="${m}" orient="auto"><path d="M0 0 L8 4 L0 8 Z" fill="${C(c)}"/></marker>`;
    B.push(`<defs>${mk('es-ib', '#0b0b0b', 6)}${mk('es-nv', '#898781', 5)}</defs>`);
    // the EP group: nodes, GPU cells, expert slots (base layer: nothing chosen)
    B.push(`<text class="dims" x="0" y="11">EP group: ${NODES} nodes × ${GPUS} GPUs × ${EPG} experts = ${NODES * EPN} routed experts</text>`);
    for (let n = 0; n < NODES; n++) {
      const y = rowY(n);
      B.push(`<text x="0" y="${y + 14}"${n ? '' : ' class="hd"'}>node ${n}${n ? '' : ' (ours)'}</text>`);
      B.push(`<rect x="${CX - 4}" y="${y - 4}" width="${GPUS * CP + 4}" height="${CH + 8}" rx="4" fill="${C('#f3f2ee')}" stroke="${C('#e1e0d9')}"/>`);
      for (let g = 0; g < GPUS; g++) B.push(this._cell(n, g, null));
    }
    // the routed token's copies: the old one fading out, the new one in
    const ow = from?.a ?? 1;   // the old token's ink, if an autoplay was interrupted mid-fade
    if (from?.cur && t < 1 && ow > 0) B.push(`<g opacity="${((1 - t) * ow).toFixed(3)}">${this._copies(from.cur, false)}</g>`);
    if (a > 0) B.push(`<g opacity="${a.toFixed(3)}">${this._copies(this.cur, true)}</g>`);
    // our GPU, outlined on top
    B.push(`<rect x="${cellX(0)}" y="${rowY(0)}" width="${CW}" height="${CH}" rx="2" fill="none" stroke="${C('#0b0b0b')}" stroke-width="1.5"/>`);
    const ly = H - 4;
    B.push(`<rect x="0" y="${ly - 9}" width="11" height="8" fill="${C('#2a78d6')}"/><text class="dims" x="15" y="${ly - 1}">the token's 8 experts</text>`);
    B.push(`<rect x="124" y="${ly - 10}" width="14" height="10" rx="2" fill="${C('#dcebfa')}" stroke="${C('#c3c2b7')}"/><text class="dims" x="142" y="${ly - 1}">a GPU it reaches</text>`);
    B.push(`<path d="M234 ${ly - 4} H252" stroke="${C('#0b0b0b')}" stroke-width="1.5" marker-end="url(#es-ib)"/><text class="dims" x="256" y="${ly - 1}">IB</text>`);
    B.push(`<path d="M276 ${ly - 4} H294" stroke="${C('#898781')}" stroke-width="1.2" marker-end="url(#es-nv)"/><text class="dims" x="298" y="${ly - 1}">NVLink</text>`);
    // this token
    const c = copies(this.cur), remote = [...c.nodes.keys()].filter((n) => n).sort((a, b) => a - b);
    if (!from && a < 1) B.push(`<g data-ink opacity="${a.toFixed(3)}">`);
    B.push(`<text class="hd" data-token="${this._auto?.n ?? T.n}" x="${RX}" y="${TOP - 26}">token ${fmt(this._auto?.n ?? T.n)}</text>`);
    B.push(`<text x="${RX}" y="${TOP - 10}"><tspan class="num" data-cur-ib="${c.ib}">${c.ib}</tspan> IB ${c.ib === 1 ? 'copy' : 'copies'}${remote.length ? ` (node${remote.length > 1 ? 's' : ''} ${remote.join(', ')})` : ''} · <tspan class="num" data-cur-nv="${c.nv}">${c.nv}</tspan> NVLink ${c.nv === 1 ? 'copy' : 'copies'}</text>`);
    if (!from && a < 1) B.push('</g>');
    if (this._msg) B.push(`<text data-seek-msg x="${RX}" y="${TOP + 12}" style="fill: var(--c-2a78d6)">${this._msg}</text>`);
    // histogram of IB copies per token, shares on a fixed 0 … 100% scale; this token's bin in blue
    // (crossfading from the previous token's); each column is a click target (seek)
    const kOld = from?.cur && t < 1 ? copies(from.cur).ib : -1;
    B.push(`<text class="dims" data-routed="${T.n}" x="${RX}" y="${HB - HH - 22}">IB copies per token, share of the ${fmt(T.n)} token${T.n > 1 ? 's' : ''} routed</text>`);
    T.hist.forEach((h, k) => {
      const s = bars ? bars[k] : lerp(from?.share?.[k] ?? h / T.n, h / T.n, t), x = RX + 22 + k * 62, bh = s * HH;
      // the blue bin's overlay and its labels' blue ease together; a bin both tokens share stays blue
      const op = k === c.ib ? (k === kOld ? lerp(ow, 1, t) : a) : k === kOld ? (1 - t) * ow : 0;
      const hi = (base) => `${k === c.ib ? ' data-cur="1"' : ''}${op > 0 ? ` style="fill: ${mix(C(base), C('#2a78d6'), op)}; font-weight: ${Math.round(400 + 200 * op)}"` : ''}`;
      B.push(`<g data-seek="${k}"><rect class="hit" x="${x - 6}" y="${HB - HH - 14}" width="56" height="${HH + 32}" rx="3" fill="transparent"/>`);
      B.push(`<rect data-bar="${k}" data-share="${h / T.n}" x="${x}" y="${f1(HB - bh)}" width="44" height="${f1(bh)}" fill="${C('#52514e')}"/>`);
      if (op > 0) B.push(`<rect x="${x}" y="${f1(HB - bh)}" width="44" height="${f1(bh)}" fill="${C('#2a78d6')}" opacity="${op.toFixed(3)}"/>`);
      B.push(`<line x1="${x - 6}" y1="${HB}" x2="${x + 50}" y2="${HB}" stroke="${C('#c3c2b7')}"/>`);
      B.push(`<text class="num" x="${x + 22}" y="${f1(HB - bh - 4)}" text-anchor="middle"${hi('#0b0b0b')}>${(s * 100).toFixed(s > 0 && s < 0.0005 ? 3 : 1)}%</text>`);
      B.push(`<text x="${x + 22}" y="${HB + 13}" text-anchor="middle"${hi('#52514e')}>${k}</text></g>`);
    });
    // the running mean, log tokens, against the sheet's K2
    B.push(`<text class="dims" x="${RX}" y="${MY0 - 14}">running mean of IB copies per token</text>`);
    for (const v of [2, 3, 4]) B.push(`<line x1="${MX0}" y1="${my(v)}" x2="${MX1}" y2="${my(v)}" stroke="${C('#e1e0d9')}"/><text class="dims" x="${MX0 - 5}" y="${my(v) + 3}" text-anchor="end">${v}</text>`);
    B.push(`<line x1="${MX0}" y1="${my(3.5)}" x2="${MX1}" y2="${my(3.5)}" stroke="${C('#0b0b0b')}" stroke-width="1" stroke-dasharray="4 3"/>`);
    B.push(`<text class="dims" x="${MX1}" y="${my(3.5) - 4}" text-anchor="end">the sheet's K2 = 3.5</text>`);
    ['1', '10', '100', '1K', '10K', '100K'].forEach((l, e) => B.push(`<line x1="${f1(mx(10 ** e))}" y1="${MY1}" x2="${f1(mx(10 ** e))}" y2="${MY1 + 3}" stroke="${C('#898781')}"/><text class="dims" x="${f1(mx(10 ** e))}" y="${MY1 + 13}" text-anchor="middle">${l}</text>`));
    B.push(`<text class="dims" x="${MX1}" y="${MY1 + 25}" text-anchor="end">tokens routed</text>`);
    const pts = [...T.trace.filter(([n]) => n < T.n), [T.n, T.ib / T.n]];
    B.push(`<polyline points="${pts.map(([n, v]) => `${f1(mx(n))},${f1(my(v))}`).join(' ')}" fill="none" stroke="${C('#2a78d6')}" stroke-width="1.5"/>`);
    B.push(`<circle cx="${f1(mx(T.n))}" cy="${f1(my(T.ib / T.n))}" r="3" fill="${C('#2a78d6')}"/>`);
    // the means so far, vs the exact ones
    const yy = H - 4;
    B.push(`<text x="${RX}" y="${yy - 14}">mean per token: <tspan class="num" data-mean="ib" data-true="${T.ib / T.n}">${(T.ib / T.n).toFixed(3)}</tspan> IB · <tspan class="num" data-mean="nv" data-true="${T.nv / T.n}">${(T.nv / T.n).toFixed(3)}</tspan> NVLink</text>`);
    B.push(`<text class="dims" x="${RX}" y="${yy}">exact, the sheet's routing: ${EXACT.ib} = 4 × 7/8 IB · ${EXACT.nv.toFixed(3)} NVLink</text>`);
    this._chart.innerHTML = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="Tokens routed through an EP group of 8 nodes, 8 GPUs each, 4 experts per GPU. Each token's 8 experts are marked; one Infiniband copy goes to each remote node that hosts any of them, landing on the GPU with our local index, which forwards it over NVLink to the other GPUs that need it. Beside it, a histogram of IB copies per token and their running mean against the sheet's K2 = 3.5.">${B.join('')}</svg>`;
  }
  // one GPU cell and its 4 expert slots; `on` = the chosen experts, drawn as a reached cell (null = base layer)
  _cell(n, g, on) {
    const x = cellX(g), y = rowY(n), B = [];
    B.push(`<rect x="${x}" y="${y}" width="${CW}" height="${CH}" rx="2" fill="${C(on ? '#dcebfa' : '#ffffff')}" stroke="${C('#c3c2b7')}"/>`);
    for (let k = 0; k < EPG; k++) {
      const e = n * EPN + g * EPG + k, chosen = on?.has(e);
      B.push(`<rect${chosen ? ` data-expert="${e}"` : ''} x="${x + 1 + (k & 1) * 12}" y="${y + 1 + (k >> 1) * 9}" width="11" height="8" fill="${C(chosen ? '#2a78d6' : '#eeede7')}"/>`);
    }
    return B.join('');
  }
  // a routed token overlaid on the base layer: reached cells, chosen experts, IB and NVLink copies
  _copies(experts, tag) {
    const B = [], on = new Set(experts), { nodes } = copies(experts), last = Math.max(...nodes.keys());
    for (const [n, gs] of [...nodes].sort((a, b) => a[0] - b[0])) {
      const y = rowY(n);
      for (let g = 0; g < GPUS; g++) if (gs.has(g) || g === 0) B.push(this._cell(n, g, on));
      if (n) {   // IB: from our GPU, down a trunk in the gutter, into the landing GPU (local index 0)
        B.push(`<path${tag ? ` data-ib="${n}"` : ''} d="M${cellX(0)} ${rowY(0) + CH / 2} H${TX} V${y + CH / 2} H${cellX(0)}" fill="none" stroke="${C('#0b0b0b')}" stroke-width="1.5" marker-end="url(#es-ib)"/>`);
        if (n < last) B.push(`<circle cx="${TX}" cy="${y + CH / 2}" r="2.5" fill="${C('#0b0b0b')}"/>`);
      }
      // NVLink: landing GPU → each other GPU holding an expert, along a bus in the gap above the row
      const gs1 = [...gs].filter((g) => g).sort((a, b) => a - b), by = y - 10;
      for (const g of gs1) {
        const x1 = cellX(g) + CW / 2;
        B.push(`<path${tag ? ` data-nv="${n}:${g}"` : ''} d="M${cellX(0) + CW / 2} ${y} V${by} H${x1} V${y}" fill="none" stroke="${C('#898781')}" stroke-width="1.2" marker-end="url(#es-nv)"/>`);
        if (g !== gs1.at(-1)) B.push(`<circle cx="${x1}" cy="${by}" r="2" fill="${C('#898781')}"/>`);
      }
    }
    return B.join('');
  }
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
if (typeof customElements !== 'undefined' && !customElements.get('dsv3-epsim')) customElements.define('dsv3-epsim', Dsv3Epsim);
