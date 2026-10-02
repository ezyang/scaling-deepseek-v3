// <dsv3-fsdpsched>: the ZeRO-3 schedule of a two-layer toy model built from
// DeepSeek-V3 MoE layers, on one GPU of DeepSeek's 2,048 (EP64), at speed of
// light (studies/03-roofline.html). Two tracks: compute (forward, backward =
// 2× forward) and InfiniBand (a parameter all-gather before each layer's
// forward and again before its backward, a gradient reduce-scatter after its
// backward). Every gather is one "move" of the layer's BF16 parameters as
// the page's cells price it: the non-expert eighth that crosses IB
// (hierarchical over the node) plus our whole expert slice; a reduce-scatter
// moves FP32 gradients, two moves' worth. Compute scales
// with the tokens in a microbatch; the collectives don't. m="2" pins the
// microbatch count (a static figure); without it the knob is microbatches
// per step, resting at 8 (the busiest GPU's local batch of 8, split into m microbatches,
// each paying its own gathers). bars: the local lens's fit chart (ZeRO-3,
// no PP) rides below, driven by the same knob — the memory the split buys. Collectives overlap compute:
// the compute track is scheduled with each gather prefetched a layer ahead,
// then each gather is drawn as late as that schedule allows. Above the tracks,
// this GPU's transient memory on an unlabeled axis shared by every figure
// (the tallest state, one microbatch, fills it): preallocated staging buffers
// for the gathers and the reduce-scatters, and the saved activations stacked
// on them. The persistent shards are a flat floor, left off.
import { DSV3 as A, HARDWARE } from './model.js';
import { PARAMS } from './params.js';
import { layerAnalysis } from './memory.js';
import { buildCells, cellsEnv } from './cells.js';
import { blockGraph, analyze, RECOMPUTE_PRESETS } from './blockgraph.js';
import { resolveMatmuls } from './recipes.js';
import { localBatch, mbChoices } from './localmodel.js';
import { C } from './theme.js';
import { knobCss } from './ui.js';
import { attachTip } from './tip.js';

const PEAK = 1456.6e12, PEAK16 = 757.7e12, IB = 50e9;   // best measured FP8 and BF16 GEMMs (the page's π^sol_fp8, π^sol_bf16: Smol), IB per GPU at spec (β_IB)
const GPUS = 2048, EP = 64, NODE = 8, SEQ = 4096;
export const LOCAL = 8;                             // the busiest GPU's sequences: ⌈15,360 ÷ 2,048⌉
export const MS = [1, 2, 4, 8];                     // microbatches per step
// one move of a MoE layer's parameters over IB, per GPU (the page's Vne/Vexp per layer)
const NX = PARAMS.moeBlock - A.routedExperts * PARAMS.expert;   // attention, shared expert, router, norms
const XS = A.routedExperts / EP * PARAMS.expert;                // our expert slice
export const MOVE_B = NX / NODE * (GPUS / NODE - 1) / (GPUS / NODE) * 2 + XS * (GPUS / EP - 1) / (GPUS / EP) * 2;
export const MOVE_S = MOVE_B / IB;
const RS_S = 2 * MOVE_S;                            // FP32 gradients: two moves' worth
// forward FLOPs per token of a MoE layer: 2 × active params + the causal attention core (the page's 6N + C3, ÷ 3),
// FP8 except the router and the attention core (BF16)
// memory: the bars' stash policy (activation bytes per 4,096-token sequence
// per layer), one layer's whole BF16 parameters and its FP32 gradients. The staging buffers are
// preallocated, each double-buffered (a layer in use while the next gathers;
// a gradient in its reduce-scatter while the next backward writes one)
const RECIPE = 'dsv3-fp8', RECOMPUTE = 'dsv3';
export const ACT_SEQ = layerAnalysis('moe', { recipe: RECIPE, recompute: RECOMPUTE, seqLen: SEQ }).savedBytes * SEQ;
export const LAYER_B = (NX + XS) * 2, GRAD_B = 2 * LAYER_B, STAGE = 2;
export const FWD8 = 2 * (PARAMS.activeMoeBlock - PARAMS.routerWeight), FWD16 = 2 * PARAMS.routerWeight + A.heads * (A.qkNope + A.qkRope + A.vHead) * SEQ;
export const fwdS = (mbs) => (FWD8 / PEAK + FWD16 / PEAK16) * mbs * SEQ;

// the schedule: blocks {k (tween key), track 'c'|'n', op, layer, mb, t0, t1}
// plus the compute track's idle gaps (op 'idle')
export function schedule(mbs) {
  const m = LOCAL / mbs, f = fwdS(mbs), B = [];
  const ops = [];                                     // compute program: per microbatch F1 F2 B2 B1
  for (let j = 0; j < m; j++) for (const [op, layer] of [['F', 1], ['F', 2], ['B', 2], ['B', 1]]) ops.push({ op, layer, mb: j, d: op === 'F' ? f : 2 * f });
  // an all-gather's key names the pass it feeds (AG1F, AG1B)
  const put = (track, op, layer, mb, t0, d, pass = '') => { B.push({ k: `${op}${layer}${pass}:${mb}`, track, op, layer, mb, t0, t1: t0 + d }); return t0 + d; };
  // one IB queue served in ready order: a gather is ready when the previous
  // compute op starts (prefetch one layer ahead), a reduce-scatter when its backward ends
  let tc = 0, tn = 0, prev = 0; const rs = [];
  const serve = (op, layer, mb, ready, pass) => { tn = put('n', op, layer, mb, Math.max(tn, ready), op === 'RS' ? RS_S : MOVE_S, pass); return tn; };
  for (const o of ops) {
    while (rs.length && rs[0].ready <= prev) { const r = rs.shift(); serve('RS', r.layer, r.mb, r.ready); }
    const ag = serve('AG', o.layer, o.mb, prev, o.op);
    prev = Math.max(tc, ag);
    tc = put('c', o.op, o.layer, o.mb, prev, o.d);
    if (o.op === 'B') rs.push({ layer: o.layer, mb: o.mb, ready: tc });
  }
  for (const r of rs) serve('RS', r.layer, r.mb, r.ready);
  // then push every gather as late as it can go without stalling compute:
  // it ends when its consumer starts (or where the next collective begins).
  // Reduce-scatters stay as early as possible (they free gradient memory)
  const at = new Map(B.map((b) => [b.k, b])), q = B.filter((b) => b.track === 'n');
  for (let i = q.length - 1, lim = Infinity; i >= 0; lim = q[i--].t0) {
    const g = q[i];
    if (g.op === 'AG') { g.t1 = Math.min(lim, at.get(g.k.slice(3, 4) + g.layer + ':' + g.mb).t0); g.t0 = g.t1 - MOVE_S; }
  }
  // the compute track's idle gaps, keyed by the op they precede
  const cs = B.filter((b) => b.track === 'c'), end = Math.max(...B.map((b) => b.t1));
  let t = 0;
  for (const c of [...cs, { k: 'end', t0: end }]) {
    if (c.t0 - t > 1e-9) B.push({ k: 'idle:' + c.k, track: 'c', op: 'idle', t0: t, t1: c.t0 });
    t = c.t1 ?? end;
  }
  // memory at t, [gather staging, reduce-scatter staging, activations] in
  // bytes: the staging buffers hold for the whole step; a forward ramps its
  // stash in and its backward ramps it out
  const ramp = (b, t) => Math.min(1, Math.max(0, (t - b.t0) / (b.t1 - b.t0)));
  const memAt = (t) => {
    if (t < 0 || t >= end) return [0, 0, 0];
    let a = 0;
    for (const b of cs) if (b.op === 'F') a += ramp(b, t) - ramp(at.get(`B${b.layer}:${b.mb}`), t);
    return [STAGE * LAYER_B, STAGE * GRAD_B, a * ACT_SEQ * mbs];
  };
  const bps = [...new Set([0, ...q.concat(cs).flatMap((b) => [b.t0, b.t1])])].sort((a, b) => a - b);
  const comp = cs.reduce((s, b) => s + b.t1 - b.t0, 0), ib = B.filter((b) => b.track === 'n').reduce((s, b) => s + b.t1 - b.t0, 0);
  return { blocks: B, m, step: end, comp, ib, exposed: end - comp, memAt, bps };
}
// a memory curve's corners: each breakpoint's left and right values (steps are vertical)
const EPS = 1e-9;
export const memPts = (S, bps = S.bps) => bps.flatMap((t) => [[t, S.memAt(t - EPS)], [t, S.memAt(t + EPS)]]);
const peak = (S) => Math.max(...memPts(S).map(([, v]) => v[0] + v[1] + v[2]));
// one fixed time axis for every knob state, so compute visibly shrinks and the collectives don't
export const T_MAX = Math.ceil(Math.max(...MS.map((m) => schedule(LOCAL / m).step)) * 100) / 100;
// and one memory axis for every figure
export const MEM_MAX = Math.max(...MS.map((m) => peak(schedule(LOCAL / m))));

const W = 738, LX = 64, X1 = W - 22, MY = 22, MH = 56, TY = MY + MH + 8, TH = 24, NY = TY + TH + 8, AY = NY + TH + 8, H = AY + 58;
const sx = (t) => LX + (X1 - LX) * t / T_MAX;
const my = (v) => MY + MH - (MH - 4) * v / MEM_MAX;
const MEM = ['#2a78d6', '#eb6834', '#eda100'];      // gather staging (AG blue), reduce-scatter staging (RS orange), activations (the bars' amber)
const FILL = { F: '#e1e0d9', B: '#c3c2b7', AG: '#2a78d6', RS: '#eb6834', idle: '#fdf1f1' };
const INK = { F: '#1c1c1a', B: '#1c1c1a', AG: '#ffffff', RS: '#ffffff', idle: '#d03b3b' };
const ms = (s) => (s * 1e3).toFixed(1) + ' ms';
const ease = (p) => 1 - (1 - p) ** 3;

const CSS = `
.fsc { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-0b0b0b);
  border: 1px solid var(--c-e1e0d9); border-radius: 6px; background: var(--c-fcfcfb); padding: 8px 10px;
  width: ${W + 22}px; max-width: 100%; box-sizing: border-box; position: relative; }
.fsc .top { display: flex; flex-wrap: wrap; align-items: stretch; gap: 8px 10px; padding-bottom: 6px; }
${knobCss('.fsc .top')}
.fsc svg { display: block; }
.fsc text { font: 11px system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--c-52514e); }
.fsc .dims { font-size: 9.5px; fill: var(--c-898781); }
.fsc .hd { font-weight: 600; fill: var(--c-1c1c1a); }
/* hover link: a gather and the compute op it feeds tint together */
.fsc rect.hl { filter: brightness(0.86); }
:root.dark .fsc rect.hl { filter: brightness(1.3); }
/* linked bars share the card: a hairline under the timeline, no card of their own */
.fsc dsv3-layer { display: block; margin: 8px 0 0; padding-top: 8px; border-top: 1px solid var(--c-e1e0d9); }
.fsc dsv3-layer[snapshot] .lv { border: 0; border-radius: 0; background: none; padding: 0; width: auto; }
`;

class Dsv3Fsdpsched extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    const pin = MS.includes(+this.getAttribute('m')) ? +this.getAttribute('m') : null;
    this._key = !pin && this.id ? 'f:' + this.id : null;
    const st = this._key ? readState(this._key) : null;
    this.m = pin ?? (MS.includes(st?.m) ? st.m : 8);
    const style = document.createElement('style'); style.textContent = CSS;
    this._root = el('div', 'fsc');
    this._top = el('div', 'top');
    this._chart = el('div');
    this._root.append(...(pin ? [] : [this._top]), this._chart);
    this.append(style, this._root);
    if (!pin) {
      const g = el('span', 'pargrp'), l = el('div', 'parlab'), r = el('div', 'parrow'), s = el('span', 'stp');
      l.textContent = 'MBs'; l.title = 'microbatches per step (local batch: 8 sequences)'; s.dataset.knob = 'm';
      for (const v of MS) {
        const b = document.createElement('button'); b.type = 'button'; b.textContent = v; b.dataset.v = v;
        b.onclick = () => this.set('m', v); s.append(b);
      }
      r.append(s); g.append(l, r); this._top.append(g);
    }
    if (this.hasAttribute('bars')) {
      // a static local-lens figure (no knobs, no URL state of its own) that this knob drives
      const l = this._bars = document.createElement('dsv3-layer');
      for (const [k, v] of [['snapshot', ''], ['local', ''], ['cumulative', ''], ['lens', 'param-bytes'], ['gbs', '15360'],
        ['recipe', 'dsv3-fp8'], ['recompute', 'dsv3'], ['controls', 'static'], ['detail', ''], ['nocaption', ''],
        ['from', JSON.stringify({ pp: 1, ep: EP, zero: 3, mb: this.m })]]) l.setAttribute(k, v);
      this._root.append(l);
    }
    attachTip(this._chart, (ev) => this._tip(ev), { parent: this._root });
    this._chart.addEventListener('mouseover', (ev) => this._hover(ev.target.closest?.('rect[data-k]')?.dataset.k ?? null));
    this._chart.addEventListener('mouseleave', () => this._hover(null));
    this._sync();
    this.render();
    addEventListener('dsv3-theme', () => this.render());
  }
  _sync() {
    for (const b of this._top.querySelectorAll('button')) b.classList.toggle('on', b.dataset.v === String(this.m));
    if (this._key) writeState(this._key, { m: this.m }, '{"m":8}');
  }
  set(knob, v) {
    if (v === this[knob]) return;
    const from = schedule(this.mbs);
    this[knob] = v;
    this._sync();
    const l = this._bars;
    if (l?.setLocal) l.setLocal(() => { l.mb = v; l._syncMb(); });
    const N = 12; let f = 0;   // ~200 ms: shared blocks slide, the rest fade
    const gen = this._gen = (this._gen ?? 0) + 1;
    const tick = () => {
      if (this._gen !== gen) return;
      f++; this._draw(from, ease(Math.min(1, f / N)));
      if (f < N) setTimeout(tick, 16);
    };
    setTimeout(tick, 16);
  }
  get mbs() { return LOCAL / this.m; }
  render() { this._gen = (this._gen ?? 0) + 1; this._draw(null, 1); }
  _draw(from, t) {
    const S = this._S = schedule(this.mbs), P = [];
    const old = new Map((from?.blocks ?? []).map((b) => [b.k, b])), now = new Set(S.blocks.map((b) => b.k));
    const lerp = (a, b) => a + (b - a) * t;
    const rect = (b, x0, x1, op) => {
      const y = b.track === 'c' ? TY : NY, w = sx(x1) - sx(x0);
      const idle = b.op === 'idle';
      let s = `<rect data-k="${b.k}" data-op="${b.op}" x="${sx(x0).toFixed(2)}" y="${y}" width="${Math.max(0, w).toFixed(2)}" height="${TH}" fill="${C(FILL[b.op])}"`
        + (idle ? ` stroke="${C('#d03b3b')}" stroke-width="0.8" stroke-dasharray="2 2"` : ` stroke="${C('#fcfcfb')}" stroke-width="1"`) + `${op < 1 ? ` opacity="${op.toFixed(3)}"` : ''}/>`;
      const lab = idle ? (w >= 40 ? 'exposed' : '') : b.track === 'n' ? String(b.layer) : b.op + b.layer;   // collectives: the layer (the fill says AG vs RS)
      if (lab && w >= (idle ? 40 : b.track === 'n' ? 8 : 22)) s += `<text x="${(sx(x0) + w / 2).toFixed(2)}" y="${y + TH / 2 + 4}" text-anchor="middle" style="fill:${C(INK[b.op])}; pointer-events:none"${op < 1 ? ` opacity="${op.toFixed(3)}"` : ''}>${lab}</text>`;
      return s;
    };
    P.push(`<text class="dims" data-hdr x="0" y="11">two DeepSeek-V3 MoE layers · one H800 of 2,048 (EP64, ZeRO-3) · ${S.m} microbatch${S.m > 1 ? 'es' : ''} of ${this.mbs} × 4,096 tokens · each gather as late as compute allows · speed of light</text>`);
    P.push(`<text class="hd" x="0" y="${TY + TH / 2 + 4}">compute</text><text class="hd" x="0" y="${NY + TH / 2 + 4}">InfiniBand</text>`);
    P.push(`<text class="hd" x="0" y="${MY + MH / 2 + 4}">memory</text>`);
    P.push(`<rect x="${LX}" y="${MY}" width="${X1 - LX}" height="${MH}" fill="${C('#f3f2ee')}"/>`);
    // the memory curve, stacked; a tween morphs it vertically over both schedules' corners
    const bps = from && t < 1 ? [...new Set([...from.bps, ...S.bps])].sort((a, b) => a - b) : S.bps;
    const cur = memPts(S, bps), pts = from && t < 1 ? memPts(from, bps).map(([x, v], i) => [x, v.map((y, j) => lerp(y, cur[i][1][j]))]) : cur;
    const xy = (x, v) => `${sx(x).toFixed(2)},${my(v).toFixed(2)}`;
    for (let j = 0; j < 3; j++) {
      const sum = (v, n) => v.slice(0, n).reduce((a, b) => a + b, 0);
      const top = pts.map(([x, v]) => xy(x, sum(v, j + 1))), bot = pts.map(([x, v]) => xy(x, sum(v, j))).reverse();
      P.push(`<polygon data-mem="${j}" points="${top.join(' ')} ${bot.join(' ')}" fill="${C(MEM[j])}"/>`);
    }
    for (const y of [TY, NY]) P.push(`<rect x="${LX}" y="${y}" width="${X1 - LX}" height="${TH}" fill="${C('#f3f2ee')}"/>`);
    // fading-out blocks first, underneath
    if (t < 1) for (const b of from.blocks) if (!now.has(b.k)) P.push(rect(b, b.t0, b.t1, 1 - t));
    for (const b of S.blocks) {
      const o = t < 1 ? old.get(b.k) : null;
      P.push(o ? rect(b, lerp(o.t0, b.t0), lerp(o.t1, b.t1), 1) : rect(b, b.t0, b.t1, t < 1 ? t : 1));
    }
    // the time axis
    const step = 0.05;
    P.push(`<line x1="${LX}" y1="${AY}" x2="${X1}" y2="${AY}" stroke="${C('#aba89f')}"/>`);
    for (let s = 0; s <= T_MAX + 1e-9; s += step) {
      const x = sx(s).toFixed(2);
      P.push(`<line x1="${x}" y1="${AY}" x2="${x}" y2="${AY + 4}" stroke="${C('#aba89f')}"/><text class="dims" x="${x}" y="${AY + 15}" text-anchor="middle">${Math.round(s * 1e3)}${s === 0 ? '' : ' ms'}</text>`);
    }
    const pct = Math.round(100 * S.exposed / S.step);
    P.push(`<text data-readout x="0" y="${AY + 34}">step (both layers): <tspan class="hd">${ms(S.step)}</tspan> = compute ${ms(S.comp)} + exposed IB ${ms(S.exposed)} (${pct}%) · IB busy ${ms(S.ib)}: ${4 * S.m} moves' worth of ${(MOVE_B / 1e6).toFixed(0)} MB per layer</text>`);
    const ly = AY + 52, sw = (x, op, label) => `<rect x="${x}" y="${ly - 8}" width="10" height="9" fill="${C(FILL[op])}"${op === 'idle' ? ` stroke="${C('#d03b3b')}" stroke-width="0.8" stroke-dasharray="2 2"` : ''}/><text class="dims" x="${x + 14}" y="${ly}">${label}</text>`;
    P.push(sw(0, 'F', 'forward'), sw(64, 'B', 'backward (2× forward)'), sw(190, 'AG', 'all-gather (BF16 weights)'), sw(330, 'RS', 'reduce-scatter (FP32 grads)'), sw(486, 'idle', 'compute waiting on IB'),
      `<rect x="614" y="${ly - 8}" width="10" height="9" fill="${C(MEM[2])}"/><text class="dims" x="628" y="${ly}">saved activations</text>`);
    this._chart.innerHTML = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="A timeline with two tracks, compute and InfiniBand, under a stacked curve of this GPU's transient memory (preallocated staging buffers for the gathers and reduce-scatters, with the saved activations on top), for two DeepSeek-V3 MoE layers under ZeRO-3. Each layer's weights are all-gathered before its forward and again before its backward, and its gradients are reduce-scattered after its backward. Compute blocks scale with the microbatch; the collectives stay the same size. Each gather runs as late as it can without stalling the layer that needs it.">${P.join('')}</svg>`;
    if (this._hk) this._hover(this._hk);
  }
  // a gather's key names its consumer (AG2B:3 → B2:3) and vice versa
  _hover(k) {
    this._hk = k;
    const m = k && /^(?:AG(\d)([FB])|([FB])(\d)):(\d+)$/.exec(k);
    const pair = m ? (m[1] ? [k, `${m[2]}${m[1]}:${m[5]}`] : [k, `AG${m[4]}${m[3]}:${m[5]}`]) : [];
    for (const r of this._chart.querySelectorAll('rect[data-k]')) r.classList.toggle('hl', pair.includes(r.dataset.k));
  }
  _tip(ev) {
    const k = ev.target.closest?.('rect[data-k]')?.dataset.k, b = k && this._S.blocks.find((x) => x.k === k);
    if (!b) return null;
    const d = b.t1 - b.t0, mb = this._S.m > 1 ? ` · microbatch ${b.mb + 1} of ${this._S.m}` : '';
    const tok = this.mbs * SEQ;
    if (b.op === 'idle') return `compute idle, waiting on InfiniBand: ${ms(d)}`;
    if (b.op === 'F' || b.op === 'B') {
      const tf = (f) => (f * tok * (b.op === 'B' ? 2 : 1) / 1e12).toFixed(1);
      return `${b.op === 'F' ? 'forward' : 'backward'} · layer ${b.layer}${mb} · ${tok.toLocaleString('en-US')} tokens\n${tf(FWD8)} TFLOP FP8 ÷ 1,456.6 TFLOP/s + ${tf(FWD16)} TFLOP BF16 ÷ 757.7 TFLOP/s = ${ms(d)}`;
    }
    const what = b.op === 'RS' ? `reduce-scatter of layer ${b.layer}'s gradients` : `all-gather of layer ${b.layer}'s weights, before its ${b.k[3] === 'B' ? 'backward' : 'forward'}`;
    const bytes = b.op === 'RS' ? 2 * MOVE_B : MOVE_B, dt = b.op === 'RS' ? 'FP32' : 'BF16';
    return `${what}${mb}\n${(bytes / 1e6).toFixed(1)} MB (${dt}) over IB per GPU: the non-expert 1/8 + our expert slice\n÷ 50 GB/s = ${ms(d)}, the same at any microbatch size`;
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
if (typeof customElements !== 'undefined' && !customElements.get('dsv3-fsdpsched')) customElements.define('dsv3-fsdpsched', Dsv3Fsdpsched);

// <dsv3-fsdpcurve>: the whole step on the average GPU (the cells' 7.5
// sequences: a layout without PP would round the batch to divide evenly),
// comms against compute as each GPU's share splits into m microbatches, at
// speed of light with comms fully overlapped. EP's all-to-alls scale with
// tokens, so they're flat; ZeRO-3 pays two BF16 gathers and an FP32
// reduce-scatter of every layer (the page's Tmb) per microbatch, so comms is
// a straight line in m with slope Tmb. Wherever comms outrun compute, the
// excess is exposed.
// Under it, on the same m axis, the busiest GPU's peak memory (the one that
// has to fit: 8 sequences): the page's cells at mb = 1 (T1, what the fit bars
// above show: shards W1 + G1 + O1, flat; activations A1, ∝ 1/m) plus the
// schedule figures' staging buffers, against the H800's 80 GiB. That line is
// a soft boundary (the CUDA context, NCCL's buffers and fragmentation aren't
// counted), so no crossing is marked: the over-HBM zone is tinted and fades
// out just under the line, and the call is per MBs state — only 8 is under.
const L_MOE = A.layers - A.denseLayers;
const N_EXP = L_MOE * A.routedExperts * PARAMS.expert, N_NE = PARAMS.total - N_EXP;
const N_BF16 = PARAMS.embed + L_MOE * PARAMS.routerWeight;   // the page's Nbf16: the output head (an embedding-sized copy) and the routers
// the page's V⁺disp + Vcomb, B per token per MoE layer per pass: 3.5 remote
// nodes × (FP8 hidden + fp32 scales per 128) out, × BF16 hidden back
const M_IB = 4 * (1 - NODE / EP);
export const EP_TOK = M_IB * (A.hidden + 4 * A.hidden / 128) + M_IB * A.hidden * 2;
const TOK = 15360 * SEQ / GPUS;                     // D_GPU
export const STEP = {
  // T⁺c: 6N + the recompute policy's replay of the MLA up-projections at FP8 (the page's C⁺fp8), the head,
  // routers and causal attention core at BF16 (C⁺bf16)
  comp: ((6 * (PARAMS.activeTotal - N_BF16) + 2 * A.layers * (A.qRank * A.heads * (A.qkNope + A.qkRope) + A.kvRank * A.heads * (A.qkNope + A.vHead))) / PEAK
    + (6 * N_BF16 + 3 * A.layers * A.heads * (A.qkNope + A.qkRope + A.vHead) * SEQ) / PEAK16) * TOK,
  ep: 2 * EP_TOK * L_MOE * TOK / IB,                                                                          // T⁺EP
  mb: (N_NE / NODE * (1 - NODE / GPUS) + N_EXP / EP * (1 - EP / GPUS)) * (2 + 2 + 4) / IB,                   // Tmb
};
export function at(m) {
  const fsdp = STEP.mb * m, comms = STEP.ep + fsdp, exposed = Math.max(0, comms - STEP.comp);
  return { m, ep: STEP.ep, fsdp, comms, comp: STEP.comp, exposed, step: STEP.comp + exposed };
}
export const CROSS = (STEP.comp - STEP.ep) / STEP.mb;   // the microbatch count where comms = compute
// the page's cells for a layout S, under the bars' recipe and stash policy
const ANA = (() => {
  const mm = resolveMatmuls({ recipe: RECIPE });
  return [['moe', RECOMPUTE], ['dense', RECOMPUTE], ['moe', 'none'], ['dense', 'none']]
    .map(([k, p]) => analyze(blockGraph(k, A, mm, SEQ), RECOMPUTE_PRESETS[p], false));
})();
const cellsAt = (S) => {
  const { get } = buildCells(cellsEnv({ ep: EP, world: GPUS, gbs: 15360, ...S }, ...ANA));
  return { shards: get('W1') + get('G1') + get('O1'), acts: get('A1'), total: get('T1') };
};
const MEM1 = cellsAt({ pp: 1, zero: 3, stage: 0, mb: 1 });
export const CAP = HARDWARE.h800.memGB * 2 ** 30;
export const MEMB = { shards: MEM1.shards, staging: STAGE * (LAYER_B + GRAD_B), acts1: MEM1.acts };
export function memAt(m) {
  const acts = MEMB.acts1 / m, total = MEMB.shards + MEMB.staging + acts;
  return { m, shards: MEMB.shards, staging: MEMB.staging, acts, total };
}

const CW = 700, CX0 = 52, CX1 = 570, CY0 = 36, CY1 = 236, SMAX = 0.8;   // the top panel: idle share of the step, 0 … 80%
const SY = CY1 + 26;   // the shared x axis between the panels (m above the line; <dsv3-fsdpcurve> adds microbatch size below)
const MY0 = SY + 42, MY1 = MY0 + 150, MMAX = 200 * 2 ** 30, CH = MY1 + 10;   // the memory panel, 0 … 200 GiB
const cx = (m) => CX0 + (CX1 - CX0) * (m - 1) / 7, cy = (s) => CY1 - (CY1 - CY0) * s / SMAX;
const mY = (b) => MY1 - (MY1 - MY0) * Math.min(b, MMAX) / MMAX, gib = (b) => (b / 2 ** 30).toFixed(2) + ' GiB';
const f2 = (v) => v.toFixed(2);
// the hover guide: from an m's top-panel point down through its memory to the axis (moved by _guide, never re-rendered)
// (ring1: a second top-panel ring, for <dsv3-ppcurve>'s 1F1B curve)
const guideSvg = (top, ring1 = false) => `<g data-guide display="none" pointer-events="none"><line x1="0" y1="${CY0}" x2="0" y2="${MY1}" stroke="${C('#52514e')}" stroke-width="0.75" stroke-dasharray="3 3"/>`
  + `<circle data-g="t" r="6" fill="none" stroke="${C(top)}" stroke-width="1.5"/>${ring1 ? `<circle data-g="t1" r="6" fill="none" stroke="${C(top)}" stroke-width="1.5"/>` : ''}<circle data-g="m" r="6" fill="none" stroke="${C('#eda100')}" stroke-width="1.5"/></g>`;
// the top panel, shared so the two charts read on one scale: gridlines, then the idle share
// share(m) over m0 … m1 at x(m) as a pale area under a red curve (area tagged `area`)
const idleTop = (P, x, share, m0, m1, area) => {
  for (let v = 0; v <= SMAX * 100; v += 10) P.push(`<line x1="${CX0}" y1="${f2(cy(v / 100))}" x2="${CX1}" y2="${f2(cy(v / 100))}" stroke="${C(v ? '#eeede7' : '#aba89f')}"/><text class="dims" x="${CX0 - 6}" y="${f2(cy(v / 100) + 3)}" text-anchor="end">${v}%</text>`);
  const pts = Array.from({ length: 141 }, (_, i) => m0 + (m1 - m0) * i / 140).map((m) => `${f2(x(m))},${f2(cy(share(m)))}`).join(' ');
  P.push(`<polygon data-area="${area}" points="${pts} ${f2(x(m1))},${CY1} ${f2(x(m0))},${CY1}" fill="${C('#fdf1f1')}"/>`,
    `<polyline data-idle points="${pts}" fill="none" stroke="${C('#d03b3b')}" stroke-width="2"/>`);
};
const CSS2 = `
.fcv { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; position: relative; width: ${CW}px; max-width: 100%; }
.fcv svg { display: block; }
.fcv text { font: 11px system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--c-52514e); }
.fcv .dims { font-size: 10px; fill: var(--c-898781); }
.fcv .hd { font-weight: 600; fill: var(--c-1c1c1a); }
.fcv circle[data-m] { cursor: default; }
.fcv .dsv3-tip { width: max-content; }   /* near the right edge, overhang the chart rather than wrap inside it */
`;
class Dsv3Fsdpcurve extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    const style = document.createElement('style'); style.textContent = CSS2;
    this._root = el('div', 'fcv');
    this._chart = this._root.appendChild(el('div'));
    this.append(style, this._root);
    attachTip(this._chart, (ev) => this._tip(ev), { parent: this._root });
    this._chart.addEventListener('mouseleave', () => this._guide(null));
    this.render();
    addEventListener('dsv3-theme', () => this.render());
  }
  render() {
    const P = [], yc = STEP.comp, sh = (m) => at(m).exposed / at(m).step, pc = (q) => Math.round(100 * q.exposed / q.step);
    P.push(`<text class="dims" data-hdr x="0" y="11">idle share of the step on one H800 of 2,048 (EP64, ZeRO-3, no PP) · the average GPU's 7.5 sequences · speed of light, comms fully overlapped</text>`);
    idleTop(P, cx, sh, 1, 8, 'exposed');
    // the crossover: below it the collectives hide under compute (none on the axis when comms exceed compute even at m = 1)
    if (CROSS >= 1) P.push(`<circle data-cross cx="${f2(cx(CROSS))}" cy="${CY1}" r="3.5" fill="${C('#fcfcfb')}" stroke="${C('#0b0b0b')}" stroke-width="1.5"/>`,
      `<text class="dims" x="${f2(cx(2.35))}" y="${CY1 - 6}">← comms = compute at m = ${CROSS.toFixed(2)}</text>`);
    P.push(`<text x="${f2(cx(5.3))}" y="${f2(cy(0.16))}" text-anchor="middle" style="fill:${C('#d03b3b')}">exposed = (comms − compute) ÷ comms</text>`,
      `<text class="dims" x="${f2(cx(5.3))}" y="${f2(cy(0.16) + 14)}" text-anchor="middle">comms = EP ${f2(STEP.ep)} s + m × T<tspan dy="2" font-size="8">mb</tspan><tspan dy="-2"> ${f2(STEP.mb)} s; compute ${f2(STEP.comp)} s</tspan></text>`);
    const rl = (y, t, cls = '', fill = '') => `<text${cls ? ` class="${cls}"` : ''} x="${CX1 + 6}" y="${f2(y)}"${fill ? ` style="fill:${C(fill)}"` : ''}>${t}</text>`;
    P.push(rl(cy(sh(8)) + 4, `exposed ${pc(at(8))}%`, 'hd', '#d03b3b'), `<text class="dims" x="${CX1 + 6}" y="${f2(cy(sh(8)) + 16)}">waiting on IB</text>`);
    // the MBs knob's states, with their exposed share of the step
    for (const m of [1, 2, 4, 8]) {
      const q = at(m), x = cx(m), y = cy(sh(m));
      P.push(`<circle data-m="${m}" cx="${f2(x)}" cy="${f2(y)}" r="3.5" fill="${C('#d03b3b')}" stroke="${C('#fcfcfb')}" stroke-width="1"/>`,
        `<text data-pct="${m}" x="${f2(m === 8 ? x - 8 : m === 1 ? x + 4 : x - 6)}" y="${f2(y - 8)}" text-anchor="${m === 1 ? 'start' : 'end'}" class="hd">${pc(q)}%</text>`);
    }
    P.push(`<text data-only class="hd" x="${CX0 + 8}" y="${CY0 + 12}">only m = 8 fits</text>`,
      `<text class="dims" x="${CX0 + 8}" y="${CY0 + 26}">the one MBs state under 80 GiB, below</text>`);
    // the memory panel: the shards and staging flat, the activations ∝ 1/m on them (clamped at the top)
    P.push(`<text class="dims" data-mhdr x="${CX0}" y="${MY0 - 7}">peak memory on the busiest GPU (8 sequences) · the fit bars' total + the staging buffers above</text>`);
    // the over-HBM zone, fading out just under the line (a soft boundary: what isn't counted eats into it)
    P.push(`<defs><linearGradient id="fcv-cap" x2="0" y2="1"><stop offset="0" stop-color="${C('#fdf1f1')}"/><stop offset="1" stop-color="${C('#fdf1f1')}" stop-opacity="0"/></linearGradient></defs>`,
      `<rect data-over x="${CX0}" y="${MY0}" width="${CX1 - CX0}" height="${f2(mY(CAP) - MY0)}" fill="${C('#fdf1f1')}"/>`,
      `<rect data-soft x="${CX0}" y="${f2(mY(CAP))}" width="${CX1 - CX0}" height="22" fill="url(#fcv-cap)"/>`);
    for (let v = 0; v <= 200; v += 40) P.push(`<line x1="${CX0}" y1="${f2(mY(v * 2 ** 30))}" x2="${CX1}" y2="${f2(mY(v * 2 ** 30))}" stroke="${C(v ? '#eeede7' : '#aba89f')}"/><text class="dims" x="${CX0 - 6}" y="${f2(mY(v * 2 ** 30) + 3)}" text-anchor="end">${v}${v ? ' GiB' : ''}</text>`);
    const flat = MEMB.shards + MEMB.staging, grid = Array.from({ length: 141 }, (_, i) => 1 + i * 0.05);
    P.push(`<rect data-area="flat" x="${CX0}" y="${f2(mY(flat))}" width="${CX1 - CX0}" height="${f2(mY(0) - mY(flat))}" fill="${C('#c3c2b7')}"/>`);
    P.push(`<polygon data-area="acts" points="${grid.map((m) => `${f2(cx(m))},${f2(mY(memAt(m).total))}`).join(' ')} ${f2(cx(8))},${f2(mY(flat))} ${f2(cx(1))},${f2(mY(flat))}" fill="${C('#eda100')}"/>`);
    for (const m of [2, 4, 8]) P.push(`<circle data-mm="${m}" cx="${f2(cx(m))}" cy="${f2(mY(memAt(m).total))}" r="3.5" fill="${C('#eda100')}" stroke="${C('#fcfcfb')}" stroke-width="1"/>`);
    P.push(`<line data-cap x1="${CX0}" y1="${f2(mY(CAP))}" x2="${CX1}" y2="${f2(mY(CAP))}" stroke="${C('#d03b3b')}" stroke-width="1.5"/>`,
      `<text x="${f2(cx(3.3))}" y="${f2(mY(150 * 2 ** 30))}" text-anchor="middle" style="fill:${C('#d03b3b')}">doesn't fit</text>`,
      `<text x="${f2(cx(4))}" y="${f2(mY(memAt(4).total) - 8)}" text-anchor="middle" style="fill:${C('#d03b3b')}">${gib(memAt(4).total)}</text>`,
      `<text class="dims" x="${f2(cx(MEMB.acts1 / (MMAX - flat)) + 14)}" y="${MY0 + 11}">↑ ${Math.round(memAt(1).total / 2 ** 30)} GiB at m = 1</text>`);
    const ml = (b, t, cls = '', fill = '') => `<text${cls ? ` class="${cls}"` : ''} x="${CX1 + 6}" y="${f2(mY(b) + 4)}"${fill ? ` style="fill:${C(fill)}"` : ''}>${t}</text>`;
    P.push(ml(CAP, 'H800 80 GiB', 'hd', '#d03b3b'), `<text class="dims" x="${CX1 + 6}" y="${f2(mY(CAP) + 16)}">less what isn't counted</text>`, ml(memAt(8).total, `memory ${gib(memAt(8).total)}`, 'hd'), ml((flat + memAt(8).total) / 2, 'saved activations'), ml(flat / 2 - 2 ** 30, 'shards + staging'));
    // the shared x axis, read two ways on one line: microbatches per step poking up, and the same
    // points as microbatch size poking down (7.5 ÷ m sequences, so round sizes land unevenly)
    P.push(`<line x1="${CX0}" y1="${SY}" x2="${CX1}" y2="${SY}" stroke="${C('#aba89f')}"/>`);
    for (let m = 1; m <= 8; m++) {
      const x = f2(cx(m));
      P.push(`<line x1="${x}" y1="${SY - 5}" x2="${x}" y2="${SY}" stroke="${C('#aba89f')}"/><text x="${x}" y="${SY - 9}" text-anchor="middle">${m}</text>`);
    }
    for (const n of [7.5, 5, 4, 3, 2, 1.5, 1.25, 1]) {
      const x = f2(cx(7.5 / n));
      P.push(`<line data-mbs="${n}" x1="${x}" y1="${SY}" x2="${x}" y2="${SY + 5}" stroke="${C('#aba89f')}"/><text x="${x}" y="${SY + 17}" text-anchor="middle">${n}</text>`);
    }
    P.push(`<text class="dims" x="${CX1 + 14}" y="${SY - 9}">microbatches, m</text><text class="dims" x="${CX1 + 14}" y="${SY + 17}">microbatch size</text>`,
      `<text class="dims" x="${CX1 + 14}" y="${SY + 29}">= 7.5 ÷ m sequences</text>`);
    P.push(guideSvg('#d03b3b'));
    this._chart.innerHTML = `<svg width="${CW}" height="${CH}" viewBox="0 0 ${CW} ${CH}" role="img" aria-label="The idle share of the step against the number of microbatches, 1 to 8, or equivalently the microbatch size, 7.5 sequences down to 0.94. Comms are EP's all-to-alls, a flat ${f2(STEP.ep)} s, plus ZeRO-3's gathers and reduce-scatters, ${f2(STEP.mb)} s per microbatch; compute is ${f2(yc)} s. ${CROSS >= 1 ? `Comms stay hidden under compute up to ${CROSS.toFixed(2)} microbatches; past that the excess is exposed,` : 'Comms exceed compute at every count, so the excess is exposed,'} (comms − compute) ÷ comms of the step: ${pc(at(2))}% at 2, ${pc(at(4))}% at 4, ${pc(at(8))}% at 8. Below, on the same axis, the busiest GPU's peak memory falls like one over the microbatch count against the H800's 80 GiB, a soft line since not everything a GPU holds is counted: at 4 microbatches it's over, at ${gib(memAt(4).total)}, and only 8 is under it, at ${gib(memAt(8).total)}, where ${pc(at(8))}% of the step is exposed.">${P.join('')}</svg>`;
  }
  // the hover hooks <dsv3-ppcurve> overrides: m → the guide's points (x, the top curve's y, the
  // memory bytes, and optionally y1, a second top curve's); a dot → its m; svg x → the nearest MBs state; m → the tip
  _pt(m) { return { x: cx(m), y: cy(at(m).exposed / at(m).step), b: memAt(m).total }; }
  _mOf(c) { return c.dataset.m ? +c.dataset.m : c.dataset.mm ? +c.dataset.mm : CROSS; }
  _snap(x) { const u = 1 + 7 * (x - CX0) / (CX1 - CX0); return u < 1.5 ? 1 : u < 3 ? 2 : u < 6 ? 4 : 8; }
  // m (or null) → the guide at that m, ringing its point on both curves (the memory one only on scale)
  _guide(m) {
    const g = this._chart.querySelector('[data-guide]');
    if (!g) return;
    g.setAttribute('display', m == null ? 'none' : 'inline');
    if (m == null) return;
    const t = g.querySelector('[data-g=t]'), t1 = g.querySelector('[data-g=t1]'), mm = g.querySelector('[data-g=m]'), p = this._pt(m);
    g.setAttribute('transform', `translate(${f2(p.x)},0)`);
    t.setAttribute('cy', f2(p.y)); t1?.setAttribute('cy', f2(p.y1));
    g.querySelector('line').setAttribute('y1', f2(Math.min(p.y, p.y1 ?? p.y) + 6));
    mm.setAttribute('cy', f2(mY(p.b))); mm.setAttribute('visibility', p.b > MMAX ? 'hidden' : 'visible');
  }
  // a dot prices itself; anywhere else in the plot snaps to the nearest MBs state (no in-between m
  // is a real configuration)
  _tip(ev) {
    const c = ev.target.closest?.('circle[data-m], circle[data-m1], circle[data-mm], circle[data-cross]');
    let m = c ? this._mOf(c) : null;
    if (!c) {
      const svg = this._chart.querySelector('svg'), r = svg.getBoundingClientRect();
      const x = (ev.clientX - r.left) * CW / r.width, y = (ev.clientY - r.top) * svg.viewBox.baseVal.height / r.height;
      if (x >= CX0 && x <= CX1 && y >= CY0 && y <= MY1) m = this._snap(x);
    }
    this._guide(m);
    return m == null ? null : this._text(m);
  }
  _text(m) {
    const q = at(m), r = memAt(m), sec = (v) => f2(v) + ' s', mm = Number.isInteger(m) ? String(m) : m.toFixed(2);
    const head = Number.isInteger(m) ? `${m} microbatch${m > 1 ? 'es' : ''} of ${+(7.5 / m).toFixed(3)} sequences`
      : `comms = compute at ${mm} microbatches`;
    return `${head}\nexposed ${Math.round(100 * q.exposed / q.step)}% of the step: ${sec(q.exposed)} of ${sec(q.step)}`
      + `\ncomms ${sec(q.comms)} = EP ${sec(q.ep)} + ${mm} × Tmb ${sec(STEP.mb)}; compute ${sec(q.comp)}`
      + `\nmemory ${gib(r.total)} on the busiest GPU: ${r.total > CAP ? `${gib(r.total - CAP)} over` : `${gib(CAP - r.total)} under`} 80 GiB`;
  }
}
if (typeof customElements !== 'undefined' && !customElements.get('dsv3-fsdpcurve')) customElements.define('dsv3-fsdpcurve', Dsv3Fsdpcurve);

// <dsv3-ppcurve>: the same two panels at the same scales for the pipelined
// layout — EP64, ZeRO-1, PP8 under DualPipeV (the cells' and 02's layout: 16
// virtual stages, per GPU the same layers, stashes and bubble as DSv3's
// published 16-stage DualPipe). Each pipeline's 60 sequences split into m
// microbatches; DualPipeV needs m ≥ 2·PP, and m = 60 is one sequence each,
// so the MBs states are 20 · 30 · 60. The comms don't depend on m: EP scales
// with tokens, PP's sends (a BF16 hidden vector per token across every
// chunk boundary, forward, and its gradient back) too, and ZeRO-1 syncs once
// per step — all under compute. What m buys is the bubble: DualPipeV's
// (PP/2 − 1)(F&B + B − 3W) over 16 stages with the page's F&B ≈ F + B = 3F,
// W ≈ F is 14F, out of m × 2 chunks × 3F of work: (PP − 1)/(3m) of compute.
// Dashed beside it, plain 1F1B on the same 8 ranks: (PP − 1)(F + B + W) of
// fill and drain out of m(F + B + W), (PP − 1)/m of compute, three times
// DualPipeV's (1F1B at m = 60 idles exactly what DualPipeV does at 20). Its
// comms are left hidden too, which flatters it: with no paired microbatch to
// overlap, a real 1F1B exposes the all-to-alls. The schedules' memory differs
// only by PP vs PP + ½ microbatches in flight, so the memory panel is DualPipeV's.
// Memory: the cells' T1 on the busiest rank, activations ∝ 1/m (the in-flight
// count stays PP + ½; each microbatch shrinks).
export const PPR = 8, PDP = GPUS / PPR, PLB = localBatch(15360, PDP), PMIN = 2 * PPR;
export const PMS = mbChoices(15360, PDP, PPR);
export const PSTEP = {
  comp: STEP.comp, ep: STEP.ep,
  // the average rank: 15 boundaries × both directions over 8 ranks
  pp: (2 * PPR - 1) / PPR * PLB * SEQ * 2 * A.hidden * 2 / IB,
  // FP32 gradients reduce-scattered + BF16 parameters all-gathered (6 B per parameter), the
  // non-expert share's IB leg over the stage's nodes, the experts over EDP
  z1: (N_NE / PPR / NODE * (1 - NODE / PDP) + N_EXP / PPR / EP * (1 - EP / PDP)) * 6 / IB,
};
export function atP(m) {
  const bubble = PSTEP.comp * (PPR - 1) / (3 * m), bubble1 = PSTEP.comp * (PPR - 1) / m, comms = PSTEP.ep + PSTEP.pp + PSTEP.z1;
  return { m, comms, comp: PSTEP.comp, bubble, step: PSTEP.comp + bubble, bubble1, step1: PSTEP.comp + bubble1 };   // bubble1, step1: 1F1B
}
const MEMP1 = Array.from({ length: PPR }, (_, stage) => ({ stage, ...cellsAt({ pp: PPR, zero: 1, stage, mb: PLB }) }))
  .reduce((a, b) => (b.total > a.total ? b : a));
export const MEMP = { rank: MEMP1.stage, shards: MEMP1.shards, actsSeq: MEMP1.acts };   // actsSeq: at one sequence per microbatch
export function memP(m) {
  const acts = MEMP.actsSeq * PLB / m;
  return { m, shards: MEMP.shards, acts, total: MEMP.shards + acts };
}
const px = (m) => CX0 + (CX1 - CX0) * (m - PMIN) / (PLB - PMIN);
class Dsv3Ppcurve extends Dsv3Fsdpcurve {
  render() {
    const P = [], q60 = atP(PLB), yc = PSTEP.comp, sh = (m) => atP(m).bubble / atP(m).step, pc = (q) => (100 * q.bubble / q.step).toFixed(1);
    const sh1 = (m) => atP(m).bubble1 / atP(m).step1, pc1 = (q) => (100 * q.bubble1 / q.step1).toFixed(1);
    const grid = Array.from({ length: 89 }, (_, i) => PMIN + i * 0.5);
    P.push(`<text class="dims" data-hdr x="0" y="11">idle share of the step on one H800 of 2,048 (EP64, ZeRO-1, PP8) · the average GPU · speed of light, comms fully overlapped</text>`);
    idleTop(P, px, sh, PMIN, PLB, 'bubble');
    // 1F1B, dashed over DualPipeV's area: a curve and hollow dots, no area of its own
    P.push(`<polyline data-idle1 points="${grid.map((m) => `${f2(px(m))},${f2(cy(sh1(m)))}`).join(' ')}" fill="none" stroke="${C('#d03b3b')}" stroke-width="1.5" stroke-dasharray="5 3"/>`);
    // the two schedules' formulas, each behind a swatch of its stroke, above both curves
    const fx = px(33), sw = (y, dash) => `<line x1="${f2(fx)}" y1="${f2(y - 4)}" x2="${f2(fx + 18)}" y2="${f2(y - 4)}" stroke="${C('#d03b3b')}" stroke-width="${dash ? 1.5 : 2}"${dash ? ' stroke-dasharray="5 3"' : ''}/>`;
    P.push(sw(cy(0.33), true), `<text x="${f2(fx + 24)}" y="${f2(cy(0.33))}" style="fill:${C('#d03b3b')}">1F1B: bubble = (PP − 1) ÷ (m + PP − 1) of the step</text>`,
      sw(cy(0.33) + 14, false), `<text x="${f2(fx + 24)}" y="${f2(cy(0.33) + 14)}" style="fill:${C('#d03b3b')}">DualPipeV: (PP − 1) ÷ (3m + PP − 1)</text>`,
      `<text class="dims" x="${f2(fx + 24)}" y="${f2(cy(0.33) + 28)}">of compute: 1F1B idles (PP − 1) ÷ m, DualPipeV a third of that</text>`,
      `<text class="dims" x="${f2(px(38))}" y="${f2(cy(0.5))}" text-anchor="middle">comms ${f2(q60.comms)} s (EP ${f2(PSTEP.ep)} s + PP sends ${f2(PSTEP.pp)} s + ZeRO-1 ${f2(PSTEP.z1)} s) stay under compute at every m:</text>`,
      `<text class="dims" x="${f2(px(38))}" y="${f2(cy(0.5) + 13)}" text-anchor="middle">nothing is exposed, the bubble is all that's idle</text>`,
      `<text data-flatter class="dims" x="${f2(px(38))}" y="${f2(cy(0.5) + 26)}" text-anchor="middle">(1F1B gets that free here: it has no paired microbatch to hide the all-to-alls behind)</text>`);
    const rl = (y, t, cls = '', fill = '') => `<text${cls ? ` class="${cls}"` : ''} x="${CX1 + 6}" y="${f2(y)}"${fill ? ` style="fill:${C(fill)}"` : ''}>${t}</text>`;
    P.push(rl(cy(sh1(PLB)) - 2, `1F1B ${pc1(q60)}%`, 'hd', '#d03b3b'), rl(cy(sh(PLB)) - 6, `DualPipeV ${pc(q60)}%`, 'hd', '#d03b3b'),
      `<text class="dims" x="${CX1 + 6}" y="${f2(cy(sh(PLB)) + 6)}">pipeline fill and drain</text>`);
    for (const m of PMS) {
      const q = atP(m), x = px(m), y = cy(sh(m)), y1 = cy(sh1(m));
      P.push(`<circle data-m="${m}" cx="${f2(x)}" cy="${f2(y)}" r="3.5" fill="${C('#d03b3b')}" stroke="${C('#fcfcfb')}" stroke-width="1"/>`,
        `<text data-pct="${m}" x="${f2(m === PLB ? x - 8 : x)}" y="${f2(y - 8)}" text-anchor="${m === PLB ? 'end' : 'middle'}" class="hd">${pc(q)}%</text>`,
        `<circle data-m1="${m}" cx="${f2(x)}" cy="${f2(y1)}" r="3.5" fill="${C('#fcfcfb')}" stroke="${C('#d03b3b')}" stroke-width="1.5"/>`,
        `<text data-pct1="${m}" x="${f2(m === PLB ? x - 8 : x)}" y="${f2(y1 - 8)}" text-anchor="${m === PLB ? 'end' : 'middle'}" class="hd">${pc1(q)}%</text>`);
    }
    P.push(`<text class="dims" x="${CX0 + 8}" y="${CY0 + 12}">DualPipeV needs m ≥ 2 × PP = ${PMIN}: the axis starts there</text>`,
      `<text data-only class="hd" x="${f2(px(PLB) - 8)}" y="${CY0 + 12}" text-anchor="end">only m = ${PLB} fits</text>`,
      `<text class="dims" x="${f2(px(PLB) - 8)}" y="${CY0 + 26}" text-anchor="end">one sequence per microbatch: the batch splits no further</text>`);
    // the memory panel, as in the chart above
    P.push(`<text class="dims" data-mhdr x="${CX0}" y="${MY0 - 7}">peak memory on the busiest GPU (rank ${MEMP.rank}) · the fit bars' total (ZeRO-1: nothing staged)</text>`);
    P.push(`<defs><linearGradient id="pcv-cap" x2="0" y2="1"><stop offset="0" stop-color="${C('#fdf1f1')}"/><stop offset="1" stop-color="${C('#fdf1f1')}" stop-opacity="0"/></linearGradient></defs>`,
      `<rect data-over x="${CX0}" y="${MY0}" width="${CX1 - CX0}" height="${f2(mY(CAP) - MY0)}" fill="${C('#fdf1f1')}"/>`,
      `<rect data-soft x="${CX0}" y="${f2(mY(CAP))}" width="${CX1 - CX0}" height="22" fill="url(#pcv-cap)"/>`);
    for (let v = 0; v <= 200; v += 40) P.push(`<line x1="${CX0}" y1="${f2(mY(v * 2 ** 30))}" x2="${CX1}" y2="${f2(mY(v * 2 ** 30))}" stroke="${C(v ? '#eeede7' : '#aba89f')}"/><text class="dims" x="${CX0 - 6}" y="${f2(mY(v * 2 ** 30) + 3)}" text-anchor="end">${v}${v ? ' GiB' : ''}</text>`);
    const flat = MEMP.shards, r60 = memP(PLB), miss = memP(PMS[PMS.length - 2]);
    P.push(`<rect data-area="flat" x="${CX0}" y="${f2(mY(flat))}" width="${CX1 - CX0}" height="${f2(mY(0) - mY(flat))}" fill="${C('#c3c2b7')}"/>`,
      `<polygon data-area="acts" points="${grid.map((m) => `${f2(px(m))},${f2(mY(memP(m).total))}`).join(' ')} ${f2(px(PLB))},${f2(mY(flat))} ${f2(px(PMIN))},${f2(mY(flat))}" fill="${C('#eda100')}"/>`);
    for (const m of PMS) P.push(`<circle data-mm="${m}" cx="${f2(px(m))}" cy="${f2(mY(memP(m).total))}" r="3.5" fill="${C('#eda100')}" stroke="${C('#fcfcfb')}" stroke-width="1"/>`);
    P.push(`<line data-cap x1="${CX0}" y1="${f2(mY(CAP))}" x2="${CX1}" y2="${f2(mY(CAP))}" stroke="${C('#d03b3b')}" stroke-width="1.5"/>`,
      `<text x="${f2(px(36))}" y="${f2(mY(150 * 2 ** 30))}" text-anchor="middle" style="fill:${C('#d03b3b')}">doesn't fit</text>`,
      `<text x="${f2(px(miss.m))}" y="${f2(mY(miss.total) - 8)}" text-anchor="middle" style="fill:${C('#d03b3b')}">${gib(miss.total)}</text>`);
    const ml = (y, t, cls = '', fill = '') => `<text${cls ? ` class="${cls}"` : ''} x="${CX1 + 6}" y="${f2(y)}"${fill ? ` style="fill:${C(fill)}"` : ''}>${t}</text>`;
    P.push(ml(mY(CAP) + 4, 'H800 80 GiB', 'hd', '#d03b3b'), `<text class="dims" x="${CX1 + 6}" y="${f2(mY(CAP) + 16)}">less what isn't counted</text>`,
      ml(mY(CAP) + 30, `memory ${gib(r60.total)}`, 'hd'), ml(mY(CAP) + 44, 'saved activations'), ml(MY1 - 2, 'shards'));
    P.push(`<line x1="${CX0}" y1="${SY}" x2="${CX1}" y2="${SY}" stroke="${C('#aba89f')}"/>`);
    for (const m of [16, 20, 30, 40, 50, 60]) {
      const x = f2(px(m));
      P.push(`<line x1="${x}" y1="${SY - 5}" x2="${x}" y2="${SY}" stroke="${C('#aba89f')}"/><text x="${x}" y="${SY - 9}" text-anchor="middle">${m}</text>`);
    }
    P.push(`<text class="dims" x="${CX1 + 14}" y="${SY - 9}">microbatches, m</text>`,
      `<text class="dims" x="${(CX0 + CX1) / 2}" y="${SY + 17}" text-anchor="middle">each pipeline's ${PLB} sequences split m ways</text>`);
    P.push(guideSvg('#d03b3b', true));
    this._chart.innerHTML = `<svg width="${CW}" height="${CH}" viewBox="0 0 ${CW} ${CH}" role="img" aria-label="The idle share of the pipelined step against microbatches per step, ${PMIN} to ${PLB}, on the same scales as the chart above. Comms — EP's all-to-alls, PP's sends and ZeRO-1's once-per-step sync — total ${f2(q60.comms)} s, under compute at ${f2(yc)} s at every microbatch count, so none are exposed. What sits idle is the pipeline bubble. Under DualPipeV it's (PP − 1) ÷ (3m + PP − 1) of the step: ${pc(atP(PMS[0]))}% at ${PMS[0]} microbatches, ${pc(atP(PMS[1]))}% at ${PMS[1]}, ${pc(q60)}% at ${PLB}, one sequence each, the most the batch allows. A dashed curve shows plain 1F1B, three times the idle compute, (PP − 1) ÷ (m + PP − 1) of the step: ${pc1(atP(PMS[0]))}%, ${pc1(atP(PMS[1]))}% and ${pc1(q60)}% at the same counts, with its comms still assumed hidden, though with no paired microbatch to overlap a real 1F1B would expose the all-to-alls. Below, the busiest GPU's peak memory falls like one over the microbatch count against the H800's soft 80 GiB line: ${miss.m} microbatches is over, at ${gib(miss.total)}, and only ${PLB} is under, at ${gib(r60.total)}.">${P.join('')}</svg>`;
  }
  _pt(m) { const q = atP(m); return { x: px(m), y: cy(q.bubble / q.step), y1: cy(q.bubble1 / q.step1), b: memP(m).total }; }
  _mOf(c) { return +(c.dataset.m ?? c.dataset.m1 ?? c.dataset.mm); }
  _snap(x) { const u = PMIN + (PLB - PMIN) * (x - CX0) / (CX1 - CX0); return PMS.reduce((a, b) => (Math.abs(b - u) < Math.abs(a - u) ? b : a)); }
  _text(m) {
    const q = atP(m), r = memP(m), sec = (v) => f2(v) + ' s', n = PLB / m;
    return `${m} microbatches of ${n} sequence${n > 1 ? 's' : ''}\nbubble ${(100 * q.bubble / q.step).toFixed(1)}% of the step: ${sec(q.bubble)} of ${sec(q.step)}`
      + `\n= ${PPR - 1} ÷ (3 × ${m}) of compute ${sec(q.comp)}; comms ${sec(q.comms)} stay under it`
      + `\n1F1B: bubble ${(100 * q.bubble1 / q.step1).toFixed(1)}% of ${sec(q.step1)}, ${PPR - 1} ÷ ${m} of compute (comms assumed hidden)`
      + `\nmemory ${gib(r.total)} on rank ${MEMP.rank}: ${r.total > CAP ? `${gib(r.total - CAP)} over` : `${gib(CAP - r.total)} under`} 80 GiB`;
  }
}
if (typeof customElements !== 'undefined' && !customElements.get('dsv3-ppcurve')) customElements.define('dsv3-ppcurve', Dsv3Ppcurve);
