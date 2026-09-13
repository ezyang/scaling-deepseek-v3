// The FSDP question, before any parallelism is chosen: the model does not fit
// on one GPU, so its parameters are spread over G GPUs and either GATHERED
// back each step (ZeRO/FSDP) or the work is partitioned instead. This module
// prices the gather route at speed of light — raw link bandwidth, FP8 peak,
// one weight all-gather per step (generous: nothing can hold the gathered
// weights) — across every sharding degree G, next to the memory each GPU
// then holds. <dsv3-fsdp> draws the sweep: rows = G, memory left, sync right.
import { DSV3, HARDWARE, modelFlopsPerToken } from './model.js';
import { PARAMS } from './params.js';
import { C } from './theme.js';
import { knobCss, fmtBytes } from './ui.js';

// DeepSeek-V3's own recipe (§3.3): bf16 weights, fp32 gradients, fp32 master
// weights + bf16 AdamW moments — 14 bytes per parameter of state
export const STATE_BYTES = { weights: 2, grads: 4, optim: 8 };
const COMPS = ['weights', 'grads', 'optim'];
const COMP_C = { weights: '#2a78d6', grads: '#eb6834', optim: '#1baf7a' };   // the byte-component family
const SHARDED = { 1: ['optim'], 2: ['optim', 'grads'], 3: ['optim', 'grads', 'weights'] };
// DeepSeek-V3 paper §1: 2.788M H800 GPU-hours for 14.8T tokens
export const REALIZED_TOK_S = 14.8e12 / (2.788e6 * 3600);

export const FSDP_DEFAULTS = { hw: 'h800', gpus: 2048, gbs: 15360, seq: 4096, zero: 3, model: 'dsv3' };

// one row of the sweep: sharding degree G in a cluster of cfg.gpus
export function fsdpRow(cfg, G) {
  const hw = HARDWARE[cfg.hw];
  // the dense counterfactual: a model with DeepSeek's ACTIVE parameter count
  // does the same FLOPs per token — only the bytes at rest change
  const P = cfg.model === 'dense' ? PARAMS.activeTotal : PARAMS.total;
  const W = cfg.gpus, R = W / G;                    // R replicas of every shard
  const sh = SHARDED[cfg.zero];
  const mem = {};
  for (const k of COMPS) mem[k] = P * STATE_BYTES[k] * (sh.includes(k) ? 1 / G : 1);
  const memTotal = COMPS.reduce((t, k) => t + mem[k], 0);
  // a group of n GPUs sits inside one node (NVLink) or spans nodes (one NIC per GPU)
  const link = (n) => n <= hw.domain ? hw.nvl : hw.nic;
  // gradient sync, fp32: reduce-scatter over the shard group, then all-reduce
  // each shard across its replicas (ring: (n−1)/n of the bytes per GPU)
  const rs = G > 1 ? P * 4 * (G - 1) / G / link(G) : 0;
  const ar = R > 1 ? 2 * (P / G) * 4 * (R - 1) / R / link(W) : 0;
  // the updated weights come back with ONE all-gather per step — ZeRO-1/2
  // after the optimizer step, ZeRO-3 as if the gathered model could be kept
  const ag = G > 1 ? P * 2 * (G - 1) / G / link(G) : 0;
  const tokens = cfg.gbs * cfg.seq / W;
  const comp = tokens * modelFlopsPerToken(DSV3, cfg.seq) / hw.flops.fp8;
  return { G, P, mem, memTotal, rs, ar, ag, sync: rs + ar + ag, tokens, comp,
    realized: tokens / REALIZED_TOK_S, fits: memTotal <= hw.memGB * 2 ** 30, crossNode: G > hw.domain };
}

export function fsdpSweep(cfg) {
  cfg = { ...FSDP_DEFAULTS, ...cfg };
  const rows = [];
  for (let G = 1; G <= cfg.gpus; G *= 2) rows.push(fsdpRow(cfg, G));
  return { cfg, rows, firstFit: rows.find((r) => r.fits) ?? null };
}

// ---- the widget ------------------------------------------------------------
const GPUS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
const GBSS = [1920, 3840, 7680, 15360, 30720];
const MAXROWS = Math.log2(GPUS[GPUS.length - 1]) + 1;   // rows reserved for the largest cluster: no reflow
const W = 900, GUT = 78, PW = 300, GAP = 50, ROWH = 17, TOP = 44;   // node captions live right of the time panel
const MX0 = GUT, TX0 = GUT + PW + GAP;
const MEM = [32, 44], TIM = [-2, 8];                     // log₂ spans: 4 GiB…16 TiB · 0.25 s…256 s
const HB = TOP + MAXROWS * ROWH + 30;
const fmtS = (s) => (s >= 100 ? s.toFixed(0) : s >= 10 ? s.toFixed(1) : s.toFixed(2)) + ' s';
const fitEase = (p) => 1 - (1 - p) ** 3;
const lerp = (a, b, t) => a + (b - a) * t;

const CSS = `
dsv3-fsdp { display: block; margin: 14px 0 26px; }
.fs { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-0b0b0b);
  border: 1px solid var(--c-e1e0d9); border-radius: 6px; background: var(--c-fcfcfb); padding: 8px 10px;
  width: max-content; max-width: 100%; box-sizing: border-box; }
.fs .top { display: flex; align-items: stretch; gap: 10px; padding-bottom: 8px; flex-wrap: wrap; }
${knobCss('.fs .top')}
.fs svg { display: block; }
.fs .dims { font: 9px system-ui; fill: var(--c-898781); }
.fs .glab { font: 10.5px ui-monospace, Menlo, monospace; }
.fs .ro { font-size: 11.5px; color: var(--c-52514e); min-height: 34px; margin-top: 4px; max-width: ${W}px; line-height: 1.45; }
.fs .ro b { color: var(--c-0b0b0b); font-weight: 600; }
`;

// node-importable (goldens/sanity read the model): the element class only exists in a browser
class Dsv3Fsdp extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    const A = (k) => this.getAttribute(k);
    const st = this.id ? readState('f:' + this.id) : null;
    this.cfg = { ...FSDP_DEFAULTS,
      hw: A('hw') ?? 'h800',
      gpus: +(st?.gpus ?? A('gpus') ?? 2048), gbs: +(st?.gbs ?? A('gbs') ?? 15360),
      zero: +(st?.zero ?? A('zero') ?? 3), model: st?.model ?? A('model') ?? 'dsv3' };
    const style = document.createElement('style'); style.textContent = CSS;
    this._root = el('div', 'fs');
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
    // −/select/+ stepper over a fixed option list (the house pattern)
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
        x.onclick = () => this._set(key, typeof this.cfg[key] === 'number' ? +v : v);
        eg.append(x);
      }
      this._ui[key] = { eg };
      return eg;
    };
    this._ui = {};
    const g1 = grp('cluster'); g1.append(row(txt('GPUs'), stepper('gpus', GPUS)));
    const g2 = grp('batch'); g2.append(row(txt('seqs/step'), stepper('gbs', GBSS)));
    const g3 = grp('what is sharded'); g3.append(row(seg('zero', [[1, 'ZeRO-1 optimizer'], [2, 'ZeRO-2 + grads'], [3, 'ZeRO-3 + weights']])));
    const g4 = grp('model'); g4.append(row(seg('model', [['dsv3', 'DeepSeek-V3 · 671B'], ['dense', 'dense · 37B']])));
    this._top.append(g1, g2, g3, g4);
    this._syncKnobs();
  }
  _syncKnobs() {
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
    if (this.id) writeState('f:' + this.id, { gpus: this.cfg.gpus, gbs: this.cfg.gbs, zero: this.cfg.zero, model: this.cfg.model });
    this._animateTo(this._layout());
  }
  // pixel-space layout: every row reserved (rows past the cluster fade out)
  _layout() {
    const S = fsdpSweep(this.cfg);
    const hw = HARDWARE[this.cfg.hw];
    const mx = (b) => MX0 + Math.min(1, Math.max(0, (Math.log2(Math.max(b, 1)) - MEM[0]) / (MEM[1] - MEM[0]))) * PW;
    const tx = (s) => TX0 + Math.min(1, Math.max(0, (Math.log2(Math.max(s, 1e-9)) - TIM[0]) / (TIM[1] - TIM[0]))) * PW;
    const rows = [];
    for (let i = 0; i < MAXROWS; i++) {
      const r = S.rows[i];
      const y = TOP + i * ROWH;
      if (!r) { rows.push({ G: 2 ** i, y, op: 0, msegs: [], tsegs: [] }); continue; }
      let cum = 0; const msegs = COMPS.map((k) => { const s = { k, x0: mx(cum), x1: mx(cum + r.mem[k]), c: COMP_C[k], v: r.mem[k] }; cum += r.mem[k]; return s; });
      const tsegs = [{ k: 'grad', x0: tx(1e-9), x1: tx(r.rs + r.ar), c: '#6b5bd2', v: r.rs + r.ar },
        { k: 'ag', x0: tx(r.rs + r.ar), x1: tx(r.sync), c: '#f3f1fb', stroke: '#6b5bd2', v: r.ag }];   // the .comm box style
      rows.push({ G: r.G, y, op: 1, msegs, tsegs, mem: r.memTotal, sync: r.sync, fits: r.fits, first: r === S.firstFit,
        cross: r.crossNode, row: r });
    }
    const r0 = S.rows[0];
    return { rows, capX: mx(hw.memGB * 2 ** 30), compX: tx(r0.comp), realX: tx(r0.realized), comp: r0.comp, realized: r0.realized,
      nodeY: TOP + (Math.log2(hw.domain) + 1) * ROWH - 4, S, hw };
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
    const dims = `class="dims"`;
    // headers + legends
    B.push(`<text ${dims} x="${MX0}" y="9" font-weight="600" fill="${C('#0b0b0b')}">memory per GPU</text>`);
    let lx = MX0 + 92;
    for (const k of COMPS) { B.push(`<rect x="${lx}" y="3" width="7" height="7" fill="${C(COMP_C[k])}"/><text ${dims} x="${lx + 10}" y="9">${k}</text>`); lx += 10 + k.length * 5.2 + 12; }
    B.push(`<text ${dims} x="${TX0}" y="9" font-weight="600" fill="${C('#0b0b0b')}">sync per step</text>`);
    lx = TX0 + 82;
    for (const [c, lab, st] of [['#6b5bd2', 'gradient reduce'], ['#f3f1fb', 'weight all-gather', '#6b5bd2']]) {
      B.push(`<rect x="${lx}" y="3" width="7" height="7" fill="${C(c)}"${st ? ` stroke="${C(st)}"` : ''}/><text ${dims} x="${lx + 10}" y="9">${lab}</text>`); lx += 10 + lab.length * 5 + 12;
    }
    const aY = TOP + MAXROWS * ROWH + 2;
    // ×2 grids
    for (let e = MEM[0]; e <= MEM[1]; e++) { const x = MX0 + (e - MEM[0]) / (MEM[1] - MEM[0]) * PW;
      B.push(`<line x1="${f1(x)}" y1="${TOP - 4}" x2="${f1(x)}" y2="${aY}" stroke="${C('#e1e0d9')}" stroke-width="1"/>`); }
    for (let e = TIM[0]; e <= TIM[1]; e++) { const x = TX0 + (e - TIM[0]) / (TIM[1] - TIM[0]) * PW;
      B.push(`<line x1="${f1(x)}" y1="${TOP - 4}" x2="${f1(x)}" y2="${aY}" stroke="${C('#e1e0d9')}" stroke-width="1"/>`); }
    for (const [e, lab] of [[33, '8 GiB'], [36, '64 GiB'], [40, '1 TiB'], [43, '8 TiB']])
      B.push(`<text ${dims} x="${f1(MX0 + (e - MEM[0]) / (MEM[1] - MEM[0]) * PW + 3)}" y="${aY + 9}">${lab}</text>`);
    for (const [e, lab] of [[0, '1 s'], [2, '4 s'], [4, '16 s'], [6, '64 s'], [8, '256 s']])
      B.push(`<text ${dims} x="${f1(TX0 + (e - TIM[0]) / (TIM[1] - TIM[0]) * PW + 3)}" y="${aY + 9}">${lab}</text>`);
    // the infeasible region (past capacity) is SHADED; the reference times are lines
    B.push(`<rect data-cap x="${f1(L.capX)}" y="${TOP - 4}" width="${f1(MX0 + PW - L.capX)}" height="${aY - TOP + 4}" fill="${C('#0b0b0b')}" opacity="0.07"/>`);
    B.push(`<text ${dims} x="${f1(L.capX)}" y="${TOP - 7}" text-anchor="middle">${L.hw.memGB} GiB</text>`);
    B.push(`<line data-comp="${L.comp}" x1="${f1(L.compX)}" y1="${TOP - 4}" x2="${f1(L.compX)}" y2="${aY}" stroke="${C('#0b0b0b')}" stroke-width="1"/>`);
    B.push(`<text ${dims} x="${f1(L.compX)}" y="${TOP - 7}" text-anchor="middle" fill="${C('#0b0b0b')}">speed of light ${fmtS(L.comp)}</text>`);
    B.push(`<line data-real="${L.realized}" x1="${f1(L.realX)}" y1="${TOP - 4}" x2="${f1(L.realX)}" y2="${aY}" stroke="${C('#898781')}" stroke-width="1" stroke-dasharray="3 2"/>`);
    B.push(`<text ${dims} x="${f1(L.realX)}" y="${TOP - 16}" text-anchor="middle">DeepSeek's realized step ${fmtS(L.realized)}</text>`);
    // node boundary
    B.push(`<line x1="${GUT - 6}" y1="${f1(L.nodeY)}" x2="${TX0 + PW}" y2="${f1(L.nodeY)}" stroke="${C('#898781')}" stroke-width="1" stroke-dasharray="2 3"/>`);
    B.push(`<text ${dims} x="${TX0 + PW + 12}" y="${f1(L.nodeY - 3)}">↑ one node · NVLink ${L.hw.nvl / 1e9} GB/s</text>`);
    B.push(`<text ${dims} x="${TX0 + PW + 12}" y="${f1(L.nodeY + 10)}">↓ across nodes · IB ${L.hw.nic / 1e9} GB/s</text>`);
    for (const r of L.rows) {
      const op = r.op < 0.999 ? ` opacity="${r.op.toFixed(3)}"` : '';
      if (r.first && r.op > 0.5) B.push(`<rect data-first="${r.G}" x="0" y="${f1(r.y - 3)}" width="${W}" height="${ROWH}" fill="${C('#fff8ea')}"/>`);
      B.push(`<g data-row="${r.G}"${op}>`);
      B.push(`<text class="glab" x="${GUT - 8}" y="${f1(r.y + 7)}" text-anchor="end" fill="${C(r.fits ? '#0b0b0b' : '#898781')}"${r.fits ? ' font-weight="600"' : ''}>G = ${r.G}</text>`);
      for (const s of r.msegs) if (s.x1 - s.x0 > 0.3)
        B.push(`<rect data-mem="${s.k}" data-true="${s.v}" x="${f1(s.x0)}" y="${f1(r.y)}" width="${f1(Math.max(0.5, s.x1 - s.x0))}" height="8" fill="${C(s.c)}"/>`);
      if (r.mem != null) B.push(`<text ${dims} data-memval="${r.mem}" x="${f1(Math.max(...r.msegs.map((s) => s.x1)) + 4)}" y="${f1(r.y + 7)}">${fmtBytes(r.mem)}</text>`);
      for (const s of r.tsegs) if (s.x1 - s.x0 > 0.3)
        B.push(`<rect data-time="${s.k}" data-true="${s.v}" x="${f1(s.x0)}" y="${f1(r.y)}" width="${f1(Math.max(0.5, s.x1 - s.x0))}" height="8" fill="${C(s.c)}"${s.stroke ? ` stroke="${C(s.stroke)}" stroke-width="1"` : ''}/>`);
      if (r.mem != null) B.push(`<text ${dims} data-syncval="${r.sync}" x="${f1(Math.max(...r.tsegs.map((s) => s.x1)) + 4)}" y="${f1(r.y + 7)}">${fmtS(r.sync)}</text>`);
      B.push(`<rect data-hit="${r.G}" x="0" y="${f1(r.y - 3)}" width="${W}" height="${ROWH}" fill="transparent"/>`);
      B.push('</g>');
    }
    this._chart.innerHTML = `<svg width="${W}" height="${HB}" viewBox="0 0 ${W} ${HB}">${B.join('')}</svg>`;
    const svg = this._chart.firstChild;
    svg.onmouseover = (e) => { const g = e.target.closest('g[data-row]'); if (g) this._readout(L, +g.dataset.row); };
    svg.onmouseleave = () => this._readout(L, null);
    this._readout(L, this._hover ?? null);
  }
  _readout(L, G) {
    this._hover = G;
    const S = L.S, cfg = S.cfg, hw = L.hw;
    const r = G != null && S.rows.find((q) => q.G === G);
    const x = (a, b) => `×${(a / b).toFixed(1)}`;
    if (r) {
      this._ro.innerHTML = `<b>G = ${r.G}</b> (${(cfg.gpus / r.G).toLocaleString('en-US')} replicas): ${fmtBytes(r.memTotal)}/GPU`
        + ` = weights ${fmtBytes(r.mem.weights)} + grads ${fmtBytes(r.mem.grads)} + optimizer ${fmtBytes(r.mem.optim)}`
        + ` → ${r.fits ? 'fits' : 'does not fit'} in ${hw.memGB} GiB. Sync ${fmtS(r.sync)}/step = gradient reduce ${fmtS(r.rs + r.ar)}`
        + ` + weight all-gather ${fmtS(r.ag)}, over ${r.crossNode ? 'InfiniBand' : 'NVLink'}: ${x(r.sync, r.comp)} speed-of-light compute, ${x(r.sync, r.realized)} the realized step.`;
      return;
    }
    const z = `ZeRO-${cfg.zero}`;
    const ff = S.firstFit;
    if (!ff) {
      const last = S.rows[S.rows.length - 1];
      this._ro.innerHTML = `<b>${z} never fits</b> on ${cfg.gpus.toLocaleString('en-US')} GPUs: even at G = ${last.G} each GPU holds ${fmtBytes(last.memTotal)} — the replicated state alone exceeds ${hw.memGB} GiB.`;
      return;
    }
    this._ro.innerHTML = `<b>${z} first fits at G = ${ff.G}</b> (${fmtBytes(ff.memTotal)}/GPU), ${ff.crossNode ? `${ff.G / hw.domain} nodes, so the sync runs over InfiniBand` : 'inside one node, on NVLink'}:`
      + ` ${fmtS(ff.sync)}/step against ${fmtS(ff.comp)} of speed-of-light compute (${x(ff.sync, ff.comp)})`
      + ` and ${fmtS(ff.realized)} realized (${x(ff.sync, ff.realized)}).`;
  }
}
function blend(A, B, t) {
  if (!A || t >= 1) return B;
  const rowsA = new Map(A.rows.map((r) => [r.G, r]));
  const seg = (a, b) => ({ ...b, x0: lerp(a.x0, b.x0, t), x1: lerp(a.x1, b.x1, t) });
  const segs = (as, bs) => bs.map((b) => seg(as.find((a) => a.k === b.k) ?? { x0: b.x0, x1: b.x0 }, b));
  const rows = B.rows.map((b) => { const a = rowsA.get(b.G) ?? { ...b, op: 0 };
    if (!b.op && a.op) return { ...a, op: lerp(a.op, 0, t) };   // a row leaving the cluster fades out in place
    return { ...b, op: lerp(a.op, b.op, t), msegs: segs(a.msegs, b.msegs), tsegs: segs(a.tsegs, b.tsegs) }; });
  return { ...B, rows, capX: lerp(A.capX, B.capX, t), compX: lerp(A.compX, B.compX, t), realX: lerp(A.realX, B.realX, t) };
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
if (typeof customElements !== 'undefined' && !customElements.get('dsv3-fsdp')) customElements.define('dsv3-fsdp', Dsv3Fsdp);
