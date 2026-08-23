// Canvas trace viewer, perfetto-flavored: WASD navigation, wheel zoom/pan,
// hover tooltips, click select, M marks, F focuses. Embeddable many times per
// page via the <dsv3-trace> custom element or the TraceViewer class.

import { fmtUs, fmtNum, DSV3 } from './model.js';
import { simulate, LEVELS, defaultConfig } from './sim.js';
import { memoryUsage, resolveMatmuls, MATMULS, RECIPES } from './memory.js';
import { blockGraph, analyze, RECOMPUTE_PRESETS } from './blockgraph.js';
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

// ---- <dsv3-memory> custom element ---------------------------------------------
// High-watermark comparison: one stacked bar of per-GPU memory at the 1F1B
// steady-state watermark (worst pipeline stage) per candidate config, against
// the hardware's capacity line. Activations are broken down mla/moe/residual/
// logits as lightness steps of one hue.
//   <dsv3-memory configs='[{"label":"PP16 EP64","pp":16,...}, ...]'></dsv3-memory>
const MEM_PARTS = [
  ['weights', (w) => w.weights, '#2a78d6', '#fff'],
  ['weights (embed/head)', (w) => w.vocab ?? 0, '#8db4e4', '#0b0b0b'],
  ['grads', (w) => w.grads, '#eb6834', '#fff'],
  ['grads (embed/head)', (w) => w.vocabGrads ?? 0, '#f5a888', '#0b0b0b'],
  ['optimizer', (w) => w.optimizer, '#1baf7a', '#0b0b0b'],
  ['optimizer (embed/head)', (w) => w.vocabOpt ?? 0, '#8fdcbe', '#0b0b0b'],
  ['act·mla', (w) => w.act.mla, '#a06a00', '#fff'],
  ['act·moe', (w) => w.act.moe, '#eda100', '#0b0b0b'],
  ['act·residual', (w) => w.act.residual, '#f5c65e', '#0b0b0b'],
  ['act·logits', (w) => w.act.logits, '#fbe1a4', '#0b0b0b'],
  ['buffers', (w) => w.buffers, '#e87ba4', '#0b0b0b'],
];
const MEM_CSS = `
.mv { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: #0b0b0b;
  border: 1px solid #e1e0d9; border-radius: 6px; background: #fcfcfb; padding: 8px 10px;
  position: relative; }
.mv-tip { ${TIP_CARD} font-size: 11px; max-width: 320px; z-index: 5; line-height: 1.45; white-space: nowrap; }
.mv-tip b { color: #0b0b0b; }
.mv-legend { display: flex; gap: 12px; flex-wrap: wrap; color: #52514e; font-size: 11px; padding-bottom: 8px; }
.mv-legend span { display: inline-flex; align-items: center; gap: 4px; }
.mv-legend i { width: 9px; height: 9px; border-radius: 2px; display: inline-block; }
.mv-row { display: grid; grid-template-columns: 200px 1fr 150px; gap: 10px; align-items: center; padding: 4px 0; }
.mv-label b { display: block; }
.mv-label, .mv-verdict { font-size: 12px; }
.mv-label span { color: #52514e; font-size: 11px; }
.mv-bar { position: relative; height: 22px; background: #f3f2ee; border-radius: 3px; }
.mv-seg { position: absolute; top: 0; height: 22px; border-radius: 2px; overflow: hidden;
  font-size: 10px; line-height: 22px; text-indent: 4px; white-space: nowrap; }
.mv-cap { position: absolute; top: -4px; bottom: -4px; width: 0; border-left: 2px dashed #0b0b0b; }
.mv-cap i { position: absolute; top: -2px; left: 3px; font-style: normal; font-size: 10px; color: #52514e; white-space: nowrap; }
.mv-clip { position: absolute; right: 0; top: 0; height: 22px; width: 14px;
  background: repeating-linear-gradient(-55deg, #fcfcfb, #fcfcfb 3px, #d03b3b 3px, #d03b3b 5px); }
.mv-verdict b { font-size: 12px; }
.mv-verdict .ok { color: #006300; } .mv-verdict .no { color: #d03b3b; }
`;
export class Dsv3Memory extends HTMLElement {
  static observedAttributes = ['configs'];
  connectedCallback() {
    if (this._pendingPatch) { const p = this._pendingPatch; this._pendingPatch = null; this.applyPatch(p); }
    else this.render();
  }
  attributeChangedCallback() {
    if (this._selfSet) { if (this.isConnected) this.render(); return; }
    // external set (page script / author): this becomes the new authored base
    this._authored = null;
    if (this._patch) this.applyPatch({});
    else if (this.isConnected) this.render();
  }
  applyPatch(patch) {
    this._patch = { ...(this._patch ?? {}), ...patch };
    this._authored ??= JSON.parse(this.getAttribute('configs') ?? '[]');
    this._selfSet = true;
    this.setAttribute('configs', JSON.stringify(this._authored.map(c => ({ ...this._patch, ...c }))));
    this._selfSet = false;
  }
  render() {
    const configs = JSON.parse(this.getAttribute('configs') ?? '[]');
    const rows = configs.map(({ label, note, ...overrides }) => {
      const cfg = defaultConfig(overrides);
      return { label, note, cfg, mem: memoryUsage(cfg) };
    });
    this.results = rows;
    // rows may target different hardware: shared linear scale, per-row capacity line
    const capMax = Math.max(1, ...rows.map(r => r.mem.capacityGB));
    const scaleMax = capMax * 1.6; // bars clip past this so blowouts stay comparable
    this.innerHTML = '';
    const style = document.createElement('style'); style.textContent = MEM_CSS;
    const root = el('div', 'mv');
    const legend = el('div', 'mv-legend');
    for (const [name, , c] of MEM_PARTS) {
      const s = el('span'); const i = el('i'); i.style.background = c;
      s.append(i, document.createTextNode(name)); legend.append(s);
    }
    root.append(legend);
    const tip = el('div', 'mv-tip');
    for (const r of rows) {
      const row = el('div', 'mv-row');
      const label = el('div', 'mv-label');
      label.innerHTML = `<b>${esc(r.label)}</b><span>${esc(r.note ?? '')}</span>`;
      const bar = el('div', 'mv-bar');
      // a row may pin the displayed pipeline stage (e.g. a reference that models
      // a mid rank); default is the worst stage
      const w = r.cfg.dispStage != null ? (r.mem.perStage[r.cfg.dispStage] ?? r.mem.worst) : r.mem.worst;
      const segs = []; // cumulative GiB ranges for pointer -> segment lookup
      let x = 0;
      for (const [name, get, c, ink] of MEM_PARTS) {
        const gb = get(w);
        if (gb < 0.05) continue;
        segs.push({ name, gb, from: x, to: x + gb });
        const seg = el('div', 'mv-seg');
        const x1 = Math.min((x + gb) / scaleMax * 100, 100);
        seg.style.left = (x / scaleMax * 100) + '%';
        seg.style.width = 'calc(' + (x1 - x / scaleMax * 100) + '% - 2px)'; // 2px surface gap
        seg.style.background = c; seg.style.color = ink;
        const wPct = x1 - x / scaleMax * 100;
        if (wPct > 18) seg.textContent = `${name} ${gb.toFixed(0)}`;
        else if (wPct > 6) seg.textContent = gb.toFixed(0);
        bar.append(seg);
        x += gb;
        if (x >= scaleMax) break;
      }
      // hover tooltip: track the pointer in GiB space so even sub-pixel segments resolve
      bar.addEventListener('mousemove', (e) => {
        // clientX-based: offsetX would be relative to the hovered child segment
        const gbAt = (e.clientX - bar.getBoundingClientRect().left) / bar.clientWidth * scaleMax;
        const s = segs.find(g => gbAt >= g.from && gbAt < g.to);
        if (!s) { tip.style.display = 'none'; return; }
        tip.innerHTML = `<b>${esc(s.name)}</b> ${s.gb.toFixed(1)} GiB · ${(s.gb / w.total * 100).toFixed(1)}% of total<br>` +
          `worst stage pp${w.stage} · ${w.inFlight} microbatch(es) in flight · total ${w.total.toFixed(0)}/${r.mem.capacityGB} GiB`;
        tip.style.display = 'block';
        // cursor-relative in both axes so the box tracks smoothly across bars
        const rootBox = root.getBoundingClientRect();
        tip.style.left = Math.min(e.clientX - rootBox.left + 12,
          rootBox.width - tip.offsetWidth - 6) + 'px';
        tip.style.top = (e.clientY - rootBox.top + 16) + 'px';
      });
      bar.addEventListener('mouseleave', () => { tip.style.display = 'none'; });
      const capGB = r.mem.capacityGB;
      const cap = el('div', 'mv-cap');
      cap.style.left = (capGB / scaleMax * 100) + '%';
      cap.innerHTML = `<i>${capGB} GiB</i>`;
      bar.append(cap);
      if (w.total > scaleMax) bar.append(el('div', 'mv-clip'));
      const verdict = el('div', 'mv-verdict');
      verdict.innerHTML = r.mem.fits
        ? `<b class="ok">✓ fits</b> ${w.total.toFixed(0)}/${capGB} GiB (pp${w.stage})`
        : `<b class="no">✗ ${(w.total / capGB).toFixed(w.total / capGB > 20 ? 0 : 1)}× over</b> ${fmtGB(w.total)}/${capGB} GiB`;
      row.append(label, bar, verdict);
      root.append(row);
    }
    root.append(tip);
    this.append(style, root);
  }
}
function fmtGB(gb) { return gb >= 1024 ? (gb / 1024).toFixed(1) + ' TiB' : gb.toFixed(0) + ' GiB'; }
if (typeof customElements !== 'undefined' && !customElements.get('dsv3-memory')) {
  customElements.define('dsv3-memory', Dsv3Memory);
}

// ---- shared linking -------------------------------------------------------------
// Patch the config of linked widgets by element id: <dsv3-memory> gets every
// row in `configs` patched; <dsv3-trace> gets its `config` patched. This is
// how the schematic and control strips drive memory and profile together.
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
    if (t.tagName === 'DSV3-MEMORY') {
      // memory rows: patches supply the BASELINE; keys a row declares stay pinned
      // (so linked controls drive shared knobs while each row keeps its deltas)
      if (t.applyPatch) t.applyPatch(patch);
      else { t._pendingPatch = { ...(t._pendingPatch ?? {}), ...patch }; }
    } else {
      const cfg = JSON.parse(t.getAttribute('config') ?? '{}');
      t.setAttribute('config', JSON.stringify({ ...cfg, ...patch }));
    }
  }
}

// ---- <dsv3-layer> custom element ------------------------------------------------
// Top-down SVG schematic of one DSv3 transformer block (plus the head).
// Every matmul carries a dtype <select>; the chosen per-matmul precisions are
// pushed to the widgets named in `for="id1 id2"` (memory now, rooflines later).
const DT_STYLE = { bf16: '#52514e', mxfp8: '#2a78d6', fp32: '#9c3a96' };
const LAYER_CSS = `
dsv3-layer { display: block; margin: 14px 0 26px; }
.lv { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: #0b0b0b;
  border: 1px solid #e1e0d9; border-radius: 6px; background: #fcfcfb; padding: 10px 12px; position: relative; }
.lv-tip { ${TIP_CARD} font-size: 11.5px; max-width: 360px; z-index: 7; line-height: 1.5; white-space: pre-line; }
.lv-tip.pinned { border-color: #eda100; box-shadow: 0 2px 10px rgba(237,161,0,0.3); }
.lv-head { display: flex; align-items: center; gap: 8px; padding-bottom: 6px; color: #52514e; flex-wrap: wrap; }
.lv-head select { font: 12px system-ui; padding: 2px 6px; border: 1px solid #c3c2b7; border-radius: 4px; background: #fff; }
.lv svg { display: block; margin: 0 auto; max-width: 100%; height: auto; }
.lv .wire { stroke: #898781; stroke-width: 1.2; fill: none; }
.lv .box { fill: #fff; stroke: #c3c2b7; }
.lv .op { fill: #f3f2ee; stroke: #e1e0d9; }
.lv .comm { fill: #f3f1fb; stroke: #6b5bd2; }
.lv .res { fill: #fcfcfb; stroke: #c3c2b7; stroke-dasharray: 3 2; }
.lv .grp { fill: none; stroke: #e1e0d9; }
.lv .name { font: 600 11px system-ui; fill: #0b0b0b; }
.lv .dims { font: 9px system-ui; fill: #898781; }
.lv .oplabel { font: 10.5px system-ui; fill: #52514e; }
.lv .grplabel { font: italic 10px system-ui; fill: #898781; }
.lv .plus { font: 600 12px system-ui; fill: #52514e; }
.lv select.dt { font: 600 10px system-ui; width: 100%; height: 20px; border: 1px solid #c3c2b7;
  border-radius: 3px; background: #fff; }
.lv button.st { display: block; width: 100%; height: 18px; font: 10px system-ui; border-radius: 3px;
  cursor: pointer; text-align: left; padding: 0 5px; overflow: hidden; white-space: nowrap; }
.lv .st-save { background: #fff8ea; border: 1px solid #eda100; color: #0b0b0b; }
.lv .st-pin { background: #fff8ea; border: 1px solid #eda100; color: #52514e; cursor: default; }
.lv .st-redo { background: #f3f2ee; border: 1px dashed #898781; color: #52514e; }
.lv .st-idle { background: transparent; border: 1px dashed #e1e0d9; color: #898781; }
.lv button.st.mode { width: 24px; padding: 0; text-align: center; height: 20px; }
.lv button.st.dtb { width: 52px; padding: 0; text-align: center; height: 20px; font-weight: 600;
  background: #fff; border: 1px solid #c3c2b7; }
.lv button.st.ktab { width: 100%; height: 24px; font: 600 11px system-ui; text-align: center;
  background: #f3f2ee; border: 1px solid #e1e0d9; color: #898781; border-radius: 6px 6px 0 0; }
.lv button.st.ktab.on { background: #fff; border-color: #c3c2b7; color: #0b0b0b; cursor: default; }
.lv .ktsub { font-weight: 400; font-size: 10px; }
.lv text.tensor { font: 10px system-ui; }
.lv .tsave { fill: #7a5200; font-weight: 600; }
.lv .tdim { fill: #898781; font-weight: 400; }
.lv .micro { fill: #f7f6f1; stroke: #d8d6cb; }
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
    const st = readUrlState(this.urlKey);
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
    // which block variant to draw: the MLA column is identical; only the FFN
    // column differs (kind="dense" pins the dense-FFN variant, default MoE)
    this.kind = st?.kind ?? (this.getAttribute('kind') === 'dense' ? 'dense' : 'moe');
    this.render();
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
    if (write) writeUrlState(this.urlKey, {
      recipe: this.getAttribute('recipe'), matmuls: this.matmuls, marks: this.marks,
      view: this.view, dispLayers: this.dispLayers, dispInflight: this.dispInflight,
      transposed: this.transposed, detail: this.detail, flatDims: this.flatDims,
      kind: this.kind,
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
    const to = this.marks[ids[0]] === false ? true : false;
    for (const id of ids) { if (to) delete this.marks[id]; else this.marks[id] = false; }
    this.render(); this.changed();
  }
  render() {
    this.innerHTML = '';
    const style = document.createElement('style'); style.textContent = LAYER_CSS;
    const root = el('div', 'lv');
    // progressive disclosure: controls="static|marks|dtype|full" gates which
    // controls are rendered (the diagram and its derived annotations always draw)
    const cmode = this.getAttribute('controls') ?? 'full';
    // static = pure structure: save-everything semantics, no quantities
    // (FLOP strips, bytes, grids, dtype tags), no tooltips, minimal caption
    this._ctl = {
      marks: cmode === 'full' || cmode === 'marks',
      dtype: cmode === 'full' || cmode === 'dtype',
      quant: cmode !== 'static',
    };
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
    head.append('DSv3 ', mkKindSel());
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
    preset.onchange = () => {
      if (preset.value === 'custom') return;
      this.setAttribute('recipe', preset.value);
      this.matmuls = resolveMatmuls({ recipe: preset.value });
      this.render(); this.changed();
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
      this.setAttribute('recompute', rsel.value);
      this.marks = { ...RECOMPUTE_PRESETS[rsel.value] };
      this.render(); this.changed();
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
      this.kind = this.getAttribute('kind') === 'dense' ? 'dense' : 'moe';
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
    tcb.onchange = () => { this.transposed = tcb.checked; this.render(); this.changed(); };
    tl.append(tcb, 'fp8ᵀ dual stash');
    if (this._ctl.dtype) head.append(tl);
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
    head.append(dl, mkDimsBtn(), reset);
    const ana = analyze(blockGraph(this.kind, DSV3, this.matmuls, 4096),
      this._ctl.quant ? this.marks : {}, this.transposed);
    if (cmode !== 'static') root.append(head);
    else {
      const mini = el('div', 'lv-head');
      // no kind select when MLA-only (kind-independent) or when the tabs carry the flip
      if (this.getAttribute('only') === 'mla' || this.hasAttribute('kindtabs')) mini.append('sizes:', mkDimsBtn());
      else mini.append('block: ', mkKindSel(), ' · sizes:', mkDimsBtn());
      root.append(mini);
    }
    // dense mode also analyzes the MoE graph, purely for LAYOUT: the dense
    // column reserves whitespace where the routing rows sit, so flipping
    // kinds keeps every surviving element in the same place
    const anaM = this.kind === 'dense'
      ? analyze(blockGraph('moe', DSV3, this.matmuls, 4096), this._ctl.quant ? this.marks : {}, this.transposed)
      : null;
    root.append(this.buildSvg(ana, anaM));
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
            ? `Shared expert${this.hasAttribute('block-only') ? '' : ' + dense MLPs'} follow${this.hasAttribute('block-only') ? 's' : ''} the ffn choices; `
            : `The shared expert${this.hasAttribute('block-only') ? ' shares' : ' and dense MLPs share'} the ffn boxes; `))
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
  }
  buildSvg(ana, anaM = null) {
    const P = [];
    // Two columns (MLA | MoE), head row underneath. The dataflow spine runs
    // down the LEFT of each column; output tensors are annotated on the spine
    // (▣ saved + block grid, ▪ = 4 KiB/token · ↻ recomputed · · not needed).
    // Aux backward artifacts (rstd, lse) exit each box to the RIGHT — always saved.
    // only="mla" / only="ffn" draws a single column (for composed anatomy
    // pages that show each component once); default draws the full block
    const ONLY = this.getAttribute('only');
    const W = 290, C1 = 60, C2 = ONLY === 'ffn' ? 60 : 512;
    const SX1 = C1 + 22, SX2 = C2 + 22, RAIL1 = C1 - 26;
    const WIDTH = ONLY === 'mla' ? C1 + W + 250
      : C2 + W + (this.detail ? 220 : 180); // right margin fits aux labels (+ shared column in detail)
    // dims display: factored (128\u00d7192) or multiplied out (24576)
    const flatten = (s) => {
      if (!this.flatDims || !s) return s;
      return String(s).split('\u2192').map(part => {
        const t = part.trim().replace(/\u00d7/g, '*');
        if (!/^[\d\s+*()]+$/.test(t)) return part.trim();
        try { return String(Function('"use strict";return (' + t + ')')()); } catch { return part.trim(); }
      }).join(' \u2192 ');
    };
    // parameter-count parentheticals on the dims lines; grouped experts
    // follow the dims toggle: factored '(29M \u00d7256)' vs total '(7.5B)'
    const fmtP = (n) => n >= 1e9 ? (n / 1e9).toFixed(1) + 'B'
      : n >= 9.95e6 ? Math.round(n / 1e6) + 'M' : n >= 1e6 ? (n / 1e6).toFixed(1) + 'M'
      : n >= 1e3 ? (n / 1e3).toFixed(1).replace(/\.0$/, '') + 'K' : String(n);
    const PCNT = {
      q_up: DSV3.qRank * DSV3.heads * (DSV3.qkNope + DSV3.qkRope),
      kv_up: DSV3.kvRank * DSV3.heads * (DSV3.qkNope + DSV3.vHead),
      o_proj: DSV3.heads * DSV3.vHead * DSV3.hidden,
      router: DSV3.hidden * DSV3.routedExperts,
      ...(this.kind === 'dense' ? {
        ffn_gate_up: DSV3.hidden * 2 * DSV3.denseInter,
        ffn_down: DSV3.denseInter * DSV3.hidden,
      } : {
        ffn_gate_up: [DSV3.hidden * 2 * DSV3.moeInter, DSV3.routedExperts],
        ffn_down: [DSV3.moeInter * DSV3.hidden, DSV3.routedExperts],
      }),
      lm_head: DSV3.hidden * DSV3.vocab,
    };
    const pstr = (id) => {
      const p = PCNT[id];
      if (!p) return '';
      if (Array.isArray(p)) return this.flatDims ? ` (${fmtP(p[0] * p[1])})` : ` (${fmtP(p[0])} \u00d7${p[1]})`;
      return ` (${fmtP(p)})`;
    };
    const dt = (id) => this.matmuls[id];
    const marks = this._ctl.quant ? this.marks : {};   // static: save everything
    const state = (id) => {
      const n = ana.byId[id];
      if (n.always) return 'pin';
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
    const boxTip = (id, dimsNote) => {
      const n = ana.byId[id];
      const f = n?.flopsTok ? `${fmtNum(n.flopsTok)} FLOP/token = ${FLOP_EXPR[id] ?? ''}` : (FLOP_EXPR[id] ?? '');
      return ` data-tip="${escAttr(f + (dimsNote ? '\n' + dimsNote : ''))}"`;
    };
    // dtype the sim ascribes to a stashed tensor (the dtype of the matmul whose
    // backward reads it — a real degree of freedom, so we surface it)
    const dtOf = (n) => {
      const b = n.outBytes / n.elems;
      return b >= 3.5 ? 'fp32' : b >= 1.7 ? 'bf16' : 'mxfp8';
    };
    const FLOP_ROW = 30;
    const FLOP_UNIT = Math.max(...['qkv_down', 'q_up', 'kv_up', 'attn', 'o_proj', 'router', 'gate_up', 'swiglu', 'ffn_down']
      .filter(id => ana.byId[id])            // dense blocks have no router
      .map(id => flopEq(ana.byId[id].flopsTok, opDt(id)))) / FLOP_ROW;
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
        P.push(`<text class="tensor tredo" x="${x}" y="${y + 8}">↻ ${esc(name0)} — recomputed</text>`);
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
      if (!this._ctl.quant) return 18;                 // one text line, no grid
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
      const n = ana.byId[id];
      if (!n.aux) return;
      const replayed = ana.replayed.has(id); // a replay regenerates its aux

      P.push(`<line class="wire" x1="${x + W}" y1="${yMid}" x2="${x + W + 10}" y2="${yMid}" marker-end="url(#arr)"/>` +
        (replayed
          ? `<text class="tensor tredo" x="${x + W + 14}" y="${yMid + 3}">↻ ${esc(n.aux.name)}</text>`
          : !this._ctl.quant
            ? `<text class="tensor tsave" x="${x + W + 14}" y="${yMid + 3}">← ${esc(n.aux.name)}</text>`
            : `<text class="tensor tsave" x="${x + W + 14}" y="${yMid + 3}">← ${esc(n.aux.name)} · ${fmtMem(n.aux.bytes)} ` +
              `<tspan fill="${DT_STYLE.fp32}">fp32</tspan></text>`));
    };
    // display-only elided kernel (detail view): cheap, no marks, not in the graph
    const DET = this.detail;
    const micro = (label, x, y, w = W, tip, pc = '') => {
      const body = `<rect class="micro" x="${x}" y="${y}" width="${w}" height="18" rx="9"/>` +
        `<text class="microlabel" x="${x + 9}" y="${y + 13}">${label}${pc ? `<tspan class="dims"> ${pc}</tspan>` : ''}</text>`;
      P.push(tip ? `<g data-tip="${escAttr(tip)}">${body}</g>` : body);
      return y + 18;
    };
    const plus = (cx, y) => P.push(`<circle cx="${cx}" cy="${y}" r="9" class="box"/>` +
      `<text class="plus" x="${cx}" y="${y + 4}" text-anchor="middle">+</text>`);
    const grp = (x, y0, y1, label, w = W + 20) => P.push(
      `<rect class="grp" x="${x - 10}" y="${y0}" width="${w}" height="${y1 - y0}" rx="6"/>` +
      `<text class="grplabel" x="${x - 2}" y="${y0 + 11}">${label}</text>`);
    const mmBox = (ids, x, y, markIds, label, dims) => {
      const spec = MATMULS.find(m => m.id === ids[0]);
      P.push(`<g${boxTip((markIds ?? ids)[0], dims ? undefined : spec.dimsNote)}>` +
        `<rect class="box" x="${x}" y="${y}" width="${W}" height="38" rx="4"/>` +
        `<text class="name" x="${x + 8}" y="${y + 13}">${label ?? spec.label}</text>` +
        `<text class="dims" x="${x + 8}" y="${y + 26}">${flatten(dims ?? spec.dims)}${pstr(ids[0])}</text></g>`);
      P.push(modeBtn(markIds ?? ids, x + W - 86, y + 6));
      P.push(dtBtn(ids[0], x + W - 58, y + 6));
      auxOut((markIds ?? ids)[0], x, y + 19);
      flopBlocks(x + 8, y + 30, ana.byId[(markIds ?? ids)[0]]?.flopsTok, dt(ids[0]));
      return y + 38;
    };
    const opNode = (id, label, x, y, cls = 'op', pc = '') => {
      const h2 = cls === 'comm' ? 22 : 27;
      P.push(`<g${boxTip(id)}>` +
        `<rect class="${cls}" x="${x}" y="${y}" width="${W}" height="${h2}" rx="6"/>` +
        `<text class="oplabel" x="${x + 10}" y="${y + 15}">${label}${pc ? `<tspan class="dims"> ${pc}</tspan>` : ''}</text></g>` +
        modeBtn([id], x + W - 30, y + 1));
      auxOut(id, x, y + Math.round(h2 / 2));
      if (cls !== 'comm') flopBlocks(x + 10, y + 19, ana.byId[id]?.flopsTok, 'vector');
      return y + h2;
    };

    // ---- column 1: MLA (skipped in only="ffn" mode) ----
    let y = 14, x1Y = 14, col1End = 44;
    if (ONLY !== 'ffn') {
    P.push(`<text class="oplabel" x="${SX1 + 14}" y="${y}">x — residual stream (7168)</text>`);
    tensorChip(['x0'], SX1 + 170, y - 8);
    const tap1 = y + 6;
    wire(SX1, y + 3, y + 18); y += 18;
    y = opNode('norm1', 'RMSNorm', C1, y, 'op', `(${fmtP(DSV3.hidden)})`);
    let g1;
    y = wireOut(['norm1'], SX1, y); g1 = y + 3; y += 21;
    let bypX = 0;                                // k_rope rail x (set in the MLA fork block)
    {
      const RX = C1 + 150 + 22;
      // the down-projection is two separate GEMMs in production stacks
      // (wq_a | wkv_a in every production stack), so it is split at every tier:
      // fork norm1-out first
      P.push(`<circle cx="${SX1}" cy="${y - 10}" r="2.5" fill="#898781"/>` +
        `<path class="wire" d="M ${SX1} ${y - 10} L ${RX} ${y - 10} L ${RX} ${y}" marker-end="url(#arr)"/>`);
      const qFrac = DSV3.qRank / (DSV3.qRank + DSV3.kvRank + DSV3.qkRope);
      const dhalf = (x, name, dims, tip, frac, withBtns, pc = '') => {
        P.push(`<g data-tip="${escAttr(tip)}">` +
          `<rect class="box" x="${x}" y="${y}" width="140" height="60" rx="4"/>` +
          `<text class="name" x="${x + 6}" y="${y + 13}">${name}</text>` +
          `<text class="dims" x="${x + 6}" y="${y + 25}">${flatten(dims)}${pc}</text></g>` +
          (withBtns ? modeBtn(['qkv_down'], x + 140 - 86, y + 29) + dtBtn('qkv_down', x + 140 - 58, y + 29) : ''));
        flopBlocks(x + 6, y + 52, ana.byId.qkv_down.flopsTok * frac, dt('qkv_down'));
      };
      const pQ = ` (${fmtP(DSV3.hidden * DSV3.qRank)})`, pKV = ` (${fmtP(DSV3.hidden * (DSV3.kvRank + DSV3.qkRope))})`;
      dhalf(C1, 'q down-proj', '7168 → 1536',
        '2 · 7168 · 1536 FLOP/token — wq_a; a separate GEMM from kv down-proj in production stacks', qFrac, true, pQ);
      dhalf(C1 + 150, 'kv down-proj', '7168 → 512 + 64',
        '2 · 7168 · (512 + 64) FLOP/token — wkv_a; shares q down-proj’s mark and dtype (one graph node)', 1 - qFrac, false, pKV);
      y += 60;
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
        P.push(`<text class="tensor tidle" x="${kx + 6}" y="${bypTop - 4}">· k_rope · ${DSV3.qkRope}</text>`);
        // pre-norm latent chips: real graph state (saved at no-AC — the latent
        // norms' backward input; the replay anchor under recompute presets)
        tensorChip(['qkv_down'], SX1 + 14, y + 24,
          { name: 'q latent', tdims: String(DSV3.qRank), frac: DSV3.qRank / latTot });
        tensorChip(['qkv_down'], RX + 14, y + 24,
          { name: 'kv latent', tdims: String(DSV3.kvRank), frac: DSV3.kvRank / latTot });
        wire(SX1, y, y + 48);
        P.push(`<path class="wire" d="M ${RX} ${y} L ${RX} ${y + 48}" marker-end="url(#arr)"/>`);
        y += 48;
        // MLA-internal RMSNorms (q_a_layernorm; kv_a_layernorm norms the 512 only)
        const normTip = 'input-form backward: reads its INPUT (pre-norm) + rstd — never its output. ' +
          'The pre-norm latent is not stashed; it is exactly recoverable from the post-norm stash, ' +
          '\u03b3, and rstd (x = y / (\u03b3\u00b7rstd)), which is why one latent copy suffices.';
        micro('RMSNorm', C1, y, 140, normTip, `(${fmtP(DSV3.qRank)})`);
        micro('RMSNorm', C1 + 150, y, 140, normTip, `(${fmtP(DSV3.kvRank)})`);
        y += 18;
        // their rstd: exits the bottom, elbows right (\u2191 = read by the op
        // above, the norm's own backward); a replayed norm regenerates it
        for (const [nid, bx] of [['q_norm', C1], ['kv_norm', C1 + 150]]) {
          const rep = ana.replayed.has(nid);
          P.push(`<path class="wire" d="M ${bx + 112} ${y} L ${bx + 112} ${y + 7} L ${bx + 124} ${y + 7}" marker-end="url(#arr)"/>` +
            `<text class="tensor ${rep ? 'tredo' : 'tsave'}" x="${bx + 128}" y="${y + 10}">${rep ? '\u21bb' : '\u2191'} rstd</text>`);
        }
        y += 14;
      }
      if (DET) {
        // the normed latents are their own graph nodes (q_norm / kv_norm)
        tensorChip(['q_norm'], SX1 + 14, y + 4);
        tensorChip(['kv_norm'], RX + 14, y + 4);
      } else {
        tensorChip(['qkv_down'], SX1 + 14, y + 4,
          { name: 'q latent', tdims: String(DSV3.qRank), frac: DSV3.qRank / latTot });
        tensorChip(['qkv_down'], RX + 14, y + 4,
          { name: 'kv latent', tdims: String(DSV3.kvRank), frac: DSV3.kvRank / latTot });
      }
      const latGap = Math.max(26, chipSpace(['qkv_down']) + 8);
      const wireTop = DET ? y - 14 : y;          // span the rstd band too — no spine gap
      wire(SX1, wireTop, y + latGap);
      P.push(`<path class="wire" d="M ${RX} ${wireTop} L ${RX} ${y + latGap}" marker-end="url(#arr)"/>`);
      y += latGap;
      const halfBox = (id, x) => {
        const m = MATMULS.find(mm2 => mm2.id === id);
        P.push(`<g${boxTip(id, m.dimsNote)}>` +
          `<rect class="box" x="${x}" y="${y}" width="140" height="60" rx="4"/>` +
          `<text class="name" x="${x + 6}" y="${y + 13}">${m.label}</text>` +
          `<text class="dims" x="${x + 6}" y="${y + 25}">${flatten(m.dims)}${pstr(id)}</text></g>` +
          modeBtn([id], x + 140 - 86, y + 29) + dtBtn(id, x + 140 - 58, y + 29));
        flopBlocks(x + 6, y + 52, ana.byId[id]?.flopsTok, dt(id));
      };
      halfBox('q_up', C1); halfBox('kv_up', C1 + 150); y += 60;
      if (DET) {
        // the up-proj outputs get names before RoPE; then two separate RoPE
        // kernels (Megatron: apply_mla_rope_for_q / _for_kv) — the kv one is
        // rope plus a little extra (split, broadcast, assemble K and V) — feed
        // q and k,v directly into attention. The k_rope rail lands here.
        P.push(`<text class="tensor tidle" x="${SX1 + 14}" y="${y + 12}">q_heads · ${flatten('128×192')}</text>` +
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
    y = mmBox(['attn'], C1, y);
    y = wireOut(['attn'], SX1, y);
    y = mmBox(['o_proj'], C1, y);
    grp(C1, g1, y + 5, 'MLA', DET ? bypX - C1 + 22 : undefined);
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
    P.push(`<g data-tip="residual add — no FLOPs; its output x1 is what the second RMSNorm&#39;s backward reads">` +
      `<rect class="res" x="${SX1 + 16}" y="${y - 11}" width="126" height="22" rx="4"/>` +
      `<text class="oplabel" x="${SX1 + 24}" y="${y + 4}">residual add</text></g>` +
      modeBtn(['x1'], SX1 + 16 + 126 - 30, y - 10));
    tensorChip(['x1'], SX1 + 16, y + 15);
    x1Y = y;
    col1End = y + 46;
    }
    }  // end MLA column

    const midX = (C1 + W + C2) / 2 + 40;

    // ---- column 2: the FFN half (MoE machinery, or one wide dense FFN);
    // skipped in only="mla" mode ----
    const nExp = DSV3.topk + DSV3.sharedExperts;   // grouped boxes carry topk/nExp, shared 1/nExp
    // kindtabs: dense/MoE flip tabs (with the per-block tally) above the FFN column
    const TABS = this.hasAttribute('kindtabs') && ONLY !== 'mla';
    let z = (ONLY === 'ffn' ? 36 : 16) + (TABS ? 34 : 0);
    if (ONLY !== 'mla') {
    if (TABS) {
      P.push('__ENC__');   // placeholder: the FFN-section enclosure, sized after the column is drawn
      // tab shapes: the ACTIVE tab fuses into the enclosure (its fill covers
      // the shared edge, its border stops at it); the inactive tab is a
      // detached grey flap sitting on the enclosure's top edge
      const tab = (x, w, kind, label, sub) => {
        const on = this.kind === kind, r = 6, y0 = 8, y1 = 34;
        const shape = `M ${x} ${y1} L ${x} ${y0 + r} Q ${x} ${y0} ${x + r} ${y0} ` +
          `L ${x + w - r} ${y0} Q ${x + w} ${y0} ${x + w} ${y0 + r} L ${x + w} ${y1}`;
        return `<g data-kind="${kind}" style="cursor:${on ? 'default' : 'pointer'}"` +
          ` title="flip the FFN column — the MLA half is identical in both block kinds">` +
          (on
            ? `<path d="${shape} Z" fill="#fcfcfb" stroke="none" transform="translate(0,1.6)"/>` +
              `<path d="${shape} Z" fill="#fcfcfb" stroke="none"/>` +
              `<path d="${shape}" fill="none" stroke="#c3c2b7"/>`
            : `<path d="${shape} Z" fill="#eeede7" stroke="#d8d6cb"/>`) +
          `<text x="${x + 10}" y="${y0 + 17}" style="font:600 11px system-ui" fill="${on ? '#0b0b0b' : '#898781'}">${label}` +
          `<tspan style="font:10px system-ui" fill="${on ? '#898781' : '#a8a69e'}"> ${sub}</tspan></text></g>`;
      };
      P.push(tab(C2 + 42, 148, 'dense', 'dense FFN', `×${DSV3.denseLayers ?? 3} · ${fmtP(3 * DSV3.hidden * DSV3.denseInter)}`) +
        tab(C2 + 198, 168, 'moe', 'MoE FFN', `×${DSV3.layers - (DSV3.denseLayers ?? 3)} · ${fmtP((DSV3.routedExperts + 1) * 3 * DSV3.hidden * DSV3.moeInter + DSV3.hidden * DSV3.routedExperts)}`));
    }
    const norm2Top = z;
    if (ONLY === 'ffn') {
      // component view: input arrives from the block wiring (post-attention x1);
      // the residual fork and add live there, not here
      P.push(`<text class="oplabel" x="${SX2 + 14}" y="${TABS ? 46 : 12}">x1 (7168) — from the block wiring</text>`);
      wire(SX2, TABS ? 40 : 6, z);
    }
    z = opNode('norm2', 'RMSNorm', C2, z, 'op', `(${fmtP(DSV3.hidden)})`);
    if (this.kind === 'dense') {
      // dense block: same spine, a single wide FFN — no router, no a2a, no
      // shared column. The column advances through the MoE rows' positions
      // (whitespace where the routing machinery sits, gaps measured on the
      // parallel MoE analysis) so flipping kinds keeps elements in place.
      tensorChip(['norm2'], SX2 + 14, z + 4);
      const spineFrom = z;   // one continuous spine through the whitespace below
      // norm2 gap (same formulas as the MoE branch), then whitespace where the
      // routing rows sit: router box (+ top-k micro in detail) + its chip,
      // a2a dispatch + its chip
      z += (DET ? Math.max(38, chipSpace(['norm2']) + 20) : Math.max(22, chipSpace(['norm2']) + 10))
        + 38 + (DET ? 18 : 0) + gapM(['router']) + 22 + gapM(['dispatch']);
      const gTop = z + 3; z += 21;
      wire(SX2, spineFrom, z);
      z = mmBox(['ffn_gate_up'], C2, z, ['gate_up'], 'ffn gate/up', `7168 → 2×${DSV3.denseInter}`);
      z = wireOut(['gate_up'], SX2, z);
      z = opNode('swiglu', 'SwiGLU', C2, z);
      z = wireOut(['swiglu'], SX2, z);
      z = mmBox(['ffn_down'], C2, z, undefined, 'ffn down', `${DSV3.denseInter} → 7168`);
      grp(C2, gTop, z + 5, 'dense FFN — every token');
      tensorChip(['ffn_down'], SX2 + 14, z + 9);
      const zc = z + 5;
      if (ONLY === 'ffn') {
        z = zc + Math.max(22, chipSpace(['ffn_down']) + 10);
      } else {
        // whitespace where the a2a combine sits; the add clamps to col1End,
        // the same row the MoE residual add lands on
        z = Math.max(zc + gapM(['ffn_down']) + 22 + gapM(['combine']) + 13, col1End - 4);
        wire(SX2, zc, z - 11);
        plus(SX2, z);
        P.push(`<g data-tip="residual add — x1 + the ffn output">` +
          `<rect class="res" x="${SX2 + 26}" y="${z - 11}" width="126" height="22" rx="4"/>` +
          `<text class="oplabel" x="${SX2 + 34}" y="${z + 4}">residual add</text></g>`);
      }
    } else {
    let shBot = 0, shTop = 0;
    const SHX = C2 + 320, shMid = SHX + 22;        // shared-expert mini column; spine down its LEFT, like every column
    const shBox = (name, dims, tip, yy, pc = '') => P.push(`<g data-tip="${escAttr(tip)}">` +
      `<rect class="box" x="${SHX}" y="${yy}" width="140" height="34" rx="4"/>` +
      `<text class="name" x="${SHX + 6}" y="${yy + 14}">${name}</text>` +
      `<text class="dims" x="${SHX + 6}" y="${yy + 27}">${flatten(dims)}${pc}</text></g>`);
    if (!DET) {
      z = wireOut(['norm2'], SX2, z);
    } else {
      // the shared expert runs on EVERY token as its own plain GEMMs — fork
      // norm2-out here; its boxes are drawn row-aligned with the routed ones
      tensorChip(['norm2'], SX2 + 14, z + 4);
      const nGap = Math.max(38, chipSpace(['norm2']) + 20);
      wire(SX2, z, z + nGap);
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
      z = micro('sigmoid · group-limited top-k · scale', C2, z);
      P.push(`<path class="wire" d="M ${C2 + W} ${gateTop} L ${gateX} ${gateTop}"/>` +
        `<text class="tensor tidle" x="${C2 + 198}" y="${z + 11}">top-k weights · 8</text>`);
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
        ` (${fmtP(DSV3.hidden * 2 * DSV3.moeInter)})`);
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
        ` (${fmtP(DSV3.moeInter * DSV3.hidden)})`);
      tensorChip(['ffn_down'], shMid + 14, z + 4, { name: 'shared out', tdims: '7168', frac: 1 / nExp });
      shBot = rowD + 34;
    }
    grp(C2, g2, z + 5, DET ? 'routed experts: top-8 of 256 — grouped GEMMs' : 'experts: top-8 of 256 routed + 1 shared');
    z = wireOut(['ffn_down'], SX2, z + 5);
    z = opNode('combine', DET ? 'a2a combine (comm + unpermute · sum)' : 'a2a combine (weighted by router)', C2, z, 'comm');
    // combine's output wire runs all the way into the x2 add; the add itself
    // is kept below column 1's residual box + x1 chip, so the x1 → x2 rail
    // turns right in clear space instead of crossing them
    tensorChip(['combine'], SX2 + 14, z + 4);
    const zc = z;                                  // combine box bottom
    if (!DET) {
      if (ONLY === 'ffn') {
        z = zc + Math.max(22, chipSpace(['combine']) + 10);
      } else {
      z = Math.max(z + Math.max(22, chipSpace(['combine']) + 10) + 13, col1End - 4);
      plus(SX2, z);
      wire(SX2, zc, z - 11);
      P.push(`<g data-tip="one fused add kernel (Megatron: add_shared_and_residual) — routed output + shared output + residual x1">` +
        `<rect class="res" x="${SX2 + 26}" y="${z - 11}" width="126" height="22" rx="4"/>` +
        `<text class="oplabel" x="${SX2 + 34}" y="${z + 4}">residual add</text></g>`);
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
      P.push(`<g data-tip="routed + shared expert outputs — Megatron fuses this with the residual add (add_shared_and_residual); split here for clarity">` +
        `<rect class="res" x="${SX2 + 26}" y="${zA - 35}" width="178" height="22" rx="4"/>` +
        `<text class="oplabel" x="${SX2 + 34}" y="${zA - 20}">add — routed + shared</text></g>`);
      if (ONLY === 'ffn') {
        z = zA;
      } else {
      const zB = Math.max(zA + 34, col1End - 4);
      wire(SX2, zA + 9, zB - 11);
      plus(SX2, zB);
      P.push(`<g data-tip="residual add — x1 + the ffn output">` +
        `<rect class="res" x="${SX2 + 26}" y="${zB - 11}" width="126" height="22" rx="4"/>` +
        `<text class="oplabel" x="${SX2 + 34}" y="${zB + 4}">residual add</text></g>`);
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
      // the enclosure the active tab fuses into — fixed extent regardless of
      // kind (the MoE-detail footprint), so it doesn't move across flips
      P[P.indexOf('__ENC__')] =
        `<rect x="${C2 - 14}" y="34" width="${(DET ? 470 : 385) + 14}" height="${z}" rx="8" fill="#fcfcfb" stroke="#c3c2b7"/>`;
    }
    }  // end FFN column (skipped in only="mla" mode)
    const col2End = ONLY === 'mla' ? 0 : z + 42;   // room for the add label under the plus

    // ---- head row (unless block-only: show the transformer block alone,
    // making no claims about the surrounding stack) ----
    let h = Math.max(col1End, col2End) + 10;
    let lmH = -20;
    if (!this.hasAttribute('block-only')) {
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
      lmH = 38 + lmRows * 6;
      P.push(`<g data-tip="${escAttr(`${fmtNum(lmFlops)} FLOP/token = ${FLOP_EXPR.lm_head}\n${lm.dimsNote}`)}">` +
        `<rect class="box" x="${C1 + 184}" y="${h}" width="240" height="${lmH}" rx="4"/>` +
        `<text class="name" x="${C1 + 192}" y="${h + 14}">${lm.label}</text>` +
        `<text class="dims" x="${C1 + 192}" y="${h + 28}">${flatten(lm.dims)}${pstr('lm_head')}</text></g>` + dtBtn('lm_head', C1 + 184 + 240 - 58, h + 7));
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

    const svgEl = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svgEl.setAttribute('width', WIDTH); svgEl.setAttribute('height', H);
    svgEl.setAttribute('viewBox', `0 0 ${WIDTH} ${H}`);
    svgEl.style.maxWidth = '100%'; svgEl.style.height = 'auto';
    svgEl.innerHTML = `<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">` +
      `<path d="M 0 0 L 8 4 L 0 8 z" fill="#898781"/></marker></defs>` + P.join('');
    for (const b of svgEl.querySelectorAll('button[data-dt]')) {
      b.onclick = () => {
        const cycle = { bf16: 'mxfp8', mxfp8: 'fp32', fp32: 'bf16' };
        this.matmuls[b.dataset.dt] = cycle[this.matmuls[b.dataset.dt]] ?? 'bf16';
        this.render(); this.changed();
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

// ---- <dsv3-stack> custom element --------------------------------------------------
// The model's layer STRUCTURE: which blocks repeat how many times —
// `embedding → dense block ×D → MoE block ×(L−D) → final norm → lm head`.
// The disclosure widget for "all layers are sparse" simplifications: `choices`
// offers dense-layer counts to toggle between, patching linked widgets' arch.
// (Future home for global/local attention patterns too.)
const STACK_CSS = `
dsv3-stack { display: block; margin: 12px 0; }
.sk { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: #0b0b0b;
  border: 1px solid #e1e0d9; border-radius: 6px; background: #fcfcfb; padding: 10px 12px;
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.sk .blk { border: 1px solid #c3c2b7; background: #fff; border-radius: 6px; padding: 4px 10px; text-align: center; }
.sk .blk small { display: block; color: #898781; font-size: 10px; }
.sk .mut { border-style: dashed; color: #a8a69e; }
.sk .mut small { color: #c3c2b7; }
.sk .pill { background: #f0efe9; border-radius: 12px; padding: 5px 12px; color: #52514e; }
.sk .arr { color: #898781; }
.sk .tg { margin-left: auto; display: flex; gap: 6px; align-items: center; color: #52514e; }
.sk .tg button { font: 11px system-ui; border: 1px solid #c3c2b7; border-radius: 4px; background: #fff;
  padding: 3px 8px; cursor: pointer; }
.sk .tg button.on { border-color: #eda100; background: #fff7e6; font-weight: 600; }
`;
class Dsv3Stack extends HTMLElement {
  static get observedAttributes() { return ['arch', 'dense', 'for', 'choices']; }
  connectedCallback() { this.init(); }
  attributeChangedCallback() { if (this.isConnected) this.init(); }
  init() {
    this.arch = JSON.parse(this.getAttribute('arch') ?? 'null');
    this.choices = (this.getAttribute('choices') ?? '').split(',').filter(s => s !== '').map(Number);
    this.dense = readUrlState('s:' + this.id)?.dense
      ?? +(this.getAttribute('dense') ?? this.arch?.denseLayers ?? 0);
    this.render();
    this.push(false);
  }
  push(persist = true) {
    if (persist && this.id) writeUrlState('s:' + this.id, { dense: this.dense });
    if (this.arch && this.getAttribute('for'))
      patchTargets(this.getAttribute('for'), { arch: { ...this.arch, denseLayers: this.dense } });
  }
  render() {
    const a = this.arch ?? DSV3;
    const L = a.layers, d = this.dense;
    this.innerHTML = '';
    const style = document.createElement('style'); style.textContent = STACK_CSS;
    const root = el('div', 'sk');
    const box = (cls, html) => { const b = el('div', cls); b.innerHTML = html; return b; };
    root.append(box('pill', 'embedding'), box('arr', '→'));
    root.append(box('blk' + (d === 0 ? ' mut' : ''),
      `<b>dense block ×${d}</b><small>MLA + FFN 7168→${a.denseInter ?? '?'}</small>`), box('arr', '→'));
    root.append(box('blk',
      `<b>MoE block ×${L - d}</b><small>MLA + top-${a.topk} of ${a.routedExperts} + ${a.sharedExperts} shared</small>`),
      box('arr', '→'));
    root.append(box('pill', 'final norm → lm head'));
    if (this.choices.length > 1) {
      const tg = el('div', 'tg');
      tg.append(document.createTextNode('first layers:'));
      for (const c of this.choices) {
        const b = document.createElement('button');
        b.textContent = c === 0 ? `all MoE ×${L} (simplification)` : `${c} dense + ${L - c} MoE`;
        if (c === d) b.classList.add('on');
        b.onclick = () => { this.dense = c; this.render(); this.push(); };
        tg.append(b);
      }
      root.append(tg);
    }
    this.append(style, root);
  }
}
if (typeof customElements !== 'undefined' && !customElements.get('dsv3-stack')) {
  customElements.define('dsv3-stack', Dsv3Stack);
}

// ---- <dsv3-controls> custom element ----------------------------------------------
// A control strip that patches linked widgets (memory + trace) so one set of
// choices drives both. fields="hardware,gpus,pp,..." picks the knobs;
// toggles="true" adds the per-refinement checkboxes (for trace targets).
// Named study/config bundles: hardware, parallelism, precision recipe and
// recompute policy switch together (the two reference configurations).
export const CONFIG_BUNDLES = {
  'nvidia-mlperf-gb300': {
    label: 'NVIDIA MLPerf 6.0 — 256×GB300 · PP2·EP32 · MXFP8 · no AC',
    values: { hardware: 'gb300', pp: 2, ep: 32, dp: 128, globalBatch: 4096, zero: 1, seqLen: 4096 },
    recipe: 'nv-mxfp8', recompute: 'none',
  },
  'dsv3-h100': {
    label: 'DSv3-style — 2048×H100 · PP8·EP64 · FP8 tile · attn-replay AC',
    values: { hardware: 'h100', pp: 8, ep: 64, dp: 256, globalBatch: 16384, zero: 1, seqLen: 4096 },
    recipe: 'dsv3-fp8', recompute: 'attn-replay',
  },
};

const FIELD_DEFS = {
  hardware: ['select', [['gb300', 'GB300 NVL72'], ['gb200', 'GB200 NVL72'], ['h800', 'H800'], ['h100', 'H100']], 'gb300'],
  dtype: ['select', [['bf16', 'BF16'], ['mxfp8', 'MXFP8']], 'bf16'],
  zero: ['select', [['3', 'FSDP (ZeRO-3)'], ['1', 'ZeRO-1'], ['0', 'ZeRO-0']], '3', true],
  recompute: ['select', [['selective', 'selective'], ['dsv3', 'dsv3 (keep gate/up)'], ['none', 'none'], ['full', 'full']], 'selective'],
  recipe: ['select', [['nv-mxfp8', 'NV MXFP8'], ['dsv3-fp8', 'DSv3 FP8'], ['bf16', 'BF16']], 'nv-mxfp8'],
  granularity: ['select', [['phase', 'microbatch'], ['layer', 'layer'], ['op', 'op']], 'layer'],
  gpus: ['number', null, 256, true], pp: ['number', null, 16, true], ep: ['number', null, 16, true],
  dp: ['number', null, 16, true], globalBatch: ['number', null, 384, true],
  microbatches: ['number', null, 24, true], mbs: ['number', null, 1, true], seqLen: ['number', null, 4096, true],
  dpRanksToSim: ['number', null, 1, true], seed: ['number', null, 42, true],
};
const TOGGLE_DEFS = [
  ['pipeline', '1F1B pipeline', true], ['epComm', 'expert a2a', true], ['dpComm', 'DP/PP comm', true],
  ['overhead', 'launch overhead', false], ['jitter', 'jitter & GC', false],
];
const CONTROLS_CSS = `
.cw { display: flex; flex-wrap: wrap; gap: 10px 16px; padding: 10px 12px; background: #fcfcfb;
  border: 1px solid #e1e0d9; border-radius: 6px; font: 13px system-ui, sans-serif; color: #0b0b0b; }
.cw label { display: flex; flex-direction: column; gap: 2px; color: #52514e; font-size: 12px; }
.cw input, .cw select { font: 13px system-ui; padding: 3px 6px; border: 1px solid #c3c2b7;
  border-radius: 4px; background: #fff; color: #0b0b0b; width: 92px; }
.cw .chk { flex-direction: row; align-items: center; gap: 5px; padding-top: 14px; }
.cw .chk input { width: auto; }
.cw .brk { flex-basis: 100%; height: 0; }
`;
export class Dsv3Controls extends HTMLElement {
  connectedCallback() {
    const fieldNames = (this.getAttribute('fields') ?? 'hardware,gpus,pp,ep,zero,recompute,mbs,seqLen,microbatches')
      .split(',').map(s => s.trim()).filter(f => FIELD_DEFS[f]);
    this.urlKey = 'c:' + (this.id || 'controls');
    this._authoredValues = JSON.parse(this.getAttribute('values') ?? '{}');
    const values = { ...this._authoredValues, ...(readUrlState(this.urlKey) ?? {}) };
    const bundleNames = (this.getAttribute('bundles') ?? '').split(',').map(s => s.trim()).filter(b => CONFIG_BUNDLES[b]);
    const style = document.createElement('style'); style.textContent = CONTROLS_CSS;
    const root = el('div', 'cw');
    this.inputs = {};
    if (bundleNames.length) {
      const label = el('label'); label.append('preset');
      const sel = document.createElement('select');
      for (const b of bundleNames) {
        const o = document.createElement('option'); o.value = b; o.textContent = CONFIG_BUNDLES[b].label; sel.append(o);
      }
      sel.style.width = 'auto';
      sel.onchange = () => this.applyBundle(sel.value);
      label.append(sel); root.append(label);
      this._bundleSel = sel;
    }
    for (const name of fieldNames) {
      const [kind, options, dflt] = FIELD_DEFS[name];
      const initial = values[name] ?? dflt;
      const label = el('label'); label.append(name);
      let input;
      if (kind === 'select') {
        input = document.createElement('select');
        for (const [v, text] of options) {
          const o = document.createElement('option');
          o.value = v; o.textContent = text; o.selected = String(v) === String(initial);
          input.append(o);
        }
      } else {
        input = document.createElement('input');
        input.type = 'number'; input.value = initial;
      }
      input.onchange = () => this.apply(true);
      this.inputs[name] = input;
      label.append(input); root.append(label);
    }
    if (this.hasAttribute('toggles')) {
      root.append(Object.assign(el('span', 'brk')));
      for (const [name, text, dflt] of TOGGLE_DEFS) {
        const label = el('label', 'chk');
        const input = document.createElement('input');
        input.type = 'checkbox'; input.checked = values[name] ?? dflt;
        input.onchange = () => this.apply(true);
        this.inputs[name] = input;
        label.append(input, text); root.append(label);
      }
    }
    this.derivedEl = document.createElement('div');
    this.derivedEl.style.cssText = 'flex-basis:100%;font:12px system-ui;color:#52514e;';
    root.append(this.derivedEl);
    const reset = document.createElement('button');
    reset.textContent = 'reset';
    reset.style.cssText = 'font:12px system-ui;align-self:end;padding:3px 10px;border:1px solid #c3c2b7;border-radius:4px;background:#fff;cursor:pointer;';
    reset.onclick = () => {
      for (const name of fieldNames) {
        const [, , dflt] = [FIELD_DEFS[name][0], FIELD_DEFS[name][1], FIELD_DEFS[name][2]];
        this.inputs[name].value = String(this._authoredValues[name] ?? FIELD_DEFS[name][2]);
      }
      if (this.hasAttribute('toggles'))
        for (const [name, , dflt] of TOGGLE_DEFS) this.inputs[name].checked = this._authoredValues[name] ?? dflt;
      clearUrlState(this.urlKey);
      this.apply(false);
    };
    root.append(reset);
    this.fieldNames = fieldNames;
    this.append(style, root);
    this.apply(false);
  }
  applyBundle(name) {
    const b = CONFIG_BUNDLES[name];
    if (!b) return;
    for (const [k, v] of Object.entries(b.values)) if (this.inputs[k]) this.inputs[k].value = String(v);
    const layer = this.getAttribute('layer') && document.getElementById(this.getAttribute('layer'));
    if (layer?.applyPreset) layer.applyPreset(b.recipe, b.recompute, b.transposed ?? false);
    this.apply(true);
  }
  apply(write = false) {
    const patch = {}, url = {};
    for (const name of this.fieldNames) {
      const v = this.inputs[name].value;
      patch[name] = FIELD_DEFS[name][3] ? +v : v;
      url[name] = patch[name];
    }
    // parallelism is the input; the cluster size is imputed and transmitted
    // (rows pinning their own pp re-derive dp = gpus/pp; m = GBS/dp per config)
    if (this.fieldNames.includes('dp') && this.fieldNames.includes('globalBatch')) {
      const gpus = patch.pp * patch.dp, m = Math.max(1, Math.round(patch.globalBatch / patch.dp));
      patch.gpus = gpus;
      delete patch.dp;
      if (this.derivedEl) this.derivedEl.textContent =
        `⇒ gpus = pp × dp = ${gpus} · mbs = 1 sequence · m = GBS ÷ dp = ${patch.globalBatch} ÷ ${gpus / patch.pp} = ${m} microbatches/step` +
        ` · 1F1B in-flight on stage s = min(pp − s, m), worst = ${Math.min(patch.pp, m)}`;
    }
    patchTargets(this.getAttribute('for'), patch);
    this.dispatchEvent(new CustomEvent('config', { detail: patch }));
    if (write) writeUrlState(this.urlKey, url);
  }
}
if (typeof customElements !== 'undefined' && !customElements.get('dsv3-controls')) {
  customElements.define('dsv3-controls', Dsv3Controls);
}
