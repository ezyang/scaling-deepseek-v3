// <dsv3-kernels pass=F|B|W mode=trace|coda>: one DeepSeek-V3 MoE layer at
// kernel-launch granularity, in the block diagram's visual language (01's
// grammar: a spine per column, GEMMs as .box, memory-bound kernels as .op
// pills, communication violet, elided glue in .micro italics). Every launch
// of the DeepSeek EP64 profile is a row — attention sub-layer on the left,
// MoE sub-layer on the right — with a two-bar meter beside it: the traced
// time (grey) over its speed of light (bronze when HBM binds, the dtype's
// precision color when the tensor cores do), on a log axis. The CODA flip
// keeps every anchor: launches that fold into a GEMM's epilogue (or an
// auxiliary reduction, FlashAttention, DeepEP, a once-per-step weight
// transpose, or a stash the forward now writes) fade to ghosts where they
// stood, the host GEMMs grow an epilogue tag, and the HBM markers move;
// the footer sums each column. Hover a row for the kernel, its operands,
// both meters and where CODA puts it. State rides the hash (`k:<id>`).
import { priced, tally, RATES } from './kerntrace.js';
import { tokensCss } from './viewer.js';
import { C } from './theme.js';
import { knobCss } from './ui.js';
import { attachTip } from './tip.js';

export const KERNELS_DEFAULTS = { pass: 'F', mode: 'trace' };
const PASSES = [['F', 'forward'], ['B', 'backward (input gradients)'], ['W', 'weight gradients']];
const MODES = [['trace', 'as traced'], ['coda', 'with CODA']];

// ---- rows: launches grouped into drawable rows, per pass ---------------------
// consecutive glue launches of one op collapse into a single `.micro` row
// (the tooltip lists them); everything else is one row per launch
export function rowsFor(pass) {
  const P = priced();
  const cols = { attn: [], mlp: [] };
  const push = (col, r) => cols[col].push(r);
  const one = (e) => ({ key: String(e.i), es: [e], lab: e.lab, kind: e.kind, op: e.op, us: e.us, raw: e.raw, b: e.b, sol: e.sol, hbm: e.hbm, flop: e.flop,
    bound: e.bound, dt: e.dt, host: e.kind === 'gemm' || e.kind === 'attn' || e.i === 175, gone: e.gone, aux: e.aux, hbmAfter: e.hbmAfter, solAfter: e.solAfter, epi: e.epi, how: e.how, comm: e.comm });
  const compute = P.filter((e) => e.pass === pass && !e.comm);
  for (const e of compute) {
    const col = e.col;
    const last = cols[col].at(-1);
    if (e.kind === 'glue' && last?.kind === 'glue' && last.op === e.op) {
      last.es.push(e); last.us += e.us; last.raw += e.raw; last.b += e.b; last.sol += e.sol; last.hbm += e.hbm; last.hbmAfter += e.hbmAfter; last.solAfter += e.solAfter;
      last.gone = last.gone && e.gone; last.aux = last.aux || e.aux; last.key += '+' + e.i; last.lab = `${e.op}: ${last.es.length} small launches`;
      continue;
    }
    push(col, one(e));
  }
  // the communication stream frames the MoE column: what DeepEP runs before and after the experts
  const comm = P.filter((e) => e.comm && (pass === 'F' ? /^(dispatch|combine)\(F\)/.test(e.ann) : pass === 'B' ? /\(B\)/.test(e.ann) : false));
  const before = comm.filter((e) => pass === 'F' ? e.ann === 'dispatch(F)' : e.ann === 'combine(B)').map(one);
  const after = comm.filter((e) => pass === 'F' ? e.ann === 'combine(F)' : e.ann === 'dispatch(B)').map(one);
  cols.mlp = [...before, ...cols.mlp, ...after];
  return cols;
}

// ---- geometry ----------------------------------------------------------------
const W = 900, COLW = 436, GUT = 64, SPX = GUT + 8, BX = SPX + 10, BW = 226, MX = BX + BW + 10, MW = 112, ROWH = 18, BH = 13, TOP = 30, FOOT = 46;
const LOG0 = 1e-6, DOUBLINGS = 12, PXD = MW / DOUBLINGS;      // 1 µs … 4,096 µs
const mx = (s) => MX + Math.max(0, Math.min(MW, Math.log2(Math.max(s, LOG0) / LOG0) * PXD));
const n = (x, d = 0) => x.toLocaleString('en-US', { maximumFractionDigits: d, minimumFractionDigits: d });
const fmtUs = (s) => s >= 1e-3 ? `${n(s * 1e3, 2)} ms` : s >= 1e-4 ? `${n(s * 1e6)} µs` : `${n(s * 1e6, s >= 1e-5 ? 1 : 2)} µs`;
const fmtB = (b) => b >= 1e9 ? `${n(b / 1e9, 2)} GB` : b >= 1e6 ? `${n(b / 1e6, 1)} MB` : b >= 1e3 ? `${n(b / 1e3, 1)} KB` : `${n(b)} B`;
const fmtF = (f) => f >= 1e12 ? `${n(f / 1e12, 2)} TFLOP` : f >= 1e9 ? `${n(f / 1e9, 1)} GFLOP` : `${n(f / 1e6)} MFLOP`;
const DT = { fp8: '#d6408b', bf16: '#52514e' };
const BRONZE = '#8c5a19';
const ell = (s, k) => s.length > k ? s.slice(0, k - 1).trimEnd() + '…' : s;
const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;');
const fitEase = (p) => 1 - (1 - p) ** 3;
const lerp = (a, b, t) => a + (b - a) * t;

const CSS = `
dsv3-kernels { display: block; margin: 14px 0 26px; }
.kn { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-0b0b0b);
  border: 1px solid var(--c-e1e0d9); border-radius: 6px; background: var(--c-fcfcfb); padding: 8px 10px; position: relative;
  width: max-content; max-width: 100%; box-sizing: border-box; }
.kn .top { display: flex; align-items: stretch; gap: 10px; padding-bottom: 8px; flex-wrap: wrap; }
${knobCss('.kn .top')}
.kn svg { display: block; }
${tokensCss('.kn')}
.kn .op { stroke-width: 1; }
.kn .lab { font: 10.5px system-ui; fill: var(--c-0b0b0b); }
.kn .box + .lab, .kn text.bold { font-weight: 600; }
.kn .dims { font: 9px system-ui; fill: var(--c-898781); }
.kn .gl { font: italic 9.5px system-ui; fill: var(--c-898781); }
.kn .microlabel { font: italic 10px system-ui; fill: var(--c-52514e); }
.kn .colhead { font: 600 11px system-ui; fill: var(--c-52514e); }
.kn .tick { stroke: var(--c-e1e0d9); stroke-width: 1; }
.kn .epi { font: 9px system-ui; fill: var(--c-7a5200); }
.kn .epibox { fill: var(--c-fff3d1); stroke: var(--c-eda100); stroke-width: 0.8; }
.kn .foot { font: 11px system-ui; fill: var(--c-52514e); }
.kn .foot b { font-weight: 600; fill: var(--c-0b0b0b); }
.kn .legend { font: 9.5px system-ui; fill: var(--c-52514e); }
.kn .ro { font-size: 11.5px; color: var(--c-52514e); margin-top: 6px; max-width: ${W}px; line-height: 1.45; min-height: 34px; }
.kn .ro b { color: var(--c-0b0b0b); font-weight: 600; }
.kn .dsv3-tip { max-width: 420px; }
.kn .dsv3-tip b { font-weight: 600; }
.kn .dsv3-tip .k { font: 10px ui-monospace, Menlo, monospace; color: var(--c-52514e); word-break: break-all; display: block; margin-top: 3px; }
`;

class Dsv3Kernels extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    const st = this.id ? readState('k:' + this.id) : null;
    this.cfg = { ...KERNELS_DEFAULTS };
    for (const k of Object.keys(KERNELS_DEFAULTS)) { const v = st?.[k] ?? this.getAttribute(k); if (v != null) this.cfg[k] = v; }
    this.fixed = this.hasAttribute('fixed');   // no knobs: a pinned figure
    const style = document.createElement('style'); style.textContent = CSS;
    this._root = el('div', 'kn');
    this._top = el('div', 'top');
    this._chart = el('div');
    this._ro = el('div', 'ro');
    this._root.append(this._top, this._chart, this._ro);
    this.append(style, this._root);
    this._ui = {};
    if (!this.fixed) this._buildKnobs(); else this._top.remove();
    attachTip(this._chart, (ev) => this._tipFor(ev.target.closest?.('g[data-row]')), { parent: this._root });
    this._L = null;
    this.render();
    addEventListener('dsv3-theme', () => this.render());
  }
  _buildKnobs() {
    const grp = (label) => { const g = el('span', 'pargrp'); const l = el('div', 'parlab'); l.textContent = label; g.append(l); return g; };
    const row = (...kids) => { const r = el('div', 'parrow'); r.append(...kids); return r; };
    const seg = (key, opts) => {
      const eg = el('span', 'stp'); eg.dataset.knob = key;
      for (const [v, lab] of opts) { const x = document.createElement('button'); x.type = 'button'; x.textContent = lab; x.dataset.v = v; x.onclick = () => this._set(key, v); eg.append(x); }
      this._ui[key] = eg; return eg;
    };
    const g1 = grp('pass'); g1.append(row(seg('pass', PASSES)));
    const g2 = grp('kernels'); g2.append(row(seg('mode', MODES)));
    this._top.append(g1, g2);
    this._syncKnobs();
  }
  _syncKnobs() { for (const [k, eg] of Object.entries(this._ui)) for (const b of eg.querySelectorAll('button')) b.classList.toggle('on', b.dataset.v === this.cfg[k]); }
  _set(k, v) {
    if (this.cfg[k] === v) return;
    this.cfg[k] = v; this._syncKnobs();
    if (this.id) writeState('k:' + this.id, this.cfg);
    this.dispatchEvent(new CustomEvent('dsv3-kernels', { detail: { ...this.cfg } }));
    if (k === 'pass') this.render(); else this._animateTo(this._layout());
  }
  // pixel-space layout: rows keyed by launch ids; the mode decides each row's presence and the HBM marker
  _layout() {
    const coda = this.cfg.mode === 'coda';
    const cols = rowsFor(this.cfg.pass);
    const H = Math.max(cols.attn.length, cols.mlp.length) * ROWH;
    const L = { coda, pass: this.cfg.pass, cols: {}, h: TOP + H + FOOT, T: tally()[this.cfg.pass] };
    for (const [c, x0] of [['attn', 0], ['mlp', COLW + 28]]) {
      L.cols[c] = cols[c].map((r, j) => {
        const y = TOP + j * ROWH;
        const ghost = coda && r.gone;
        const sol = coda && r.host ? r.solAfter : r.sol, hbm = coda ? r.hbmAfter : r.hbm;
        const aux = ghost && r.aux;   // a small kernel remains where the launch stood: ghosted, with its remaining floor
        return { ...r, x0, y, ghost, a: ghost ? (aux ? 0.6 : 0.3) : 1, tx: r.us ? mx(r.us * 1e-6) : MX, sx: sol > 0 && (!ghost || aux) ? mx(sol) : MX, hx: hbm > 0 && !ghost ? mx(hbm) : MX,
          showEpi: coda && r.host && r.epi?.length };
      });
    }
    return L;
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
    // legend + axis heads
    B.push(`<text class="legend" x="${BX}" y="12">one row per launch: <tspan font-weight="600">GEMM</tspan> boxes · memory-bound kernel pills · <tspan font-style="italic">collapsed glue</tspan> · DeepEP violet</text>`);
    const lx = COLW + 28 + BX + 60;
    B.push(`<rect x="${lx}" y="6" width="12" height="4" fill="${C('#898781')}"/><text class="legend" x="${lx + 15}" y="11">traced</text>`);
    B.push(`<rect x="${lx + 54}" y="6" width="12" height="4" fill="${C(BRONZE)}"/><text class="legend" x="${lx + 69}" y="11">HBM floor</text>`);
    B.push(`<rect x="${lx + 124}" y="6" width="12" height="4" fill="${C(DT.fp8)}"/><text class="legend" x="${lx + 139}" y="11">tensor-core floor (fp8 · <tspan fill="${C(DT.bf16)}">bf16</tspan>)</text>`);
    for (const [c, head] of [['attn', 'attention sub-layer'], ['mlp', 'MoE sub-layer']]) {
      const rows = L.cols[c], x0 = rows[0]?.x0 ?? (c === 'attn' ? 0 : COLW + 28);
      B.push(`<text class="colhead" x="${x0 + BX}" y="${TOP - 8}">${head}</text>`);
      // the meter's log axis: ticks at 1, 10, 100, 1000 µs
      for (const us of [1, 10, 100, 1000]) {
        const x = x0 + mx(us * 1e-6);
        B.push(`<line class="tick" x1="${f1(x)}" y1="${TOP - 4}" x2="${f1(x)}" y2="${TOP + rows.length * ROWH}"/>`);
        B.push(`<text class="dims" x="${f1(x)}" y="${TOP - 7}" text-anchor="middle">${us >= 1000 ? '1 ms' : us + ' µs'}</text>`);
      }
      // the spine
      if (rows.length) B.push(`<line class="wire" x1="${x0 + SPX}" y1="${TOP - 2}" x2="${x0 + SPX}" y2="${TOP + rows.length * ROWH - 4}"/>`);
      let lastOp = null;
      for (const r of rows) {
        const g = [];
        const y = r.y, xb = r.x0 + BX;
        if (r.op !== lastOp) { g.push(`<text class="gl" x="${r.x0 + GUT}" y="${y + BH - 3}" text-anchor="end">${esc(ell(r.op, 13))}</text>`); lastOp = r.op; }
        g.push(`<line class="wire" x1="${r.x0 + SPX}" y1="${y + BH / 2}" x2="${xb}" y2="${y + BH / 2}"/>`);
        const cls = r.comm ? 'comm' : r.kind === 'gemm' || r.kind === 'attn' ? 'box' : r.kind === 'glue' ? 'micro' : 'op';
        const rx = cls === 'op' || cls === 'micro' ? 6 : 2;
        g.push(`<rect class="${cls}" x="${xb}" y="${y}" width="${BW}" height="${BH}" rx="${rx}"${r.ghost ? ' stroke-dasharray="3 2"' : ''}${cls === 'micro' ? ` fill="${C('#f7f6f1')}" stroke="${C('#d8d6cb')}"` : ''}/>`);
        const e0 = r.es[0];
        const dims = r.kind === 'gemm' ? ` <tspan class="dims">${n(e0.M)}×${n(e0.Kd)}→${n(e0.N)}${e0.groups > 1 ? ` ×${e0.groups}` : ''}</tspan>` : '';
        const tcls = cls === 'micro' ? 'microlabel' : cls === 'box' ? 'lab bold' : 'lab';
        // under CODA a host's label yields room to its epilogue tag (what the GEMM absorbed)
        const labN = r.showEpi ? 14 : r.kind === 'gemm' ? 22 : 42;
        g.push(`<text class="${tcls}" x="${xb + 5}" y="${y + BH - 3}">${esc(ell(r.lab, labN))}${r.showEpi ? '' : dims}</text>`);
        if (r.showEpi) {
          const x1 = xb + 5 + Math.min(r.lab.length, labN) * 6.3 + 6, w = xb + BW - 4 - x1;
          const txt = ell(r.epi.join(' · '), Math.floor((w - 6) / 4.7));
          g.push(`<rect class="epibox" x="${f1(x1)}" y="${y + 1.5}" width="${f1(w)}" height="${BH - 3}" rx="3"/>`);
          g.push(`<text class="epi" x="${f1(x1 + 3)}" y="${y + BH - 4}">${esc(txt)}</text>`);
        }
        // the meter: traced over its floor
        const mx0 = r.x0 + MX;
        if (!r.ghost) g.push(`<rect data-bar="trace" x="${mx0}" y="${y + 1}" width="${f1(Math.max(0.6, r.x0 + r.tx - mx0))}" height="${(BH - 3) / 2}" fill="${C('#898781')}"/>`);
        if (!r.ghost || (r.aux && r.sx > MX + 0.5)) {
          if (!r.comm) {
            const fill = r.host ? (r.sol === r.flop ? C(DT[r.dt] ?? '#52514e') : C(BRONZE)) : C(BRONZE);
            g.push(`<rect data-bar="sol" x="${mx0}" y="${y + 1 + (BH - 3) / 2 + 1}" width="${f1(Math.max(0.6, r.x0 + r.sx - mx0))}" height="${(BH - 3) / 2}" fill="${fill}"/>`);
            if (r.host && r.flop) g.push(`<rect data-bar="hbm" x="${f1(r.x0 + r.hx - 1)}" y="${y + 1 + (BH - 3) / 2}" width="2" height="${(BH - 3) / 2 + 2}" fill="${C(BRONZE)}"/>`);
          }
        }
        g.push(`<rect data-hit x="${r.x0}" y="${y - 2}" width="${COLW}" height="${ROWH}" fill="transparent"/>`);
        B.push(`<g data-row="${r.key}" data-kind="${r.kind}" data-gone="${r.ghost ? 1 : 0}" opacity="${r.a.toFixed(3)}">${g.join('')}</g>`);
      }
    }
    // the footer: each column's sums, then the layer's
    const T = L.T, fy = L.h - FOOT + 16;
    const nonGemm = (k) => k === 'trace' ? T.other.trace : k === 'hbm' ? T.other.hbm * 1e6 : T.other.hbmAfter * 1e6;
    const b = (t) => `<tspan font-weight="600" fill="${C('#0b0b0b')}">${t}</tspan>`;
    B.push(`<text class="foot" x="${BX}" y="${fy}" data-foot="trace">traced, this pass: ${b(n(T.gemm.trace / 1e3, 2) + ' ms')} in GEMMs and attention, ${b(n(T.other.trace / 1e3, 2) + ' ms')} in the other ${T.other.kept + T.other.gone + T.other.aux} launches</text>`);
    B.push(`<text class="foot" x="${BX}" y="${fy + 15}" data-foot="floor">HBM floor of those ${T.other.kept + T.other.gone + T.other.aux}: ${b(n(nonGemm('hbm') / 1e3, 2) + ' ms')}${L.coda ? ` → with CODA ${b(n(nonGemm('after') / 1e3, 2) + ' ms')} in ${T.other.kept + T.other.aux} launches; hosts: ${n(T.gemm.hbmAfter * 1e3, 1)} ms of HBM under ${n(T.gemm.flop * 1e3, 1)} ms on the tensor cores${T.gemm.over.length ? `, ${T.gemm.over.length} tips HBM-bound (+${n(T.gemm.exposed * 1e6)} µs)` : ''}` : ` — traced ×${n(T.other.eff, 1)} above it`}</text>`);
    this._chart.innerHTML = `<svg width="${W}" height="${L.h}" viewBox="0 0 ${W} ${L.h}">${B.join('')}</svg>`;
    this._readout(L);
  }
  _readout(L) {
    const T = L.T, p = PASSES.find(([k]) => k === L.pass)[1];
    const g = T.gemm, o = T.other;
    this._ro.innerHTML = `<b>${p}</b>, one MoE layer, one 4,096-token microbatch, one GPU of 64: ${T.launches} kernel launches. ` +
      `GEMMs and attention run ×${n(g.trace * 1e-6 / g.sol, 2)} their speed of light (${fmtUs(g.sol)} at 03's measured rates); the other ${o.kept + o.gone + o.aux} launches take ${fmtUs(o.trace * 1e-6)} against an HBM floor of ${fmtUs(o.hbm)}, ×${n(o.eff, 1)}. ` +
      (L.coda ? `Under CODA ${o.gone} of them fold away, ${o.aux} shrink to reductions over tile partials, and the floor of what remains is ${fmtUs(o.hbmAfter)}; at today's ×${n(o.eff, 1)} that is ${fmtUs(o.projected)} of kernel time instead of ${fmtUs(o.trace * 1e-6)}.` : 'Flip to CODA to see which fold into GEMM epilogues.');
  }
  _tipFor(g) {
    if (!g) return null;
    const L = this._L, key = g.dataset.row;
    const r = [...L.cols.attn, ...L.cols.mlp].find((q) => q.key === key);
    if (!r) return null;
    const d = document.createElement('div');
    const es = r.es;
    const lines = [];
    lines.push(`<b>${esc(r.lab)}</b>${es.length === 1 && es[0].inf ? ' <i>(operand inferred from the launch shape)</i>' : ''}`);
    if (es.length === 1) {
      const e = es[0];
      lines.push(`${e.ann} · stream ${e.s} · grid ${e.grid.join('×')} · block ${e.block.join('×')}`);
      lines.push(`traced ${fmtUs(e.raw * 1e-6)}${e.rows ? ` (normalized to 32,768 routed rows: ${fmtUs(e.us * 1e-6)})` : ''}, mean of 4 cycles`);
      if (e.kind === 'gemm') lines.push(`${n(e.M)} × ${n(e.Kd)} → ${n(e.N)}${e.groups > 1 ? ` (${e.groups} experts)` : ''}: ${fmtF(e.f)} ÷ ${n(RATES[e.dt] / 1e12)} TFLOP/s = ${fmtUs(e.flop)}; ${fmtB(e.b)} ÷ 3.35 TB/s = ${fmtUs(e.hbm)} → <b>${e.bound}-bound</b>, traced ×${n(e.x, 2)}`);
      else if (e.kind === 'attn') lines.push(`${fmtF(e.f)} ÷ ${n(RATES[e.dt] / 1e12)} TFLOP/s = ${fmtUs(e.flop)}; ${fmtB(e.b)} ÷ 3.35 TB/s = ${fmtUs(e.hbm)} → traced ×${n(e.x, 2)}`);
      else if (!e.comm) lines.push(`${fmtB(e.b)} through HBM ÷ 3.35 TB/s = <b>${fmtUs(e.hbm)}</b>${e.hbm > 0 ? ` → traced ×${n(e.x, 1)}` : ''}`);
      else lines.push('communication (priced on the network in 03, not on HBM)');
      if (e.kind === 'gemm' || e.kind === 'attn' || e.i === 175) { if (e.epi?.length) lines.push(`CODA epilogue: ${e.epi.join(' · ')}${e.add ? ` (+${fmtB(e.add)} of epilogue traffic)` : ''}${e.out ? ` (output ${fmtB(-e.out)} smaller)` : ''} → HBM ${fmtUs(e.hbmAfter)}, still under ${fmtUs(e.flop)}`); }
      else if (!e.comm) lines.push(`CODA: ${HOW[e.how]}${typeof e.host === 'number' ? ` of <i>${esc(priced()[e.host].lab)}</i>` : e.host ? ` (${e.host})` : ''}${e.after > 0 ? ` — ${fmtB(e.after)} remains` : ''}`);
      if (e.note) lines.push(`<i>${esc(e.note)}</i>`);
      lines.push(`<span class="k">${esc(e.name.length > 200 ? e.name.slice(0, 200) + '…' : e.name)}</span>`);
    } else {
      lines.push(`${es.length} launches, ${fmtUs(r.raw * 1e-6)} traced in all; HBM floor ${fmtUs(r.hbm)} — latency-bound, not bandwidth-bound`);
      for (const e of es) lines.push(`· ${esc(e.lab.replace(/^[^:]*: /, ''))} — ${fmtUs(e.raw * 1e-6)}, grid ${e.grid[0]}`);
      lines.push(`CODA: ${es.every((e) => e.gone) ? 'fold away' : es.some((e) => e.gone) ? 'some fold into the router GEMM\'s epilogue; the rest stay' : 'stay (not GEMM-adjacent)'}`);
    }
    d.innerHTML = lines.join('<br>');
    return d;
  }
}
const HOW = { epilogue: 'into the epilogue', aux: 'an auxiliary reduction over tile partials', fa: 'into FlashAttention\'s own kernels', a2a: 'into DeepEP\'s kernels', cache: 'a weight transpose: once per step, not per microbatch', stash: 'a stash the forward epilogue writes instead', keep: 'stays' };

function blend(A, B, t) {
  if (!A || t >= 1 || A.pass !== B.pass) return B;
  const cols = {};
  for (const c of ['attn', 'mlp']) cols[c] = B.cols[c].map((b, i) => { const a = A.cols[c][i]; return a ? { ...b, a: lerp(a.a, b.a, t), sx: lerp(a.sx, b.sx, t), hx: lerp(a.hx, b.hx, t), ghost: b.ghost && t > 0.5 ? true : a.ghost && t <= 0.5, showEpi: t > 0.5 ? b.showEpi : a.showEpi } : b; });
  return { ...B, cols, coda: t > 0.5 ? B.coda : A.coda };
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
if (typeof customElements !== 'undefined' && !customElements.get('dsv3-kernels')) customElements.define('dsv3-kernels', Dsv3Kernels);
