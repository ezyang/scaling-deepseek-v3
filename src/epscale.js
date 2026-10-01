// <dsv3-epscale>: EP's speed-of-light time ÷ compute's (6ND at BF16 peak) per
// step on H800s, across the scaling ladder (src/ladder.js). The EP group stays
// DSv3's (EP64 = 8 nodes, a token deduped onto 4 of them: 3.5 remote), so per
// token and layer EP moves R·h copies (FP8 + its 1×128 scales, then BF16
// back) while compute grows as h²: the ratio falls as 1/s ∝ C^(−1/6), except
// at the small end where the fixed-vocabulary lm head pads 6N. At DSv3 this is
// exactly 03's T+EP ÷ Tc. The element's text content becomes the caption.
import { ladderPoint, LOG_C_DSV3 } from './ladder.js';
import { HARDWARE } from './model.js';
import { C } from './theme.js';

const HW = HARDWARE.h800, FB = HW.flops.bf16 / HW.nic;   // BF16 FLOPs per IB byte
const R = 4 * (1 - 1 / 8);                               // remote nodes per token
// T_EP ÷ T for one rung: IB bytes/token (fwd + bwd, dispatch + combine) at F/B vs 6N
export function epRatio(p) {
  const h = p.a.hidden, Lm = p.a.layers - p.a.denseLayers;
  return 2 * R * ((h + 4 * h / 128) + 2 * h) * Lm * FB / p.sixN;
}

const LO = 19, HI = 27, K = 161;                   // x domain (log10 FLOP), as <dsv3-ladder>'s
const W = 738, X0 = 60, PW = 570, Y0 = 22, PH = 150, AX = Y0 + PH, H = AX + 32;
const YLO = Math.log10(0.2), YHI = Math.log10(5);
const px = (lc) => X0 + (lc - LO) / (HI - LO) * PW;
const py = (u) => Y0 + (1 - (Math.log10(u) - YLO) / (YHI - YLO)) * PH;
const f1 = (v) => v.toFixed(1);
const sup = (n) => String(n).replace(/./g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹⁻'['0123456789-'.indexOf(d)]);
const fmtC = (lc) => { const e = Math.floor(lc + 1e-9); return `${(10 ** (lc - e)).toFixed(1)}×10${sup(e)}`; };
const fmtN = (n) => n >= 1e12 ? +(n / 1e12).toPrecision(3) + 'T' : n >= 1e9 ? +(n / 1e9).toPrecision(3) + 'B' : +(n / 1e6).toPrecision(3) + 'M';

const CSS = `
dsv3-epscale { display: block; margin: 14px 0 22px; }
.eps { display: block; max-width: 100%; height: auto; font: 11px system-ui, -apple-system, "Segoe UI", sans-serif; }
.eps text { fill: var(--c-52514e); }
.eps .dims { font-size: 9px; fill: var(--c-898781); }
.epscap { width: ${W}px; max-width: 100%; margin: 6px 0 0; text-align: center; font: 11px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-52514e); }
`;

// the curve, DSv3's point, and the break-even crossing on the falling side
function layout() {
  const pts = Array.from({ length: K }, (_, i) => { const lc = LO + (HI - LO) * i / (K - 1); return [lc, epRatio(ladderPoint(lc, 4096))]; });
  let lo = LOG_C_DSV3 - 2, hi = LOG_C_DSV3;
  for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (epRatio(ladderPoint(m, 4096)) > 1) lo = m; else hi = m; }
  const Ns = []; for (let e = LO; e <= HI; e++) Ns.push(ladderPoint(e, 4096).N);
  return { pts, dsv3: epRatio(ladderPoint(LOG_C_DSV3, 4096)), cross: (lo + hi) / 2, Ns };
}

class Dsv3Epscale extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    const cap = this.textContent.trim();
    const style = document.createElement('style'); style.textContent = CSS;
    this._fig = document.createElement('div');
    this.replaceChildren(style, this._fig);
    if (cap) { const c = document.createElement('div'); c.className = 'epscap'; c.textContent = cap; this.append(c); }
    this._L = layout();
    this.render();
    addEventListener('dsv3-theme', () => this.render());
  }
  render() {
    const L = this._L, B = [], dims = 'class="dims"';
    for (const u of [0.2, 0.5, 1, 2, 5]) {
      B.push(`<line x1="${X0}" y1="${f1(py(u))}" x2="${X0 + PW}" y2="${f1(py(u))}" stroke="${C(u === 1 ? '#898781' : '#e1e0d9')}"${u === 1 ? ' stroke-dasharray="4 3"' : ''}/>`);
      B.push(`<text ${dims} x="${X0 - 5}" y="${f1(py(u) + 3)}" text-anchor="end">${u}×</text>`);
    }
    B.push(`<text ${dims} x="${X0}" y="${Y0 - 8}">EP time ÷ compute time per step, log scale</text>`);
    B.push(`<text ${dims} x="${X0 + PW - 4}" y="${f1(py(1) - 5)}" text-anchor="end">EP-bound ↑</text>`);
    B.push(`<text ${dims} x="${X0 + PW - 4}" y="${f1(py(1) + 12)}" text-anchor="end">compute-bound ↓</text>`);
    for (let e = LO; e <= HI; e++) {
      const x = f1(px(e));
      B.push(`<line x1="${x}" y1="${AX}" x2="${x}" y2="${AX + 3}" stroke="${C('#898781')}"/>`);
      B.push(`<text ${dims} x="${x}" y="${AX + 13}" text-anchor="middle">10${sup(e)}</text>`);
      B.push(`<text ${dims} x="${x}" y="${AX + 25}" text-anchor="middle">${fmtN(L.Ns[e - LO])}</text>`);
    }
    B.push(`<text ${dims} x="${X0 + PW + 24}" y="${AX + 13}">C, FLOP</text><text ${dims} x="${X0 + PW + 24}" y="${AX + 25}">N, params</text>`);
    B.push(`<polyline data-curve points="${L.pts.map(([lc, u]) => `${f1(px(lc))},${f1(py(u))}`).join(' ')}" fill="none" stroke="${C('#0b0b0b')}" stroke-width="2"/>`);
    const xs = px(HI - 0.9), us = epRatio(ladderPoint(HI - 0.9, 4096));
    B.push(`<text x="${f1(xs)}" y="${f1(py(us) - 9)}" text-anchor="middle"><tspan font-size="13">∝</tspan> C${sup('-1')}ᐟ⁶</text>`);
    // DSv3: dashed guide as in <dsv3-ladder>, its point and value
    const dx = px(LOG_C_DSV3), dy = py(L.dsv3);
    B.push(`<line x1="${f1(dx)}" y1="${Y0 - 4}" x2="${f1(dx)}" y2="${AX}" stroke="${C('#0b0b0b')}" stroke-width="0.75" stroke-dasharray="2 2"/>`);
    B.push(`<text ${dims} x="${f1(dx)}" y="${Y0 - 8}" text-anchor="middle">DeepSeek-V3</text>`);
    B.push(`<circle data-dsv3 data-true="${L.dsv3}" cx="${f1(dx)}" cy="${f1(dy)}" r="3.5" fill="${C('#0b0b0b')}"/>`);
    B.push(`<text x="${f1(dx - 7)}" y="${f1(dy + 15)}" text-anchor="end">${L.dsv3.toFixed(2)}×</text>`);
    // break-even: where the falling side crosses 1
    const cx = px(L.cross), cy = py(1);
    B.push(`<circle data-cross data-true="${L.cross}" cx="${f1(cx)}" cy="${f1(cy)}" r="3.5" fill="${C('#fcfcfb')}" stroke="${C('#0b0b0b')}" stroke-width="1.5"/>`);
    B.push(`<text x="${f1(cx - 8)}" y="${f1(cy + 14)}" text-anchor="end">break-even at C ≈ ${fmtC(L.cross)}</text>`);
    const aria = `Across the scaling ladder on H800s, EP communication time divided by compute time per step, on a log scale. It peaks near 2.5 at 10^20 FLOP, falls through break-even at about ${fmtC(L.cross)} FLOP, is ${L.dsv3.toFixed(2)} at DeepSeek-V3, and keeps falling as C to the minus one sixth.`;
    this._fig.innerHTML = `<svg class="eps" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="${aria}">${B.join('')}</svg>`;
  }
}
if (typeof customElements !== 'undefined' && !customElements.get('dsv3-epscale')) customElements.define('dsv3-epscale', Dsv3Epscale);
