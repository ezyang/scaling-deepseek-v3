// DeepSeek-V4.1-Flash study widgets: the 01-style architecture diagram.
// <dsv41-anatomy> = margin plan + expansion cone + tabbed block diagram,
// mirroring <dsv3-anatomy>, but self-contained: V4.1's block (mHC residual
// streams, SWA + CSA2 global attention with cross-layer KV/index reuse, a
// grouped low-rank out-proj) doesn't fit the DSv3 renderer, so this module
// draws its own SVG in the same visual grammar (docs/diagram-grammar.md) and
// imports only stable helpers. Numbers come from src/dsv41model.js.

import { fmtP, tokensCss, applyHighlight } from './viewer.js';
import { C } from './theme.js';
import { DSV41, VPARAMS, KV_PER_TOKEN } from './dsv41model.js';

const A = DSV41, K = VPARAMS;
const PJ = A.heads * A.headDim;   // 32768

// the five attention kinds (tabs, plan blocks, tally rows share these)
export const KINDS = {
  swa: { label: 'SWA only', sub: `×${A.kinds.swa}`, n: A.kinds.swa, m: 0, where: 'layers 0–1' },
  full2: { label: 'Full · m=2', sub: `×${A.kinds.full2}`, n: A.kinds.full2, m: 2, where: 'encoder, layers 2 · 8 · 14' },
  full1: { label: 'Full · m=1', sub: `×${A.kinds.full1}`, n: A.kinds.full1, m: 1, where: 'decoder, layer 20' },
  reindex: { label: 'Reindex', sub: `×${A.kinds.reindex}`, n: A.kinds.reindex, m: 1, where: 'decoder, layers 24 · 28 · 32 · 36' },
  reuse: { label: 'Reuse', sub: `×${A.kinds.reuse}`, n: A.kinds.reuse, m: 0, where: '15 encoder (m=2) + 15 decoder (m=1)' },
};
const KIND_KEYS = Object.keys(KINDS);

// totals cross the fmtP=B ceiling in places; keep B up to 999
const fmtPB = (n) => n >= 1e12 ? (n / 1e12).toFixed(2) + 'T' : fmtP(n);

// URL-hash state, same mechanics as viewer.js (#v41:<id>=<json>)
const readState = (key) => {
  try { const v = new URLSearchParams(location.hash.slice(1)).get(key); return v ? JSON.parse(v) : null; }
  catch { return null; }
};
const writeState = (key, obj) => {
  const p = new URLSearchParams(location.hash.slice(1));
  p.set(key, JSON.stringify(obj));
  history.replaceState(null, '', '#' + p.toString());
};

// ---- <dsv41-layer> ---------------------------------------------------------
// One V4.1 transformer block: the attention column (tabbed by CSA2 mode) and
// the MoE FFN column, both wrapped in Single-Pass mHC's pre/post mixes.
// Static tier only: structure, dims, parameter parentheticals, params lens.

const KV_CSS = `
dsv41-layer { display: block; margin: 14px 0 26px; }
.kv { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-0b0b0b);
  border: 1px solid var(--c-e1e0d9); border-radius: 6px; background: var(--c-fcfcfb); padding: 10px 12px; position: relative; }
.kv-head { display: flex; align-items: center; gap: 8px; padding-bottom: 6px; color: var(--c-52514e); flex-wrap: wrap; }
.kv-head button { font: 11px ui-monospace, monospace; padding: 2px 8px; border: 1px solid var(--c-c3c2b7);
  border-radius: 4px; background: var(--c-ffffff); color: var(--c-0b0b0b); cursor: pointer; margin-left: 8px; min-width: 8ch; box-sizing: content-box; }
.kv-head .where { margin-left: auto; font-size: 11px; color: var(--c-898781); font-style: italic; }
.kv svg { display: block; margin: 0 auto; }
.kv-scroll { overflow-x: auto; -webkit-overflow-scrolling: touch; }
${tokensCss('.kv')}
.kv text.tensor { font: 10px system-ui; }
.kv .tidle { fill: var(--c-a8a69e); }
.kv .tdim { fill: var(--c-898781); font-weight: 400; }
.kv .micro { fill: var(--c-f7f6f1); stroke: var(--c-d8d6cb); }
.kv .microlabel { font: italic 10px system-ui; fill: var(--c-52514e); }
.kv .cache { fill: var(--c-fff8ea); stroke: var(--c-eda100); }
.kv .fp4 { fill: var(--c-d6408b); }
.kv .fp8 { fill: var(--c-7b2fa8); }
.kv svg.hlm > :not(.hl):not(defs) { opacity: 0.3; }
.kv g[data-op].hl .dims { fill: var(--c-52514e); font-weight: 600; }
`;

export class Dsv41Layer extends HTMLElement {
  connectedCallback() {
    const st = readState('v41:' + (this.id || 'layer'));
    const k = st?.k ?? this.getAttribute('kind');
    this.kind = KIND_KEYS.includes(k) ? k : 'full2';
    this.flatDims = st ? !!st.flat : this.hasAttribute('flat');
    this.detail = this.hasAttribute('detail');
    this.activeView = null;   // null | 'decode' | 'prefill' (the tally's toggle)
    const style = document.createElement('style'); style.textContent = KV_CSS;
    this._card = document.createElement('div'); this._card.className = 'kv';
    this.append(style, this._card);
    this.render();
    addEventListener('dsv3-theme', () => this.render());
  }
  changed() {
    writeState('v41:' + (this.id || 'layer'), { k: this.kind, flat: this.flatDims ? 1 : 0 });
    this.dispatchEvent(new CustomEvent('recipe'));
  }
  highlightOps(ids) { this._hl = ids ? new Set(ids) : null; this.applyHl(); }
  applyHl() { if (this._card.querySelector('svg')) applyHighlight(this._card, this._hl); }
  render() {
    const P = this.getAttribute('lens') === 'params';   // params lens: counts, not dims
    const F = this.flatDims;
    const kind = this.kind, KD = KINDS[kind];
    const isFull = kind === 'full2' || kind === 'full1';
    const hasIdx = isFull || kind === 'reindex';
    const global = kind !== 'swa';
    const S = [];
    const W = 1090, H = 706;

    // ---- geometry ----------------------------------------------------------
    // attention column: two sub-columns (SWA-KV spine left, q spine right),
    // then the indexer column and the global-KV column (its spine on the
    // RIGHT so the index-K side path can sit left of it without a crossing)
    const RAIL1 = 34, C1 = 74, SX1 = 94, W1 = 280;      // attention column, left spine (SWA KV → attention → out-proj)
    const CQ = 224, SXQ = 244, WQ = 130;                // q sub-column
    const CI = 372, SXI = 392, WI = 150;                // indexer column
    const CG = 540, SXG = 670, WG = 150, GBR = 530;     // global-KV column (right spine) + its side-branch vertical
    const UPX = 702, RAIL2 = 714;                       // x1 → FFN top; x1 → final mix
    const C2 = 732, SX2 = 752, W2 = 214;                // FFN column
    const GATEX = 958;                                  // top-k weights rail
    const CSH = 966, WSH = 112, SXSH = 986;             // shared-expert sub-column
    // shared row anchors (kind flips preserve them)
    const yTop = 16, yPre = 40, yN1 = 72, yChip1 = 106, yTab = 118, yEnc = 144, yBr = 152;
    const yA = 164, yB = 204, yQf = 238, yC = 250, yD = 290, yE = 380, yAtt = 440,
      yInv = 494, yOA = 530, yOB = 578, yEncB = 618, yPost = 632, yX1 = 668;
    // FFN rows
    const yRout = 164, yRoutM = 208, yDisp = 240, yGrp = 286, yGU = 308, ySw = 360,
      yFD = 400, yGrpB = 444, yComb = 466, yMAdd = 520;

    // ---- drawing helpers (the grammar's tokens) ----------------------------
    const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
    const pk = (n, mult) => P || n ? ` <tspan class="dims">(${fmtP(n)}${mult ? ` ×${mult}` : ''})</tspan>` : '';
    const box = (id, x, y, w, name, dims, h = 32) =>
      S.push(`<g data-op="${id}"><rect class="box" x="${x}" y="${y}" width="${w}" height="${h}" rx="4"/>` +
        `<text class="name" x="${x + 8}" y="${y + 14}">${esc(name)}</text>` +
        (dims ? `<text class="dims" x="${x + 8}" y="${y + 26}">${dims}</text>` : '') + `</g>`);
    const pill = (id, x, y, w, label, h = 20, cls = 'op') =>
      S.push(`<g data-op="${id}"><rect class="${cls}" x="${x}" y="${y}" width="${w}" height="${h}" rx="6"/>` +
        `<text class="oplabel" x="${x + 9}" y="${y + 14}">${label}</text></g>`);
    const micro = (x, y, w, label, h = 16) =>
      S.push(`<rect class="micro" x="${x}" y="${y}" width="${w}" height="${h}" rx="6"/>` +
        `<text class="microlabel" x="${x + 9}" y="${y + 12}">${label}</text>`);
    const comm = (id, x, y, w, label, h = 22) =>
      S.push(`<g data-op="${id}"><rect class="comm" x="${x}" y="${y}" width="${w}" height="${h}" rx="6"/>` +
        `<text class="oplabel" x="${x + 9}" y="${y + 15}" fill="${C('#3f3288')}">${label}</text></g>`);
    const res = (id, sx, y, w, label) => {   // ⊕ on the spine + dashed label box to its right
      S.push(`<g data-op="${id}"><circle cx="${sx}" cy="${y + 10}" r="7" fill="${C('#fcfcfb')}" stroke="${C('#c3c2b7')}"/>` +
        `<text class="plus" x="${sx - 3.5}" y="${y + 14}">+</text>` +
        `<rect class="res" x="${sx + 14}" y="${y}" width="${w}" height="20" rx="6"/>` +
        `<text class="oplabel" x="${sx + 23}" y="${y + 14}">${label}</text></g>`);
    };
    const wv = (x, y1, y2, arrow = true) =>
      S.push(`<line class="wire" x1="${x}" y1="${y1}" x2="${x}" y2="${y2}"${arrow ? ' marker-end="url(#v41arr)"' : ''}/>`);
    const poly = (pts, arrow = true) =>
      S.push(`<polyline class="wire" points="${pts.map(p => p.join(',')).join(' ')}"${arrow ? ' marker-end="url(#v41arr)"' : ''} fill="none"/>`);
    const dot = (x, y) => S.push(`<circle cx="${x}" cy="${y}" r="2.5" fill="${C('#898781')}"/>`);
    const chip = (x, y, name, dims, anchor) => {
      if (P) return;   // the params lens drops tensor chips (01's convention)
      S.push(`<text class="tensor tidle" x="${x}" y="${y}"${anchor ? ` text-anchor="${anchor}"` : ''}>· ${esc(name)}${dims ? ` <tspan class="tdim">· ${dims}</tspan>` : ''}</text>`);
    };
    const grp = (x, y, w, h, label, lx = 10) =>
      S.push(`<rect class="grp" x="${x}" y="${y}" width="${w}" height="${h}" rx="6"/>` +
        (label ? `<text class="grplabel" x="${x + lx}" y="${y + 13}">${label}</text>` : ''));
    // tab flaps: the active tab fuses into the enclosure below it
    const tab = (x, w, key, cur, label, sub) => {
      const on = cur === key, r = 6, y0 = yTab, y1 = yTab + 26;
      const shape = `M ${x} ${y1} L ${x} ${y0 + r} Q ${x} ${y0} ${x + r} ${y0} L ${x + w - r} ${y0} Q ${x + w} ${y0} ${x + w} ${y0 + r} L ${x + w} ${y1}`;
      return `<g data-tab="${key}" style="cursor:${on ? 'default' : 'pointer'}">` +
        (on
          ? `<path d="${shape} Z" fill="${C('#fcfcfb')}" stroke="none"/><path d="${shape}" fill="none" stroke="${C('#c3c2b7')}"/>`
          : `<path d="${shape} Z" fill="${C('#eeede7')}" stroke="${C('#d8d6cb')}"/>`) +
        `<text x="${x + 10}" y="${yTab + 17}" style="font:600 11px system-ui" fill="${C(on ? '#0b0b0b' : '#898781')}">${label}` +
        `<tspan style="font:10px system-ui" fill="${C(on ? '#898781' : '#a8a69e')}"> ${sub}</tspan></text></g>`;
    };
    // kind enclosure with a gap under the active tab (the tab's own edge shows)
    const enc = (x, y, w, h, tabX, tabW) =>
      S.push(`<path class="grp" d="M ${tabX} ${y} L ${x + 6} ${y} Q ${x} ${y} ${x} ${y + 6} ` +
        `L ${x} ${y + h - 6} Q ${x} ${y + h} ${x + 6} ${y + h} L ${x + w - 6} ${y + h} ` +
        `Q ${x + w} ${y + h} ${x + w} ${y + h - 6} L ${x + w} ${y + 6} Q ${x + w} ${y} ${x + w - 6} ${y} ` +
        `L ${tabX + tabW} ${y}" fill="none"/>`);
    const heads = (h, dd) => F ? String(h * dd) : `${h}×${dd}`;

    S.push(`<defs><marker id="v41arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" ` +
      `orient="auto-start-reverse"><path d="M 0 0 L 8 4 L 0 8 z" fill="${C('#898781')}"/></marker></defs>`);

    // ---- head: the residual stream (4 mHC copies) in, fork to the rail ------
    S.push(`<text class="oplabel" x="${SX1 + 14}" y="${yTop - 4}">x — residual stream (${A.hcMult} mHC copies × ${A.hidden})</text>`);
    wv(SX1, yTop, yPre);
    dot(SX1, yTop + 6);
    poly([[SX1, yTop + 6], [RAIL1, yTop + 6], [RAIL1, yPost + 10], [SX1 - 9, yPost + 10]], true);
    pill('hc_attn', C1, yPre, W1, `mHC pre-mix — Σ of ${A.hcMult} copies · coefs from x${pk(K.hcSub)}`);
    wv(SX1, yPre + 20, yN1);
    pill('norm1', C1, yN1, W1, `RMSNorm${pk(A.hidden)}`);
    wv(SX1, yN1 + 20, yA);
    chip(SX1 + 14, yChip1, 'norm1 out', A.hidden);
    // the side branch under the tab row: q sub-column, indexer weights, global KV
    dot(SX1, yBr);
    const wx = CI + WI - 8;   // the indexer's per-head-weights drop
    const brEnd = isFull ? SXG : hasIdx ? wx : SXQ;
    poly([[SX1, yBr], [brEnd, yBr]], false);
    dot(SXQ, yBr); wv(SXQ, yBr, yA);
    if (isFull) wv(SXG, yBr, yA);

    // ---- attention column tabs + enclosure ---------------------------------
    const tabs = []; let tx = C1 + 22;
    for (const k of KIND_KEYS) {
      const w = k === 'swa' ? 94 : k.startsWith('full') ? 106 : k === 'reindex' ? 92 : 88;
      tabs.push({ k, x: tx, w }); S.push(tab(tx, w, k, kind, KINDS[k].label, KINDS[k].sub)); tx += w + 6;
    }
    const at = tabs.find(t => t.k === kind);
    enc(C1 - 10, yEnc, CG + WG + 10 - (C1 - 10), yEncB - yEnc, at.x, at.w);

    // -- SWA KV path (every layer), left spine
    box('swa_kv', C1, yA, 140, 'SWA kv proj', P ? fmtP(K.kv) : `${A.hidden} → ${A.headDim}${pk(K.kv)}`);
    wv(SX1, yA + 32, yB);
    pill('kv_norm', C1, yB, 140, `RMSNorm${pk(A.headDim)}`, 18);
    wv(SX1, yB + 18, yC);
    chip(SX1 + 14, yB + 30, 'k = v latent', `${A.headDim}`);
    pill('rope_swa', C1, yC, 140, `RoPE (${A.ropeDim} of ${A.headDim})`);
    wv(SX1, yC + 20, yD);
    pill('swa_cache', C1, yD, 140, `KV cache · <tspan class="fp8">FP8</tspan> · ring ${A.window}`, 20, 'cache');
    wv(SX1, yD + 20, yAtt);
    chip(SX1 + 14, yD + 36, 'window kv', `${A.window}×${A.headDim}`);
    // -- q path, right sub-column
    box('q_down', CQ, yA, WQ, 'q down-proj', P ? fmtP(K.qDown) : `${A.hidden} → ${A.qRank}${pk(K.qDown)}`);
    wv(SXQ, yA + 32, yB);
    pill('q_norm', CQ, yB, WQ, `RMSNorm${pk(A.qRank)}`, 18);
    wv(SXQ, yB + 18, yC);
    if (hasIdx) { dot(SXQ, yQf); poly([[SXQ, yQf], [CI - 3, yQf]], true); }
    chip(SXQ + 14, yB + 30, 'q latent', `${A.qRank}`);
    box('q_up', CQ, yC, WQ, 'q up-proj', P ? fmtP(K.qUp) : `${A.qRank} → ${heads(A.heads, A.headDim)}${pk(K.qUp)}`);
    wv(SXQ, yC + 32, yD + 10);
    pill('rope_q', CQ, yD + 10, WQ, `RoPE (${A.ropeDim} of ${A.headDim})`);
    wv(SXQ, yD + 30, yAtt);
    chip(SXQ + 14, yD + 46, 'q', heads(A.heads, A.headDim));

    // -- indexer column (Full + Reindex): index q proj → score → top-512
    if (hasIdx) {
      box('idx_q', CI, yQf - 16, 108, 'index q proj', P ? fmtP(K.idxQ) : `${A.qRank} → ${heads(A.idxHeads, A.idxDim)}${pk(K.idxQ)}`);
      wv(SXI, yQf + 16, yE);
      chip(SXI + 14, yQf + 30, 'index q', heads(A.idxHeads, A.idxDim));
      // per-head weights: a 5120 → 32 projection of norm1 out, dropped from the branch row
      if (isFull) dot(wx, yBr);
      wv(wx, yBr, yE);
      chip(wx - 6, yE - 8, 'w', `${A.idxHeads}`, 'end');
      box('idx_score', CI, yE, WI, kind === 'reindex' ? `score in pool · top-${A.topk}` : `index score · top-${A.topk}`,
        P ? fmtP(K.idxW) : `w: ${A.hidden} → ${A.idxHeads}${pk(K.idxW)}`, 36);
      poly([[SXI, yE + 36], [SXI, yAtt + 8], [C1 + W1 + 3, yAtt + 8]], true);
      chip(SXI + 14, yAtt - 6, `top-${A.topk} indices`);
    } else if (kind === 'reuse') {
      if (!P) S.push(`<text class="tensor tidle" x="${CI}" y="${yAtt + 2}">· top-${A.topk} indices <tspan class="tdim">← last indexing layer</tspan></text>`);
      poly([[CI + 8, yAtt + 8], [C1 + W1 + 3, yAtt + 8]], true);
    }

    // -- global KV column: Full computes it; Reindex / Reuse read a source layer's cache
    if (isFull) {
      const m = KD.m;
      box('comp_kv', CG, yA, WG, m > 1 ? 'compress kv + gate proj' : 'global kv proj',
        P ? fmtP(K.compKv + (m > 1 ? K.compGate : 0)) : `${A.hidden} → ${A.headDim}${m > 1 ? ` + ${A.headDim}` : ''}${pk(K.compKv + (m > 1 ? K.compGate : 0))}`);
      wv(SXG, yA + 32, yB);
      pill('comp_norm', CG, yB, WG, `${m > 1 ? `pool ×${m} · ` : ''}RMSNorm${pk(K.compNorm)}`, 18);
      wv(SXG, yB + 18, yC + 6);
      dot(SXG, yB + 30);
      pill('rope_main', CG, yC + 6, WG, `RoPE (${A.ropeDim}) · FP4 quant`);
      wv(SXG, yC + 26, yD);
      pill('main_cache', CG, yD, WG, `main KV cache · <tspan class="fp4">FP4</tspan>`, 20, 'cache');
      wv(SXG, yD + 20, yAtt + 18, false);
      poly([[SXG, yAtt + 18], [C1 + W1 + 3, yAtt + 18]], true);
      chip(SXG - 6, yD + 36, `main kv · ${m > 1 ? 'N/2' : 'N'} entries`, A.headDim, 'end');
      // index K: projected from the pre-RoPE latent (a side box left of the spine)
      const yIK = yE - 40;
      poly([[SXG, yB + 30], [GBR, yB + 30], [GBR, yIK + 16], [CG - 3, yIK + 16]], true);
      box('idx_k', CG, yIK, 120, 'index k proj', P ? fmtP(K.idxK + K.idxKNorm) : `${A.headDim}→${A.idxDim} · RoPE${pk(K.idxK + K.idxKNorm)}`);
      wv(CG + 20, yIK + 32, yIK + 46);
      pill('idx_k_cache', CG, yIK + 46, 120, `index K · <tspan class="fp4">FP4</tspan>`, 18, 'cache');
      poly([[CG, yIK + 55], [CI + WI + 3, yIK + 55]], true);
    } else if (global) {
      const src = kind === 'reindex' ? `layer ${A.kvSource[3]}` : 'source layer';
      if (!P) {
        S.push(`<text class="tensor tidle" x="${CG}" y="${yD + 14}">· main KV <tspan class="tdim">← ${src} cache</tspan></text>`);
        if (kind === 'reindex') S.push(`<text class="tensor tidle" x="${CG}" y="${yE - 30}">· index K <tspan class="tdim">← ${src} cache</tspan></text>`);
      }
      poly([[CG + 60, yD + 20], [CG + 60, yAtt + 18], [C1 + W1 + 3, yAtt + 18]], true);
      if (kind === 'reindex') poly([[CG + 8, yE - 24], [CG + 8, yE + 10], [CI + WI + 3, yE + 10]], true);
    }

    // -- attention + the shared tail
    const attDims = kind === 'swa'
      ? `window ${A.window} · sink${pk(K.sink)}`
      : `window ${A.window} + top-${A.topk} main KV · sink${pk(K.sink)}`;
    box('attn', C1, yAtt, W1, `sparse attention · ${heads(A.heads, A.headDim)} · k = v latent`, attDims, 36);
    wv(SX1, yAtt + 36, yInv);
    chip(SX1 + 14, yInv - 6, 'attn out', heads(A.heads, A.headDim));
    pill('inv_rope', C1, yInv, W1, `inverse RoPE (${A.ropeDim} of ${A.headDim})`);
    wv(SX1, yInv + 20, yOA);
    box('o_a', C1, yOA, W1, `attn out-proj A (grouped ×${A.oGroups})`,
      P ? fmtP(K.oA) : `${A.oGroups}× (${F ? PJ / A.oGroups : `${A.heads / A.oGroups}×${A.headDim}`} → ${A.oRank}) · bf16${pk(K.oA)}`);
    wv(SX1, yOA + 32, yOB);
    box('o_b', C1, yOB, W1, 'attn out-proj B', P ? fmtP(K.oB) : `${A.oGroups}×${A.oRank} → ${A.hidden}${pk(K.oB)}`);
    wv(SX1, yOB + 32, yPost - 2);
    chip(SX1 + 14, yOB + 46, 'attn proj out', A.hidden);
    res('res1', SX1, yPost, 214, `mHC post-mix — post·y + comb(${A.hcMult}×${A.hcMult})·x`);
    wv(SX1, yPost + 17, yX1, false);
    chip(SX1 + 14, yX1 + 14, 'x1 (residual copies)', `${A.hcMult}×${A.hidden}`);
    // x1 → FFN column top (up and over), plus its rail to the final mix
    dot(SX1, yX1);
    poly([[SX1, yX1], [RAIL2, yX1], [RAIL2, yPost + 10], [SX2 - 9, yPost + 10]], true);
    dot(UPX, yX1);
    poly([[UPX, yX1], [UPX, 28], [SX2, 28], [SX2, yPre]], true);
    pill('hc_ffn', C2, yPre, W2, `mHC pre-mix · coefs from x1${pk(K.hcSub)}`);
    wv(SX2, yPre + 20, yN1);
    pill('norm2', C2, yN1, W2, `RMSNorm${pk(A.hidden)}`);
    wv(SX2, yN1 + 20, yRout);
    chip(SX2 + 14, yChip1, 'norm2 out', A.hidden);

    // ---- FFN column: one kind (MoE on every layer) -------------------------
    const AV = this.activeView != null;
    const EX = AV ? A.ffnTopk : A.routedExperts;
    const f1x = C2 + 24, f1w = 180;
    S.push(tab(f1x, f1w, 'moe', 'moe', 'MoE FFN', `×${A.layers} · ${fmtP(AV ? K.activeMoeFfn : K.moeFfn)}`));
    enc(C2 - 10, yEnc, CSH + WSH + 10 - (C2 - 10), yEncB - yEnc, f1x, f1w);
    dot(SX2, yBr);
    poly([[SX2, yBr], [SXSH, yBr], [SXSH, yGU]], true);
    box('router', C2, yRout, W2, 'router', P ? fmtP(K.routerWeight) : `${A.hidden} → ${A.routedExperts}${pk(K.routerWeight)}`);
    wv(SX2, yRout + 32, yRoutM);
    S.push(`<g data-op="router_bias">`);
    micro(C2, yRoutM, W2, `√softplus · bias (text|img) · top-${A.ffnTopk}${pk(K.routerBias)}`);
    S.push(`</g>`);
    wv(SX2, yRoutM + 16, yDisp);
    poly([[C2 + W2, yRoutM + 8], [GATEX, yRoutM + 8], [GATEX, ySw + 10], [C2 + W2 + 3, ySw + 10]], true);
    if (!P) S.push(`<text class="tensor tidle" text-anchor="end" x="${GATEX - 6}" y="${yDisp - 6}">· top-k weights <tspan class="tdim">· ${A.ffnTopk} · renorm · ×1.5</tspan></text>`);
    comm('dispatch', C2, yDisp, W2, 'a2a dispatch (permute + comm)');
    wv(SX2, yDisp + 22, yGU);
    chip(SX2 + 14, yDisp + 38, 'dispatched', `${A.ffnTopk}×${A.hidden}`);
    grp(C2 - 6, yGrp, W2 + 12, yGrpB - yGrp, `routed experts ×${A.routedExperts} · <tspan class="fp4">FP4</tspan> · ${fmtP(K.expert)} each`, SX2 + 20 - C2);
    box('gate_up', C2, yGU, W2, `ffn gate/up (grouped ×${A.ffnTopk})`,
      P ? `${fmtP(2 * K.expert / 3)} ×${EX}` : F ? `${A.hidden} → ${2 * A.moeInter} (${fmtP(2 * K.expert / 3 * EX)})` : `${A.hidden} → 2×${A.moeInter}${pk(2 * K.expert / 3, EX)}`);
    wv(SX2, yGU + 32, ySw);
    chip(SX2 + 14, yGU + 46, 'gate, up', `${A.ffnTopk}×2×${A.moeInter}`);
    pill('swiglu', C2, ySw, W2, 'SwiGLU (clamp 10) · × top-k weight');
    wv(SX2, ySw + 20, yFD);
    box('ffn_down', C2, yFD, W2, `ffn down (grouped ×${A.ffnTopk})`,
      P ? `${fmtP(K.expert / 3)} ×${EX}` : F ? `${A.moeInter} → ${A.hidden} (${fmtP(K.expert / 3 * EX)})` : `${A.moeInter} → ${A.hidden}${pk(K.expert / 3, EX)}`);
    wv(SX2, yFD + 32, yComb);
    chip(SX2 + 14, yGrpB + 12, 'expert outputs');
    comm('combine', C2, yComb, W2, 'a2a combine (comm + unpermute)');
    wv(SX2, yComb + 22, yMAdd - 2);
    // shared expert: every token, FP8, alongside the routed group
    grp(CSH - 6, yGrp, WSH + 12, yGrpB - yGrp);
    S.push(`<text class="grplabel" x="${CSH - 6}" y="${yGrp - 8}">shared expert · <tspan class="fp8">FP8</tspan></text>`);
    box('shared_gu', CSH, yGU, WSH, 'shared gate/up', P ? fmtP(2 * K.shared / 3) : `${A.hidden} → 2×${A.moeInter}${pk(2 * K.shared / 3)}`);
    wv(SXSH, yGU + 32, ySw);
    pill('shared_sw', CSH, ySw, WSH, 'SwiGLU');
    wv(SXSH, ySw + 20, yFD);
    box('shared_down', CSH, yFD, WSH, 'shared down', P ? fmtP(K.shared / 3) : `${A.moeInter} → ${A.hidden}${pk(K.shared / 3)}`);
    chip(SXSH + 8, yGrpB + 12, 'shared out');
    poly([[SXSH, yFD + 32], [SXSH, yMAdd + 10], [SX2 + 14 + 170 + 3, yMAdd + 10]], true);
    res('moe_add', SX2, yMAdd, 170, 'add — routed + shared');
    wv(SX2, yMAdd + 17, yPost - 2);
    chip(SX2 + 14, yMAdd + 34, 'ffn out', A.hidden);
    res('res2', SX2, yPost, 200, `mHC post-mix — post·y + comb·x1`);
    wv(SX2, yPost + 17, yPost + 36, false);
    chip(SX2 + 14, yPost + 50, 'x2 (block output)', `${A.hcMult}×${A.hidden}`);

    // ---- assemble ----------------------------------------------------------
    const head = document.createElement('div'); head.className = 'kv-head';
    const lab = document.createElement('span'); lab.textContent = 'sizes:';
    const b = document.createElement('button');
    b.textContent = this.flatDims ? String(PJ) : `${A.heads}×${A.headDim}`;
    b.title = `toggle sizes: factored (${A.heads}×${A.headDim}) vs multiplied out (${PJ})`;
    b.onclick = () => { this.flatDims = !this.flatDims; this.render(); this.changed(); };
    const where = document.createElement('span'); where.className = 'where';
    where.textContent = `${KD.label} attention · ${KD.where} · ${fmtP(K.block[kind] - K.moeFfn - K.misc)} attention params`;
    head.append(lab, b, where);
    const scroll = document.createElement('div'); scroll.className = 'kv-scroll';
    scroll.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${S.join('')}</svg>`;
    this._card.replaceChildren(head, scroll);
    for (const g of this._card.querySelectorAll('[data-tab]')) {
      g.addEventListener('click', () => {
        const key = g.dataset.tab;
        if (!KIND_KEYS.includes(key) || this.kind === key) return;
        this.kind = key; this.render(); this.changed();
      });
    }
    this.applyHl();
  }
}
customElements.define('dsv41-layer', Dsv41Layer);

// ---- <dsv41-anatomy-plan> --------------------------------------------------
// The margin plan: embedding → SWA ×2 → causal encoder (3 groups of Full +
// Reuse ×5) → decoder (Full + Reuse ×3, then [Reindex + Reuse ×3] ×4) → final
// RMSNorm → lm head. Engram sits before layers 1 and 14; the encoder's final
// hidden state feeds the decoder's one global KV (CED). Clicking a block
// flips the diagram's attention kind; the dashed cone ties plan to diagram.

const PLAN_CSS = `
dsv41-anatomy-plan { display: block; }
.vanp { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-0b0b0b); }
.vanp svg { display: block; max-width: 100%; height: auto; }
${tokensCss('.vanp')}
.vanp .box.on { fill: var(--c-fff8ea); stroke: var(--c-eda100); }
.vanp [data-kind] { cursor: pointer; }
.vanp [data-kind].on { cursor: default; }
.vanp svg.hlm > :not(.hl):not(defs) { opacity: 0.3; }
.vanp g[data-op].hl rect { fill: var(--c-fff8ea); stroke: var(--c-eda100); }
.vanp g[data-op].hl .dims { fill: var(--c-52514e); font-weight: 600; }
.vanp .eng { fill: var(--c-f3f1fb); stroke: var(--c-6b5bd2); }
`;

export class Dsv41AnatomyPlan extends HTMLElement {
  connectedCallback() {
    const style = document.createElement('style'); style.textContent = PLAN_CSS;
    this._root = document.createElement('div'); this._root.className = 'vanp';
    this.append(style, this._root);
    const host = this.parentElement;
    if (host && getComputedStyle(host).position === 'static') host.style.position = 'relative';
    this._ov = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    this._ov.setAttribute('class', 'anat-cone');
    this._ov.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;overflow:visible;';
    host?.append(this._ov);
    this.draw();
    queueMicrotask(() => {
      const le = this.layerEl();
      le?.addEventListener('recipe', () => this.draw());
      if (le && 'ResizeObserver' in window) new ResizeObserver(() => this.expansion()).observe(le);
      this.draw();
    });
    const up = () => this.expansion();
    window.addEventListener('resize', up);
    window.addEventListener('scroll', up, { passive: true });
    addEventListener('dsv3-theme', () => this.draw());
  }
  layerEl() { return document.getElementById(this.getAttribute('layer') ?? ''); }
  expansion() {
    if (!this._ov) return;
    const src = this._root.querySelector('g.on rect');
    const dst = this.layerEl()?.querySelector('.kv');
    if (!src || !dst) { this._ov.innerHTML = ''; return; }
    const h = this._ov.getBoundingClientRect(), a = src.getBoundingClientRect(), b = dst.getBoundingClientRect();
    const L = (x1, y1, x2, y2) => {
      const g = Math.max(24, (x2 - x1) * 0.6);
      return `<path d="M ${x1 - h.left} ${y1 - h.top} C ${x1 - h.left + g} ${y1 - h.top}, ` +
        `${x2 - h.left - g} ${y2 - h.top}, ${x2 - h.left} ${y2 - h.top}" ` +
        `fill="none" stroke="${C('#c3c2b7')}" stroke-width="1.2" stroke-dasharray="5 4"/>`;
    };
    this._ov.innerHTML = L(a.right, a.top, b.left, b.top) + L(a.right, a.bottom, b.left, b.bottom);
  }
  draw() {
    const l = this.layerEl();
    const kind = l?.kind ?? 'full2';
    const PL = l?.getAttribute('lens') === 'params';
    const S = [];
    S.push(`<defs><marker id="v41planarr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" ` +
      `orient="auto-start-reverse"><path d="M 0 0 L 8 4 L 0 8 z" fill="${C('#898781')}"/></marker></defs>`);
    const BX = 8, W = 158, CX = BX + W / 2;
    let y = 14;
    const wire = (gap, name) => {
      S.push(`<line class="wire" x1="${CX}" y1="${y}" x2="${CX}" y2="${y + gap}" marker-end="url(#v41planarr)"/>`);
      if (name) S.push(`<text class="dims" x="${CX + 7}" y="${y + gap / 2 + 3}" font-style="italic">${name}</text>`);
      y += gap;
    };
    const op = (label, dims, opId, cls = 'op') => {
      S.push(`${opId ? `<g data-op="${opId}">` : ''}<rect class="${cls}" x="${BX}" y="${y}" width="${W}" height="22" rx="6"/>` +
        `<text class="oplabel" x="${BX + 9}" y="${y + 15}">${label}${dims ? ` <tspan class="dims">${dims}</tspan>` : ''}</text>` +
        `${opId ? '</g>' : ''}`);
      y += 22;
    };
    const blockBox = (k, label, dims, x = BX, w = W, opId = `block-${k}`) => {
      const on = kind === k;
      S.push(`<g data-kind="${k}" data-op="${opId}" class="${on ? 'on' : ''}">` +
        `<rect class="box${on ? ' on' : ''}" x="${x}" y="${y}" width="${w}" height="34" rx="4"/>` +
        `<text class="name" x="${x + 8}" y="${y + 14}">${label}</text>` +
        `<text class="dims" x="${x + 8}" y="${y + 27}">${dims}</text></g>`);
      y += 34;
    };
    const note = (t) => { S.push(`<text class="dims" x="${BX}" y="${y + 10}" font-style="italic">${t}</text>`); y += 16; };
    const xs = `x · ${A.hcMult}×${A.hidden}`;
    S.push(`<text class="oplabel" x="${CX - 40}" y="${y - 2}">tokens · image patches</text>`);
    wire(14);
    op('embedding', `(${fmtP(K.embed)})`, 'embed');
    note(`images: ViT → aligner (${fmtP(K.vision)})`);
    wire(20, xs);
    blockBox('swa', 'SWA-only block', `layer 0 · ${fmtP(K.block.swa)}`);
    wire(16);
    op(`Engram · layer 1`, `(${fmtP(K.engram[0])})`, 'engram1', 'eng');
    wire(16);
    blockBox('swa', 'SWA-only block', `layer 1 · ${fmtP(K.block.swa)}`, BX, W, 'block-swa');
    wire(24, xs);
    // causal encoder: 3 groups of (Full m=2 + Reuse ×5), unrolled around the
    // second Engram module, which runs at the input of layer 14 = group 3's Full
    const encGroup = (label) => {
      const gTop = y; y += 20;
      blockBox('full2', 'CSA2 Full · m=2', `${fmtP(K.block.full2)}`, BX + 6, W - 12);
      wire(22, xs);
      blockBox('reuse', 'CSA2 Reuse · m=2 ×5', `${fmtP(K.block.reuse)} each`, BX + 6, W - 12);
      y += 8;
      S.push(`<rect class="grp" x="${BX - 4}" y="${gTop}" width="${W + 8}" height="${y - gTop}" rx="6"/>` +
        `<text class="grplabel" x="${BX + 4}" y="${gTop + 14}">${label}</text>`);
    };
    encGroup('causal encoder ×2 (layers 2–13)');
    wire(16);
    op('Engram · layer 14', `(${fmtP(K.engram[1])})`, 'engram2', 'eng');
    wire(16);
    encGroup('causal encoder (layers 14–19)');
    wire(22, 'enc. hidden state');
    note('→ decoder global KV (CED)');
    // decoder: Full + Reuse ×3, then [Reindex + Reuse ×3] ×4
    let gTop = y; y += 20;
    blockBox('full1', 'CSA2 Full · m=1', `layer 20 · ${fmtP(K.block.full1)}`, BX + 6, W - 12);
    wire(22, xs);
    blockBox('reuse', 'CSA2 Reuse · m=1 ×3', `${fmtP(K.block.reuse)} each`, BX + 6, W - 12);
    y += 8;
    S.push(`<rect class="grp" x="${BX - 4}" y="${gTop}" width="${W + 8}" height="${y - gTop}" rx="6"/>` +
      `<text class="grplabel" x="${BX + 4}" y="${gTop + 14}">decoder (layers 20–23)</text>`);
    wire(18, xs);
    gTop = y; y += 20;
    blockBox('reindex', 'CSA2 Reindex · m=1', `${fmtP(K.block.reindex)}`, BX + 6, W - 12);
    wire(22, xs);
    blockBox('reuse', 'CSA2 Reuse · m=1 ×3', `${fmtP(K.block.reuse)} each`, BX + 6, W - 12);
    y += 8;
    S.push(`<rect class="grp" x="${BX - 4}" y="${gTop}" width="${W + 8}" height="${y - gTop}" rx="6"/>` +
      `<text class="grplabel" x="${BX + 4}" y="${gTop + 14}">×4 (layers 24–39)</text>`);
    wire(22, xs);
    op('final RMSNorm', `(${fmtP(K.finalNorm)})`, 'final_norm');
    wire(24, `norm out · ${A.hidden}`);
    S.push(`<g data-op="lm_head"><rect class="box" x="${BX}" y="${y}" width="${W}" height="34" rx="4"/>` +
      `<text class="name" x="${BX + 8}" y="${y + 14}">lm head</text>` +
      `<text class="dims" x="${BX + 8}" y="${y + 27}">${PL ? `(${fmtP(K.embed)})` : `${A.hidden} → ${A.vocab} (${fmtP(K.embed)})`}</text></g>`);
    y += 34;
    wire(24, `logits · ${A.vocab}`);
    op('softmax / loss', null);
    y += 6;
    note(`DSpark drafter (${fmtP(K.dspark)}) not shown`);
    y += 8;
    this._root.innerHTML = `<svg viewBox="0 0 ${BX + W + 10} ${y + 30}" width="${BX + W + 10}" height="${y + 30}">${S.join('')}</svg>`;
    for (const g of this._root.querySelectorAll('[data-kind]')) {
      g.onclick = () => {
        const l = this.layerEl();
        if (!l || l.kind === g.dataset.kind) return;
        l.kind = g.dataset.kind; l.render(); l.changed();
        this.draw();
      };
    }
    this.applyHl();
    requestAnimationFrame(() => this.expansion());
  }
  highlightOps(ids) { this._hl = ids ? new Set(ids) : null; this.applyHl(); }
  applyHl() { applyHighlight(this._root, this._hl); }
}
customElements.define('dsv41-anatomy-plan', Dsv41AnatomyPlan);

// ---- <dsv41-param-tally> ---------------------------------------------------
// Same interaction model as <dsv3-param-tally> (hover previews, click pins,
// term-level highlights, exact-value tooltips). Three modes: total, active
// per decoded token, active per prefill token (CED runs only the encoder).

const ATTN_SWA = ['swa_kv', 'kv_norm', 'rope_swa', 'swa_cache'];
const ATTN_Q = ['q_down', 'q_norm', 'q_up', 'rope_q'];
const ATTN_OUT = ['attn', 'inv_rope', 'o_a', 'o_b'];
const ATTN_CORE = [...ATTN_SWA, ...ATTN_Q, ...ATTN_OUT];
const COMP = ['comp_kv', 'comp_norm', 'rope_main', 'main_cache'];
const IDX_K = ['idx_k', 'idx_k_cache'];
const IDX_Q = ['idx_q', 'idx_score'];
const MOE_EXPERTS = ['gate_up', 'ffn_down', 'shared_gu', 'shared_down'];
const NORM_OPS = ['norm1', 'norm2', 'hc_attn', 'hc_ffn', 'res1', 'res2'];
const attnQ = K.qDown + K.qNorm + K.qUp, attnKv = K.kv + K.kvNorm, attnOut = K.oA + K.oB + K.sink;
const mkAttnTerms = () => [
  { name: 'attn q (down·norm·up)', nv: attnQ, ops: ATTN_Q },
  { name: 'SWA kv (proj·norm)', nv: attnKv, ops: ATTN_SWA },
  { name: 'attn out (grouped A·B·sink)', nv: attnOut, ops: ATTN_OUT },
];
const mkMoeTerms = (active) => [
  { name: `experts (${active ? `${A.ffnTopk} active` : `${A.routedExperts} routed`} + ${A.sharedExperts} shared)`,
    nv: (active ? K.activeRouted : K.routed) + K.shared, ops: MOE_EXPERTS },
  { name: 'router', nv: K.routerWeight, ops: ['router'] },
  { name: 'router biases (text · image)', nv: K.routerBias, ops: ['router_bias'] },
  { name: 'mHC coefs + norms', nv: K.misc, ops: NORM_OPS },
];
const blockRow = (label, kind, plan, extraTerms, extra, count, mult, prefillPer, prefillTerms) => ({
  label, kind, plan,
  terms: [...mkAttnTerms(), ...extraTerms, ...mkMoeTerms(false)],
  per: K.block[kind], count, mult,
  decode: { per: K.activeBlock[kind], terms: [...mkAttnTerms(), ...extraTerms, ...mkMoeTerms(true)] },
  prefill: prefillPer == null ? { per: K.activeBlock[kind], terms: [...mkAttnTerms(), ...extraTerms, ...mkMoeTerms(true)] }
    : { per: prefillPer, terms: prefillTerms },
});
const compTerm = (m) => ({ name: m > 1 ? 'compressor (kv·gate·norm)' : 'global kv (proj·norm)', nv: K.compKv + (m > 1 ? K.compGate : 0) + K.compNorm, ops: COMP });
const idxKTerm = { name: 'index k (proj·norm)', nv: K.idxK + K.idxKNorm, ops: IDX_K };
const idxQTerm = { name: 'indexer (q proj · weights)', nv: K.indexer, ops: IDX_Q };
const notRun = (what) => [{ name: what, val: 'not run in prefill', nv: 0, ops: [] }];
const TALLY_ROWS = [
  { label: 'embedding', kind: null, plan: ['embed'],
    terms: [{ name: 'lookup table', val: `${A.hidden} × ${A.vocab}`, nv: K.embed, ops: [] }],
    per: K.embed, count: 1, mult: '× 1',
    decode: { per: 0, terms: [{ name: 'lookup', val: 'not counted', nv: 0, ops: [] }] },
    prefill: { per: 0, terms: [{ name: 'lookup', val: 'not counted', nv: 0, ops: [] }] } },
  blockRow('SWA-only block', 'swa', ['block-swa'], [], 0, A.kinds.swa, `× ${A.kinds.swa} (layers 0–1)`),
  blockRow('encoder block, CSA2 Full m=2', 'full2', ['block-full2'], [compTerm(2), idxKTerm, idxQTerm], K.full2, A.kinds.full2, `× ${A.kinds.full2} (layers 2 · 8 · 14)`),
  blockRow('encoder block, CSA2 Reuse', 'reuse', ['block-reuse'], [], 0, A.encLayers - A.kinds.swa - A.kinds.full2, `× ${A.encLayers - A.kinds.swa - A.kinds.full2}`),
  blockRow('decoder block, CSA2 Full m=1', 'full1', ['block-full1'], [compTerm(1), idxKTerm, idxQTerm], K.full1, A.kinds.full1, `× ${A.kinds.full1} (layer 20)`,
    K.compKv + K.compNorm + K.idxK + K.idxKNorm, [compTerm(1), idxKTerm, ...notRun('the rest of the block')]),
  blockRow('decoder block, CSA2 Reindex', 'reindex', ['block-reindex'], [idxQTerm], K.reindex, A.kinds.reindex, `× ${A.kinds.reindex}`, 0, notRun('decoder block')),
  blockRow('decoder block, CSA2 Reuse', 'reuse', ['block-reuse'], [], 0, A.decLayers - A.kinds.full1 - A.kinds.reindex, `× ${A.decLayers - A.kinds.full1 - A.kinds.reindex}`, 0, notRun('decoder block')),
  { label: 'final norm', kind: null, plan: ['final_norm'],
    terms: [{ name: 'RMSNorm', nv: K.finalNorm, ops: [] }], per: K.finalNorm, count: 1, mult: '× 1',
    prefill: { per: 0, terms: notRun('final norm') } },
  { label: 'lm head', kind: null, plan: ['lm_head'],
    terms: [{ name: 'output matrix', val: `${A.hidden} × ${A.vocab}`, nv: K.embed, ops: [] }], per: K.embed, count: 1, mult: '× 1',
    prefill: { per: 0, terms: notRun('lm head') } },
  ...A.engramLayers.map((L, i) => ({
    label: `Engram · layer ${L}`, kind: null, plan: [`engram${i + 1}`], engram: true,
    terms: [
      { name: 'hash tables (FP8)', val: `${A.engramRows[i].toLocaleString('en-US')} × ${A.engramDim}`, nv: K.engramTables[i], ops: [] },
      { name: 'kv proj + gate weights', nv: K.engramProj, ops: [] },
    ],
    per: K.engram[i], count: 1, mult: '× 1',
    decode: { per: K.engramProj, terms: [{ name: 'table rows', val: 'lookup, not counted', nv: 0, ops: [] }, { name: 'kv proj + gate weights', nv: K.engramProj, ops: [] }] },
    prefill: { per: K.engramProj, terms: [{ name: 'table rows', val: 'lookup, not counted', nv: 0, ops: [] }, { name: 'kv proj + gate weights', nv: K.engramProj, ops: [] }] },
  })),
];
for (const r of TALLY_ROWS) {
  for (const t of r.terms.concat(r.decode?.terms ?? [], r.prefill?.terms ?? [])) if (!t.val) t.val = fmtP(t.nv);
  r.ops = [...new Set(r.terms.flatMap(t => t.ops))];
  for (const m of ['decode', 'prefill']) if (r[m]) r[m].ops = [...new Set(r[m].terms.flatMap(t => t.ops))];
}
const rowIn = (r, mode) => mode !== 'total' && r[mode] ? { ...r, ...r[mode] } : r;

const TALLY_CSS = `
dsv41-param-tally { display: block; margin: 14px 0; }
.ptal { font: 13.5px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-0b0b0b); position: relative; }
.ptal .pnum { cursor: pointer; }
.ptal .ptip { display: none; position: absolute; z-index: 6; background: var(--c-ffffff);
  border: 1px solid var(--c-c3c2b7); border-radius: 4px; padding: 2px 8px; font: 11px ui-monospace, Menlo, monospace;
  box-shadow: 0 2px 8px rgba(11,11,11,0.12); pointer-events: none; white-space: nowrap; }
.ptal table { border-collapse: collapse; width: 100%; max-width: 760px; }
.ptal th, .ptal td { text-align: left; padding: 5px 12px 5px 7px; border-bottom: 1px solid var(--c-e1e0d9);
  font-variant-numeric: tabular-nums; vertical-align: top; }
.ptal th { color: var(--c-52514e); font-weight: 600; font-size: 12.5px; }
.ptal td.num { text-align: right; padding-right: 0; white-space: nowrap; }
.ptal .title { font: 600 12px system-ui; color: var(--c-52514e); margin: 0 0 4px; }
.ptal .formula { color: var(--c-898781); font-size: 12.5px; }
.ptal td .fterm { border-bottom: 1px dotted var(--c-c3c2b7); }
.ptal .fterm:hover { color: var(--c-0b0b0b); }
.ptal td .fterm:hover { border-bottom-color: var(--c-52514e); }
.ptal .fterm.pin { color: var(--c-0b0b0b); font-weight: 600; }
.ptal td .fterm.pin { border-bottom: 1px solid var(--c-52514e); }
.ptal .fxline { display: grid; grid-template-columns: 10px 1fr auto; gap: 0 6px; align-items: baseline; }
.ptal .fxline .fxop { color: var(--c-a8a69e); }
.ptal .fxline .fxval { text-align: right; font-variant-numeric: tabular-nums; }
.ptal tbody tr { cursor: pointer; }
.ptal tbody tr:hover { background: var(--c-f7f6f1); }
.ptal tbody tr.sel { background: var(--c-ffffff); box-shadow: inset 3px 0 0 var(--c-52514e); }
.ptal tbody tr.sel td:first-child { font-weight: 600; }
.ptal tbody tr.eng td { color: var(--c-52514e); }
.ptal tfoot td { font-weight: 600; border-bottom: none; }
.ptal tfoot tr.with td { font-weight: 400; color: var(--c-52514e); padding-top: 0; }
.ptal .mbtn { font-weight: 400; color: var(--c-898781); cursor: pointer; border-bottom: 1px dotted var(--c-c3c2b7); }
.ptal .mbtn.on { font-weight: 600; color: var(--c-0b0b0b); border-bottom: 1px solid var(--c-52514e); cursor: default; }
.ptal.compact { font-size: 11.5px; }
.ptal.compact .title { font: 600 11px system-ui; color: var(--c-52514e); margin: 0 0 2px; }
.ptal.compact td { padding: 3px 6px 3px 7px; }
.ptal.compact .formula { font-size: 10px; display: block; }
.ptal.compact .fxout { min-height: 150px; padding: 4px 0 0 7px; font-size: 10px;
  color: var(--c-52514e); line-height: 1.5; }
`;

export class Dsv41ParamTally extends HTMLElement {
  connectedCallback() {
    const m = this.getAttribute('mode');
    this._mode = ['decode', 'prefill'].includes(m) ? m : 'total';
    const style = document.createElement('style'); style.textContent = TALLY_CSS;
    this._root = document.createElement('div');
    this._root.className = 'ptal' + (this.hasAttribute('compact') ? ' compact' : '');
    this.append(style, this._root);
    this.build();
  }
  build() {
    const compact = this.hasAttribute('compact');
    const mode = this._mode;
    const root = this._root;
    const rows = TALLY_ROWS.map(r => rowIn(r, mode));
    // total: the backbone headline (552B) with Engram as a second line; the
    // active modes count Engram's projections in (the advertised 16B / 8B)
    const engram = rows.filter(r => r.engram).reduce((t, r) => t + r.per * r.count, 0);
    const backbone = rows.reduce((t, r) => t + r.per * r.count, 0) - (mode === 'total' ? engram : 0);
    const num = (v) => `<span class="pnum" data-v="${v}">${fmtPB(v)}</span>`;
    const modeBtn = (m, label) =>
      `<span class="mbtn${mode === m ? ' on' : ''}" data-mode="${m}">${label}</span>`;
    const head = `parameters: ${modeBtn('total', 'total')} · ${modeBtn('decode', 'active / decoded token')} · ${modeBtn('prefill', 'active / prefill token')}`;
    const ttl = { total: 'backbone total', decode: 'active, decode', prefill: 'active, prefill' }[mode];
    root.innerHTML =
      `<div class="title">${head}</div><table>` +
      (compact ? '' : `<thead><tr><th>component</th><th>parameters, per copy</th>` +
        `<th>copies</th><th style="text-align:right">total</th></tr></thead>`) +
      `<tbody>` +
      rows.map((r, i) => compact
        ? `<tr data-row="${i}"${r.engram ? ' class="eng"' : ''}><td>${r.label}<span class="formula">${fmtP(r.per)} ${r.mult}</span></td>` +
          `<td class="num">${num(r.per * r.count)}</td></tr>`
        : `<tr data-row="${i}"${r.engram ? ' class="eng"' : ''}><td>${r.label}</td>` +
          `<td><span class="formula">${r.terms.map((t, j) => `<span class="fterm" data-t="${j}">${t.name} ${t.val}</span>`).join(' + ')} =</span> ${num(r.per)}</td>` +
          `<td>${r.mult}</td><td class="num">${num(r.per * r.count)}</td></tr>`).join('') +
      `</tbody><tfoot><tr class="bb"><td${compact ? '' : ' colspan="3"'}>${ttl}</td><td class="num">${num(backbone)}</td></tr>` +
      (mode === 'total' ? `<tr class="with"><td${compact ? '' : ' colspan="3"'}>with Engram</td><td class="num">${num(backbone + engram)}</td></tr>` : '') +
      `</tfoot></table>` +
      (compact ? `<div class="fxout"></div>` : '');
    const lid = this.getAttribute('layer') ?? '';
    const layer = () => document.getElementById(lid);
    const plan = () => document.querySelector(`dsv41-anatomy-plan[layer="${lid}"]`);
    const state = { pin: null, hover: null };
    const rowOf = (st) => rows[st.ri];
    const opsOf = (st) => st.ti == null ? rowOf(st).ops : rowOf(st).terms[st.ti].ops;
    const termsHtml = (r) => r.terms.map((t, i) =>
      `<div class="fterm fxline" data-t="${i}"><span class="fxop">${i ? '+' : '='}</span>` +
      `<span class="fxname">${t.name}</span><span class="fxval">${t.val}</span></div>`).join('');
    const wireTerms = (container, ri, inRow = true) => {
      for (const sp of container.querySelectorAll('.fterm')) {
        const ti = +sp.dataset.t;
        sp.onmouseenter = () => { state.hover = { ri, ti }; apply(); };
        sp.onmouseleave = () => { state.hover = inRow ? { ri, ti: null } : null; apply(); };
        sp.onclick = (ev) => {
          ev.stopPropagation();
          if (state.pin?.ri === ri && state.pin?.ti === ti) state.pin = { ri, ti: null };
          else pinTo(ri, ti);
          apply();
        };
      }
    };
    const pinTo = (ri, ti) => {
      state.pin = { ri, ti };
      const r = rows[ri], l = layer();
      if (!l) return;
      if (r.kind && l.kind !== r.kind) { l.kind = r.kind; l.render(); l.changed(); }
    };
    const apply = () => {
      const cur = state.hover ?? state.pin;
      layer()?.highlightOps?.(cur ? opsOf(cur) : null);
      plan()?.highlightOps?.(cur ? rowOf(cur).plan : null);
      for (const tr of root.querySelectorAll('tbody tr'))
        tr.classList.toggle('sel', state.pin?.ri === +tr.dataset.row);
      const fx = root.querySelector('.fxout');
      if (fx) {
        const show = state.hover ?? state.pin;
        const want = show ? String(show.ri) : '';
        if (fx.dataset.ri !== want) {
          fx.dataset.ri = want;
          fx.innerHTML = want === '' ? '' : termsHtml(rowOf(show));
          if (want !== '') wireTerms(fx, show.ri, false);
        }
      }
      for (const sp of root.querySelectorAll('.fterm')) {
        const ri = sp.closest('tr') ? +sp.closest('tr').dataset.row : state.pin?.ri;
        sp.classList.toggle('pin', state.pin != null && state.pin.ti != null &&
          state.pin.ri === ri && state.pin.ti === +sp.dataset.t);
      }
    };
    for (const tr of root.querySelectorAll('tbody tr')) {
      const ri = +tr.dataset.row;
      tr.onmouseenter = () => { state.hover = { ri, ti: null }; apply(); };
      tr.onmouseleave = () => { state.hover = null; apply(); };
      tr.onclick = () => {
        if (state.pin?.ri === ri && state.pin?.ti == null) state.pin = null;
        else pinTo(ri, null);
        apply();
      };
      wireTerms(tr, ri);
    }
    const tip = document.createElement('div'); tip.className = 'ptip';
    root.append(tip);
    let tipPin = false;
    const showTip = (sp, ev) => {
      tip.textContent = Number(sp.dataset.v).toLocaleString('en-US');
      const r = root.getBoundingClientRect();
      tip.style.left = Math.max(0, ev.clientX - r.left + 12) + 'px';
      tip.style.top = (ev.clientY - r.top + 14) + 'px';
      tip.style.display = 'block';
    };
    for (const sp of root.querySelectorAll('.pnum')) {
      sp.onmouseenter = (ev) => { if (!tipPin) showTip(sp, ev); };
      sp.onmousemove = (ev) => { if (!tipPin) showTip(sp, ev); };
      sp.onmouseleave = () => { if (!tipPin) tip.style.display = 'none'; };
      sp.onclick = (ev) => {
        ev.stopPropagation();
        tipPin = true; showTip(sp, ev);
        const tr = sp.closest('tbody tr');
        if (tr) pinTo(+tr.dataset.row, null);
        apply();
      };
    }
    if (this._dismiss) document.removeEventListener('click', this._dismiss);
    this._dismiss = () => { tipPin = false; tip.style.display = 'none'; };
    document.addEventListener('click', this._dismiss);
    for (const b of root.querySelectorAll('.mbtn')) {
      b.onclick = () => {
        if (this._mode === b.dataset.mode) return;
        this._mode = b.dataset.mode;
        const l = layer();
        if (l) {
          l.activeView = this._mode === 'total' ? null : this._mode;
          l.highlightOps?.(null); l.render(); l.changed();
        }
        plan()?.highlightOps?.(null);
        this.build();
      };
    }
  }
}
customElements.define('dsv41-param-tally', Dsv41ParamTally);

// ---- <dsv41-anatomy> -------------------------------------------------------
const ANAT_CSS = `
dsv41-anatomy { display: block; margin: 14px 0 26px; }
dsv41-anatomy .anat-grid { display: grid; grid-template-columns: 186px minmax(0, 1fr);
  gap: 0 28px; align-items: start; position: relative; left: 50%;
  transform: translateX(-50%); width: min(1330px, calc(100vw - 32px)); }
dsv41-anatomy .anat-grid > * { min-width: 0; }
dsv41-anatomy dsv41-anatomy-plan { margin-top: 46px; }
@media (max-width: 860px) {
  dsv41-anatomy .anat-grid { grid-template-columns: 1fr; gap: 18px 0; }
  dsv41-anatomy dsv41-anatomy-plan { margin-top: 0; }
  .anat-cone { display: none; }
}
`;
const FWD = ['detail', 'lens', 'nocaption', 'kind', 'flat'];
export class Dsv41Anatomy extends HTMLElement {
  connectedCallback() {
    const lid = this.getAttribute('layer') ?? ((this.id || 'dsv41-anatomy') + '-layer');
    const style = document.createElement('style'); style.textContent = ANAT_CSS;
    const grid = document.createElement('div'); grid.className = 'anat-grid';
    const col1 = document.createElement('div');
    const plan = document.createElement('dsv41-anatomy-plan');
    plan.setAttribute('layer', lid);
    col1.append(plan);
    if (this.hasAttribute('tally')) {
      const tal = document.createElement('dsv41-param-tally');
      tal.setAttribute('layer', lid);
      tal.setAttribute('compact', '');
      col1.append(tal);
    }
    const layer = document.createElement('dsv41-layer');
    layer.id = lid;
    for (const a of FWD) if (this.hasAttribute(a)) layer.setAttribute(a, this.getAttribute(a));
    grid.append(col1, layer);
    this.append(style, grid);
  }
}
customElements.define('dsv41-anatomy', Dsv41Anatomy);

export { KV_PER_TOKEN };
