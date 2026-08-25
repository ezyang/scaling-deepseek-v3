// The anatomy composition: <dsv3-anatomy> = a vertical margin plan
// (<dsv3-anatomy-plan>, the top-level structure in the block diagram's own
// visual language) beside ONE full transformer block (<dsv3-layer kindtabs>),
// joined by a dashed expansion cone from the highlighted plan block to the
// diagram. Clicking a plan block or a tab flips the FFN column; the two stay
// in sync. The plan shows embedding → dense ×3 → MoE ×58 → final RMSNorm →
// lm head → softmax/loss with parameter counts — no MLA/FFN/residual
// internals (that's the block diagram's job).

import { DSV3 } from './model.js';
import { fmtP, tokensCss, applyHighlight } from './viewer.js';
import { PARAMS } from './params.js';

// named parameter quantities, shared with the diagram's tabs (src/params.js)
const A = DSV3;
const { embed: E, mla: MLA, denseBlock: DENSE, moeBlock: MOE } = PARAMS;

// the block diagram's visual-language tokens, plus the plan's own bits
const CSS = `
dsv3-anatomy-plan { display: block; }
.anp { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: #0b0b0b; }
.anp svg { display: block; max-width: 100%; height: auto; }
${tokensCss('.anp')}
.anp .box.on { fill: #fff8ea; stroke: #eda100; }
.anp [data-kind] { cursor: pointer; }
.anp [data-kind].on { cursor: default; }
.anp svg.hlm > :not(.hl):not(defs) { opacity: 0.3; }
/* the plan's tally-highlighted items wear the same save-yellow as the
   active block kind — grey pills alone were too understated */
.anp g[data-op].hl rect { fill: #fff8ea; stroke: #eda100; }
.anp g[data-op].hl .dims { fill: #52514e; font-weight: 600; }
`;

export class Dsv3AnatomyPlan extends HTMLElement {
  connectedCallback() {
    const style = document.createElement('style'); style.textContent = CSS;
    this._root = document.createElement('div'); this._root.className = 'anp';
    this.append(style, this._root);
    // overlay across the shared container: the angled "expansion" lines from
    // the highlighted plan block to the transformer-block diagram it expands into
    const host = this.parentElement;
    if (host && getComputedStyle(host).position === 'static') host.style.position = 'relative';
    this._ov = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    this._ov.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;overflow:visible;';
    host?.append(this._ov);
    this.draw();
    // re-sync the highlight when the layer flips from its own tabs
    queueMicrotask(() => {
      this.layerEl()?.addEventListener('recipe', () => this.draw());
      this.draw();
    });
    const up = () => this.expansion();
    window.addEventListener('resize', up);
    window.addEventListener('scroll', up, { passive: true });   // the plan is sticky
  }
  layerEl() { return document.getElementById(this.getAttribute('layer') ?? ''); }
  // dashed lines from the active plan box's right corners to the diagram
  // card's left corners — "this block expands into that diagram"
  expansion() {
    if (!this._ov) return;
    const src = this._root.querySelector('g.on rect');
    const dst = this.layerEl()?.querySelector('.lv');
    if (!src || !dst) { this._ov.innerHTML = ''; return; }
    const h = this._ov.getBoundingClientRect(), a = src.getBoundingClientRect(), b = dst.getBoundingClientRect();
    // flow-style cubics: horizontal out of the block, horizontal into the
    // card — the top line arcs up and the bottom arcs down, so they read as
    // two distinct edges of the expansion cone
    const L = (x1, y1, x2, y2) => {
      const g = Math.max(24, (x2 - x1) * 0.6);
      return `<path d="M ${x1 - h.left} ${y1 - h.top} C ${x1 - h.left + g} ${y1 - h.top}, ` +
        `${x2 - h.left - g} ${y2 - h.top}, ${x2 - h.left} ${y2 - h.top}" ` +
        `fill="none" stroke="#c3c2b7" stroke-width="1.2" stroke-dasharray="5 4"/>`;
    };
    this._ov.innerHTML = L(a.right, a.top, b.left, b.top) + L(a.right, a.bottom, b.left, b.bottom);
  }
  draw() {
    const kind = this.layerEl()?.kind ?? 'moe';
    const S = [];
    S.push(`<defs><marker id="planarr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">` +
      `<path d="M 0 0 L 8 4 L 0 8 z" fill="#898781"/></marker></defs>`);
    const BX = 8, W = 158, CX = BX + W / 2;   // spine runs through the box centers
    // named intermediates on the wires, in the diagram's tensor-label style
    const wire = (gap, name) => {
      S.push(`<line class="wire" x1="${CX}" y1="${y}" x2="${CX}" y2="${y + gap}" marker-end="url(#planarr)"/>`);
      if (name) S.push(`<text class="dims" x="${CX + 7}" y="${y + gap / 2 + 3}" font-style="italic">${name}</text>`);
      y += gap;
    };
    let y = 14;
    S.push(`<text class="oplabel" x="${CX - 22}" y="${y - 2}">tokens</text>`);
    wire(14);
    const op = (label, dims, h = 22, opId = null) => {
      S.push(`${opId ? `<g data-op="${opId}">` : ''}<rect class="op" x="${BX}" y="${y}" width="${W}" height="${h}" rx="6"/>` +
        `<text class="oplabel" x="${BX + 9}" y="${y + 15}">${label}${dims ? ` <tspan class="dims">${dims}</tspan>` : ''}</text>${opId ? '</g>' : ''}`);
      y += h;
    };
    const blockBox = (k, label, dims) => {
      const on = kind === k;
      S.push(`<g data-kind="${k}" data-op="block-${k}" class="${on ? 'on' : ''}">` +
        `<rect class="box${on ? ' on' : ''}" x="${BX}" y="${y}" width="${W}" height="34" rx="4"/>` +
        `<text class="name" x="${BX + 8}" y="${y + 14}">${label}</text>` +
        `<text class="dims" x="${BX + 8}" y="${y + 27}">${dims}</text></g>`);
      y += 34;
    };
    op('embedding', `(${fmtP(E)})`, 22, 'embed');
    wire(24, `x · ${A.hidden}`);
    blockBox('dense', `dense block ×${A.denseLayers}`, `${fmtP(DENSE)} each`);
    wire(24, `x · ${A.hidden}`);
    blockBox('moe', `MoE block ×${A.layers - A.denseLayers}`, `${fmtP(MOE)} each`);
    wire(24, `x · ${A.hidden}`);
    op('final RMSNorm', `(${fmtP(A.hidden)})`, 22, 'final_norm');
    wire(24, `norm out · ${A.hidden}`);
    S.push(`<g data-op="lm_head"><rect class="box" x="${BX}" y="${y}" width="${W}" height="34" rx="4"/>` +
      `<text class="name" x="${BX + 8}" y="${y + 14}">lm head</text>` +
      `<text class="dims" x="${BX + 8}" y="${y + 27}">${A.hidden} → ${A.vocab} (${fmtP(E)})</text></g>`);
    y += 34;
    wire(24, `logits · ${A.vocab}`);
    op('softmax / loss', null);
    y += 8;
    const H = y + 44, WD = BX + W + 10;
    this._root.innerHTML = `<svg viewBox="0 0 ${WD} ${H}" width="${WD}" height="${H}">${S.join('')}</svg>`;
    for (const g of this._root.querySelectorAll('[data-kind]')) {
      g.onclick = () => {
        const l = this.layerEl();
        if (!l || l.kind === g.dataset.kind) return;
        l.kind = g.dataset.kind; l.render(); l.changed(true);
        this.draw();
      };
    }
    this.applyHl();
    requestAnimationFrame(() => this.expansion());   // measure after layout settles
  }
  highlightOps(ids) { this._hl = ids ? new Set(ids) : null; this.applyHl(); }  // null clears, [] fades all
  applyHl() { applyHighlight(this._root, this._hl); }
}
customElements.define('dsv3-anatomy-plan', Dsv3AnatomyPlan);

// <dsv3-anatomy layer-id="..." [layer attrs...]>: the shipped composition.
// Forwards layer attributes to an inner <dsv3-layer kindtabs block-only>
// whose id is layer-id (so URL state, dsv3-controls layer= links, and page
// scripts keep working). The grid breaks out of a width-capped <main> so the
// diagram renders at natural size (it scales only below ~1330px viewports).
const ANAT_CSS = `
dsv3-anatomy { display: block; margin: 14px 0 26px; }
dsv3-anatomy .anat-grid { display: grid; grid-template-columns: 186px minmax(0, 1fr);
  gap: 0 28px; align-items: start; position: relative; left: 50%;
  transform: translateX(-50%); width: min(1330px, calc(100vw - 32px)); }
dsv3-anatomy dsv3-anatomy-plan { margin-top: 46px; }
`;
const FWD = ['controls', 'recipe', 'recompute', 'detail', 'transposed', 'for',
  'nocaption', 'kind', 'xlayers', 'xinflight'];
export class Dsv3Anatomy extends HTMLElement {
  connectedCallback() {
    const lid = this.getAttribute('layer-id') ?? ((this.id || 'anatomy') + '-layer');
    const style = document.createElement('style'); style.textContent = ANAT_CSS;
    const grid = document.createElement('div'); grid.className = 'anat-grid';
    const col1 = document.createElement('div');
    const plan = document.createElement('dsv3-anatomy-plan');
    plan.setAttribute('layer', lid);
    col1.append(plan);
    if (this.hasAttribute('tally')) {   // the parameter tally lives in the margin, below the plan
      const tal = document.createElement('dsv3-param-tally');
      tal.setAttribute('layer-id', lid);
      tal.setAttribute('compact', '');
      col1.append(tal);
    }
    const layer = document.createElement('dsv3-layer');
    layer.id = lid;
    layer.setAttribute('kindtabs', '');
    layer.setAttribute('block-only', '');
    for (const a of FWD) if (this.hasAttribute(a)) layer.setAttribute(a, this.getAttribute(a));
    grid.append(col1, layer);
    this.append(style, grid);
  }
}
customElements.define('dsv3-anatomy', Dsv3Anatomy);


// <dsv3-param-tally layer-id="..." [compact]>: the parameter count computed
// FROM the diagram, spreadsheet-style. Each row is a derived sum; clicking it
// highlights the diagram "cells" (boxes) whose grey parentheticals it sums —
// everything else fades (the tabs' visual language) — plus the plan box
// carrying its multiplier. Rows over the hidden FFN kind flip the diagram to
// that kind first, so the cells are always visible. compact = the narrow
// two-column form that <dsv3-anatomy tally> mounts in the margin below the
// plan. RMSNorm weights are counted; only the MTP module is omitted.
const T = PARAMS, Q = PARAMS;

const MLA_OPS = ['qkv_down', 'q_up', 'kv_up', 'o_proj', 'norm1', 'q_norm', 'kv_norm'];
const TALLY_ROWS = [
  { label: 'embedding', kind: null, ops: [], plan: ['embed'],
    formula: `${A.hidden} × ${A.vocab}`, per: E, count: 1, mult: '× 1' },
  { label: 'dense block', kind: 'dense',
    ops: [...MLA_OPS, 'ffn_gate_up', 'ffn_down', 'norm2'], plan: ['block-dense'],
    formula: `attn qkv ${fmtP(Q.attnQkv)} + attn out ${fmtP(Q.attnOut)} + ffn ${fmtP(Q.denseFfn)} + norms ${fmtP(Q.normsBlk)}`,
    per: DENSE, count: A.denseLayers, mult: `× ${A.denseLayers} blocks` },
  { label: 'MoE block', kind: 'moe',
    ops: [...MLA_OPS, 'router', 'ffn_gate_up', 'ffn_down', 'shared', 'norm2'], plan: ['block-moe'],
    formula: `attn qkv ${fmtP(Q.attnQkv)} + attn out ${fmtP(Q.attnOut)} + experts ${fmtP(Q.expert)} × (${A.routedExperts} routed + ${A.sharedExperts} shared) + router ${fmtP(T.router)} + norms ${fmtP(Q.normsBlk)}`,
    per: MOE, count: A.layers - A.denseLayers, mult: `× ${A.layers - A.denseLayers} blocks` },
  { label: 'final RMSNorm', kind: null, ops: [], plan: ['final_norm'],
    formula: `${A.hidden}`, per: A.hidden, count: 1, mult: '× 1' },
  { label: 'lm head', kind: null, ops: ['lm_head'], plan: ['lm_head'],
    formula: `${A.hidden} × ${A.vocab}`, per: E, count: 1, mult: '× 1' },
];
const TALLY_CSS = `
dsv3-param-tally { display: block; margin: 14px 0; }
.ptal { font: 13.5px system-ui, -apple-system, "Segoe UI", sans-serif; color: #0b0b0b; }
.ptal table { border-collapse: collapse; width: 100%; max-width: 760px; }
.ptal th, .ptal td { text-align: left; padding: 5px 12px 5px 7px; border-bottom: 1px solid #e1e0d9;
  font-variant-numeric: tabular-nums; vertical-align: top; }
.ptal th { color: #52514e; font-weight: 600; font-size: 12.5px; }
.ptal td.num { text-align: right; padding-right: 0; white-space: nowrap; }
.ptal .title, .ptal .note { padding-left: 7px; }
.ptal .formula { color: #898781; font-size: 12.5px; }
.ptal tbody tr { cursor: pointer; }
.ptal tbody tr:hover { background: #f7f6f1; }
.ptal tbody tr.sel { background: #fff; box-shadow: inset 3px 0 0 #52514e; }
.ptal tbody tr.sel td:first-child { font-weight: 600; }
.ptal tfoot td { font-weight: 600; border-bottom: none; }
.ptal .note { color: #898781; font-size: 12px; margin-top: 4px; }
.ptal.compact { font-size: 11.5px; }
.ptal.compact .title { font: 600 11px system-ui; color: #52514e; margin: 0 0 2px; }
.ptal.compact td { padding: 3px 6px 3px 7px; }
.ptal.compact .formula { font-size: 10px; display: block; }
.ptal.compact .note { font-size: 10px; font-style: italic; }
.ptal.compact .fxout { min-height: 52px; padding: 4px 0 0 7px; font-size: 10px;
  color: #52514e; line-height: 1.4; }   /* fixed slot: selecting never reflows */
`;
export class Dsv3ParamTally extends HTMLElement {
  connectedCallback() {
    const compact = this.hasAttribute('compact');
    const style = document.createElement('style'); style.textContent = TALLY_CSS;
    const root = document.createElement('div'); root.className = 'ptal' + (compact ? ' compact' : '');
    const total = TALLY_ROWS.reduce((t, r) => t + r.per * r.count, 0);
    root.innerHTML =
      (compact ? `<div class="title">parameters</div>` : '') +
      `<table>` +
      (compact ? '' : `<thead><tr><th>component</th><th>parameters, per copy</th>` +
        `<th>copies</th><th style="text-align:right">total</th></tr></thead>`) +
      `<tbody>` +
      TALLY_ROWS.map((r, i) => compact
        ? `<tr data-row="${i}"><td>${r.label}<span class="formula">${fmtP(r.per)} ${r.mult}</span></td>` +
          `<td class="num">${fmtP(r.per * r.count)}</td></tr>`
        : `<tr data-row="${i}"><td>${r.label}</td>` +
          `<td><span class="formula">${r.formula} =</span> ${fmtP(r.per)}</td>` +
          `<td>${r.mult}</td><td class="num">${fmtP(r.per * r.count)}</td></tr>`).join('') +
      `</tbody><tfoot><tr><td${compact ? '' : ' colspan="3"'}>total</td><td class="num">${fmtP(total)}</td></tr></tfoot></table>` +
      (compact ? `<div class="fxout"></div>` : '') +
      `<div class="note">${compact
        ? 'click a row to highlight what it sums · MTP omitted'
        : 'click a row to highlight the diagram cells it sums · only the MTP module is omitted'}</div>`;
    this.append(style, root);
    const lid = this.getAttribute('layer-id') ?? '';
    const layer = () => document.getElementById(lid);
    const plan = () => document.querySelector(`dsv3-anatomy-plan[layer="${lid}"]`);
    for (const tr of root.querySelectorAll('tbody tr')) {
      tr.onclick = () => {
        const r = TALLY_ROWS[+tr.dataset.row], on = !tr.classList.contains('sel');
        for (const t of root.querySelectorAll('tr.sel')) t.classList.remove('sel');
        const l = layer();
        if (on && r.kind && l && l.kind !== r.kind) {   // the cells live on the hidden FFN kind: flip to it
          l.kind = r.kind; l.render(); l.changed(true);
        }
        tr.classList.toggle('sel', on);
        const fx = root.querySelector('.fxout');
        if (fx) fx.textContent = on ? `= ${r.formula}` : '';
        l?.highlightOps?.(on ? r.ops : null);
        plan()?.highlightOps?.(on ? r.plan : null);
      };
    }
  }
}
customElements.define('dsv3-param-tally', Dsv3ParamTally);
