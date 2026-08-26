// The anatomy composition: <dsv3-anatomy> = a vertical margin plan
// (<dsv3-anatomy-plan>, the top-level structure in the block diagram's own
// visual language) beside ONE full transformer block (<dsv3-layer tabs>),
// joined by a dashed expansion cone from the highlighted plan block to the
// diagram. Clicking a plan block or a tab flips the FFN column; the two stay
// in sync. The plan shows embedding → dense ×3 → MoE ×58 → final RMSNorm →
// lm head → softmax/loss with parameter counts — no MLA/FFN/residual
// internals (that's the block diagram's job).

import { DSV3 } from './model.js';
import { fmtP, fmtBytes, tokensCss, applyHighlight } from './viewer.js';
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
    const l = this.layerEl();
    const kind = l?.kind ?? 'moe';
    const AV = !!l?.activeView;                                  // the tally's active/token toggle
    const LB = l?.getAttribute('lens') === 'param-bytes';        // bytes framing
    const KM = kind === 'dense' ? A.denseLayers : A.layers - A.denseLayers;
    // byte strips share the diagram's scale unit: the block's largest op
    // fills one row (× block count under cumulative). Block boxes get NO
    // strips — their bytes are shown expanded on the right (never double
    // count a byte anywhere in the figure).
    const ABS = LB && l?.getAttribute('strips') === 'absolute';
    const UNIT = PARAMS.largestOp[kind] * (l?.cumulative && !ABS ? KM : 1) / 30;
    const strip = (x, y, nParams) => {
      if (!LB) return '';
      const n = Math.round(nParams / UNIT);
      if (!n)   // nonzero but sub-square (e.g. the embedding under ×58): hollow trace
        return nParams ? `<rect x="${x}" y="${y}" width="4" height="3.5" fill="none" stroke="#2a78d6" stroke-width="0.8"/>` : '';
      let g = '';
      for (let i = 0; i < n; i++)
        g += `<rect x="${x + (i % 30) * 5}" y="${y + Math.floor(i / 30) * 5}" width="4" height="3.5" fill="#2a78d6"/>`;
      return g;
    };
    const pv = (n) => LB ? fmtBytes(n * 2) : fmtP(n);
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
    const op = (label, dims, h = 22, opId = null, stripN = 0) => {
      const hh = LB && stripN ? h + 8 : h;   // reserved strip row (stable across toggles)
      S.push(`${opId ? `<g data-op="${opId}">` : ''}<rect class="op" x="${BX}" y="${y}" width="${W}" height="${hh}" rx="6"/>` +
        `<text class="oplabel" x="${BX + 9}" y="${y + 15}">${label}${dims ? ` <tspan class="dims">${dims}</tspan>` : ''}</text>` +
        (LB && stripN ? strip(BX + 9, y + 20, stripN) : '') + `${opId ? '</g>' : ''}`);
      y += hh;
    };
    const blockBox = (k, label, dims) => {
      const on = kind === k;
      S.push(`<g data-kind="${k}" data-op="block-${k}" class="${on ? 'on' : ''}">` +
        `<rect class="box${on ? ' on' : ''}" x="${BX}" y="${y}" width="${W}" height="34" rx="4"/>` +
        `<text class="name" x="${BX + 8}" y="${y + 14}">${label}</text>` +
        `<text class="dims" x="${BX + 8}" y="${y + 27}">${dims}</text></g>`);
      y += 34;
    };
    op('embedding', AV && !LB ? '(not counted)' : `(${pv(E)})`, 22, 'embed', E);
    wire(24, `x · ${A.hidden}`);
    blockBox('dense', `dense block ×${A.denseLayers}`, `${pv(DENSE)} each`);
    wire(24, `x · ${A.hidden}`);
    blockBox('moe', `MoE block ×${A.layers - A.denseLayers}`, `${pv(AV && !LB ? PARAMS.activeMoeBlock : MOE)} each`);
    wire(24, `x · ${A.hidden}`);
    op('final RMSNorm', `(${pv(A.hidden)})`, 22, 'final_norm');
    wire(24, `norm out · ${A.hidden}`);
    S.push(`<g data-op="lm_head"><rect class="box" x="${BX}" y="${y}" width="${W}" height="${LB ? 42 : 34}" rx="4"/>` +
      `<text class="name" x="${BX + 8}" y="${y + 14}">lm head</text>` +
      `<text class="dims" x="${BX + 8}" y="${y + 27}">${A.hidden} → ${A.vocab} (${pv(E)})</text>` +
      (LB ? strip(BX + 8, y + 31, E) : '') + `</g>`);
    y += LB ? 42 : 34;
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

// <dsv3-anatomy layer="..." [layer attrs...]>: the shipped composition.
// Forwards layer attributes to an inner <dsv3-layer tabs scope="block">
// whose id is the layer attr (so URL state, dsv3-controls links, and page
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
  'nocaption', 'kind', 'xlayers', 'xinflight', 'lens', 'strips'];
export class Dsv3Anatomy extends HTMLElement {
  connectedCallback() {
    const lid = this.getAttribute('layer') ?? ((this.id || 'anatomy') + '-layer');
    const style = document.createElement('style'); style.textContent = ANAT_CSS;
    const grid = document.createElement('div'); grid.className = 'anat-grid';
    const col1 = document.createElement('div');
    const plan = document.createElement('dsv3-anatomy-plan');
    plan.setAttribute('layer', lid);
    col1.append(plan);
    if (this.hasAttribute('tally')) {   // the parameter tally lives in the margin, below the plan
      const tal = document.createElement('dsv3-param-tally');
      tal.setAttribute('layer', lid);
      tal.setAttribute('compact', '');
      if (this.getAttribute('lens') === 'param-bytes') tal.setAttribute('units', 'bytes');
      col1.append(tal);
    }
    const layer = document.createElement('dsv3-layer');
    layer.id = lid;
    layer.setAttribute('tabs', '');
    layer.setAttribute('scope', 'block');
    for (const a of FWD) if (this.hasAttribute(a)) layer.setAttribute(a, this.getAttribute(a));
    grid.append(col1, layer);
    this.append(style, grid);
  }
}
customElements.define('dsv3-anatomy', Dsv3Anatomy);


// <dsv3-param-tally layer="..." [compact]>: the parameter count computed
// FROM the diagram, spreadsheet-style. Each row is a derived sum; clicking it
// highlights the diagram "cells" (boxes) whose grey parentheticals it sums —
// everything else fades (the tabs' visual language) — plus the plan box
// carrying its multiplier. Rows over the hidden FFN kind flip the diagram to
// that kind first, so the cells are always visible. compact = the narrow
// two-column form that <dsv3-anatomy tally> mounts in the margin below the
// plan. RMSNorm weights and the router correction biases are counted. The
// auxiliary MTP module is outside this main-model diagram and tally.
const T = PARAMS, Q = PARAMS;

// each formula TERM carries the diagram cells it covers, so hovering a
// variable in the equation highlights exactly its boxes; a row's cells are
// the union of its terms'
const ATTN_QKV_OPS = ['qkv_down', 'q_up', 'kv_up'];
const NORM_OPS = ['norm1', 'q_norm', 'kv_norm', 'norm2'];
const TALLY_ROWS = [
  { label: 'embedding', kind: null, plan: ['embed'],
    terms: [{ name: 'lookup table', val: `${A.hidden} × ${A.vocab}`, nv: E, ops: [] }],
    per: E, count: 1, mult: '× 1',
    active: { per: 0,
      terms: [{ name: 'lookup', val: 'not counted', ops: [] }] } },
  { label: 'dense block', kind: 'dense', plan: ['block-dense'],
    terms: [
      { name: 'attn qkv', val: fmtP(Q.attnQkv), nv: Q.attnQkv, ops: ATTN_QKV_OPS },
      { name: 'attn out', val: fmtP(Q.attnOut), nv: Q.attnOut, ops: ['o_proj'] },
      { name: 'ffn', val: fmtP(Q.denseFfn), nv: Q.denseFfn, ops: ['ffn_gate_up', 'ffn_down'] },
      { name: 'norms', val: fmtP(Q.normsBlk), nv: Q.normsBlk, ops: NORM_OPS },
    ],
    per: DENSE, count: A.denseLayers, mult: `× ${A.denseLayers} blocks` },
  { label: 'MoE block', kind: 'moe', plan: ['block-moe'],
    terms: [
      { name: 'attn qkv', val: fmtP(Q.attnQkv), nv: Q.attnQkv, ops: ATTN_QKV_OPS },
      { name: 'attn out', val: fmtP(Q.attnOut), nv: Q.attnOut, ops: ['o_proj'] },
      { name: `experts × (${A.routedExperts} routed + ${A.sharedExperts} shared)`, val: fmtP(Q.expert), nv: Q.expert, ops: ['ffn_gate_up', 'ffn_down', 'shared'] },
      { name: 'router', val: fmtP(T.routerWeight), nv: T.routerWeight, ops: ['router'] },
      { name: 'correction bias', val: String(T.routerBias), nv: T.routerBias, ops: ['router_bias'] },
      { name: 'norms', val: fmtP(Q.normsBlk), nv: Q.normsBlk, ops: NORM_OPS },
    ],
    per: MOE, count: A.layers - A.denseLayers, mult: `× ${A.layers - A.denseLayers} blocks`,
    active: { per: PARAMS.activeMoeBlock,
      terms: [
        { name: 'attn qkv', val: fmtP(Q.attnQkv), nv: Q.attnQkv, ops: ATTN_QKV_OPS },
        { name: 'attn out', val: fmtP(Q.attnOut), nv: Q.attnOut, ops: ['o_proj'] },
        { name: `experts × (${A.topk} active + ${A.sharedExperts} shared)`, val: fmtP(Q.expert), nv: Q.expert, ops: ['ffn_gate_up', 'ffn_down', 'shared'] },
        { name: 'router', val: fmtP(T.routerWeight), nv: T.routerWeight, ops: ['router'] },
        { name: 'correction bias', val: String(T.routerBias), nv: T.routerBias, ops: ['router_bias'] },
        { name: 'norms', val: fmtP(Q.normsBlk), nv: Q.normsBlk, ops: NORM_OPS },
      ] } },
  { label: 'final RMSNorm', kind: null, plan: ['final_norm'],
    terms: [{ name: 'weight', val: String(A.hidden), nv: A.hidden, ops: [] }],
    per: A.hidden, count: 1, mult: '× 1' },
  { label: 'lm head', kind: null, plan: ['lm_head'],
    terms: [{ name: 'output matrix', val: `${A.hidden} × ${A.vocab}`, nv: E, ops: ['lm_head'] }],
    per: E, count: 1, mult: '× 1' },
];
for (const r of TALLY_ROWS) {
  for (const t of r.terms.concat(r.active?.terms ?? [])) t.t = `${t.name} ${t.val}`;
  r.ops = [...new Set(r.terms.flatMap(t => t.ops))];
  if (r.active) r.active.ops = [...new Set(r.active.terms.flatMap(t => t.ops))];
}
// resolve a row in the current mode ('total' | 'active')
const rowIn = (r, mode) => mode === 'active' && r.active ? { ...r, ...r.active } : r;
const TALLY_CSS = `
dsv3-param-tally { display: block; margin: 14px 0; }
.ptal { font: 13.5px system-ui, -apple-system, "Segoe UI", sans-serif; color: #0b0b0b; position: relative; }
.ptal .pnum { cursor: pointer; }
.ptal .ptip { display: none; position: absolute; z-index: 6; background: #fff;
  border: 1px solid #c3c2b7; border-radius: 4px; padding: 2px 8px; font: 11px ui-monospace, Menlo, monospace;
  box-shadow: 0 2px 8px rgba(11,11,11,0.12); pointer-events: none; white-space: nowrap; }
.ptal table { border-collapse: collapse; width: 100%; max-width: 760px; }
.ptal th, .ptal td { text-align: left; padding: 5px 12px 5px 7px; border-bottom: 1px solid #e1e0d9;
  font-variant-numeric: tabular-nums; vertical-align: top; }
.ptal th { color: #52514e; font-weight: 600; font-size: 12.5px; }
.ptal td.num { text-align: right; padding-right: 0; white-space: nowrap; }
.ptal .title, .ptal .note { padding-left: 7px; }
.ptal .formula { color: #898781; font-size: 12.5px; }
.ptal td .fterm { border-bottom: 1px dotted #c3c2b7; }
.ptal .fterm:hover { color: #0b0b0b; }
.ptal td .fterm:hover { border-bottom-color: #52514e; }
.ptal .fterm.pin { color: #0b0b0b; font-weight: 600; }
.ptal td .fterm.pin { border-bottom: 1px solid #52514e; }
.ptal .fxline { display: grid; grid-template-columns: 10px 1fr auto; gap: 0 6px; align-items: baseline; }
.ptal .fxline .fxop { color: #a8a69e; }
.ptal .fxline .fxval { text-align: right; font-variant-numeric: tabular-nums; }
.ptal tbody tr { cursor: pointer; }
.ptal tbody tr:hover { background: #f7f6f1; }
.ptal tbody tr.sel { background: #fff; box-shadow: inset 3px 0 0 #52514e; }
.ptal tbody tr.sel td:first-child { font-weight: 600; }
.ptal tfoot td { font-weight: 600; border-bottom: none; }
.ptal .note { color: #898781; font-size: 12px; margin-top: 4px; }
.ptal.compact { font-size: 11.5px; }
.ptal.compact .title { font: 600 11px system-ui; color: #52514e; margin: 0 0 2px; }
.ptal .title { font: 600 12px system-ui; color: #52514e; margin: 0 0 4px; }
.ptal .mbtn { font-weight: 400; color: #898781; cursor: pointer; border-bottom: 1px dotted #c3c2b7; }
.ptal .mbtn.on { font-weight: 600; color: #0b0b0b; border-bottom: 1px solid #52514e; cursor: default; }
.ptal.compact td { padding: 3px 6px 3px 7px; }
.ptal.compact .formula { font-size: 10px; display: block; }
.ptal.compact .note { font-size: 10px; font-style: italic; }
.ptal.compact .fxout { min-height: 96px; padding: 4px 0 0 7px; font-size: 10px;
  color: #52514e; line-height: 1.5; }   /* fixed slot sized for the six-line MoE equation */
`;
export class Dsv3ParamTally extends HTMLElement {
  connectedCallback() {
    this._mode = this.getAttribute('mode') === 'active' ? 'active' : 'total';
    const style = document.createElement('style'); style.textContent = TALLY_CSS;
    this._root = document.createElement('div');
    this._root.className = 'ptal' + (this.hasAttribute('compact') ? ' compact' : '');
    this.append(style, this._root);
    this.build();
  }
  build() {
    const compact = this.hasAttribute('compact');
    const bytes = this.getAttribute('units') === 'bytes';   // bf16 memory framing
    const mode = bytes ? 'total' : this._mode;   // active vs total doesn't change resident bytes
    const root = this._root;
    const rows = TALLY_ROWS.map(r => rowIn(r, mode));
    const total = rows.reduce((t, r) => t + r.per * r.count, 0);
    const fv = (v) => bytes ? fmtBytes(v * 2) : fmtP(v);
    const num = (v) => `<span class="pnum" data-v="${bytes ? v * 2 : v}">${fv(v)}</span>`;
    const tval = (t) => bytes && t.nv != null ? fmtBytes(t.nv * 2) : t.val;
    const modeBtn = (m, label) =>
      `<span class="mbtn${mode === m ? ' on' : ''}" data-mode="${m}">${label}</span>`;
    const head = bytes ? 'parameter memory (bf16)'
      : `parameters: ${modeBtn('total', 'total')} · ${modeBtn('active', 'active / token')}`;
    root.innerHTML =
      (compact ? `<div class="title">${head}</div>` : `<div class="title">${head}</div>`) +
      `<table>` +
      (compact ? '' : `<thead><tr><th>component</th><th>parameters, per copy</th>` +
        `<th>copies</th><th style="text-align:right">total</th></tr></thead>`) +
      `<tbody>` +
      rows.map((r, i) => compact
        ? `<tr data-row="${i}"><td>${r.label}<span class="formula">${fv(r.per)} ${r.mult}</span></td>` +
          `<td class="num">${num(r.per * r.count)}</td></tr>`
        : `<tr data-row="${i}"><td>${r.label}</td>` +
          `<td><span class="formula">${r.terms.map((t, j) => `<span class="fterm" data-t="${j}">${t.name} ${tval(t)}</span>`).join(' + ')} =</span> ${num(r.per)}</td>` +
          `<td>${r.mult}</td><td class="num">${num(r.per * r.count)}</td></tr>`).join('') +
      `</tbody><tfoot><tr><td${compact ? '' : ' colspan="3"'}>total</td><td class="num">${num(total)}</td></tr></tfoot></table>` +
      (compact ? `<div class="fxout"></div>` : '');
    const lid = this.getAttribute('layer') ?? '';
    const layer = () => document.getElementById(lid);
    const plan = () => document.querySelector(`dsv3-anatomy-plan[layer="${lid}"]`);
    // interaction model: HOVER previews a row's (or a single term's) cells,
    // CLICK pins them; the display always shows hover ?? pin. Pinning a row
    // of the hidden FFN kind flips the diagram (previews don't).
    const state = { pin: null, hover: null };          // {ri, ti|null}
    const rowOf = (st) => rows[st.ri];
    const opsOf = (st) => st.ti == null ? rowOf(st).ops : rowOf(st).terms[st.ti].ops;
    const termsHtml = (r) => r.terms.map((t, i) =>
      `<div class="fterm fxline" data-t="${i}"><span class="fxop">${i ? '+' : '='}</span>` +
      `<span class="fxname">${t.name}</span><span class="fxval">${tval(t)}</span></div>`).join('');
    const wireTerms = (container, ri, inRow = true) => {
      for (const sp of container.querySelectorAll('.fterm')) {
        const ti = +sp.dataset.t;
        sp.onmouseenter = () => { state.hover = { ri, ti }; apply(); };
        // fxout terms are NOT on a row: leaving must clear the hover, or the
        // pinned term's highlight is masked by a stranded row-level hover
        sp.onmouseleave = () => { state.hover = inRow ? { ri, ti: null } : null; apply(); };
        sp.onclick = (ev) => {
          ev.stopPropagation();
          if (state.pin?.ri === ri && state.pin?.ti === ti) state.pin = { ri, ti: null };  // unpin term, keep row
          else pinTo(ri, ti);
          apply();
        };
      }
    };
    const pinTo = (ri, ti) => {
      state.pin = { ri, ti };
      const r = rows[ri], l = layer();
      if (r.kind && l && l.kind !== r.kind) { l.kind = r.kind; l.render(); l.changed(true); }
    };
    const apply = () => {
      const cur = state.hover ?? state.pin;
      layer()?.highlightOps?.(cur ? opsOf(cur) : null);
      plan()?.highlightOps?.(cur ? rowOf(cur).plan : null);
      for (const tr of root.querySelectorAll('tbody tr'))
        tr.classList.toggle('sel', state.pin?.ri === +tr.dataset.row);
      const fx = root.querySelector('.fxout');
      if (fx) {   // the equation slot follows hover ?? pin (fixed height: no reflow)
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
        if (state.pin?.ri === ri && state.pin?.ti == null) state.pin = null;   // unpin
        else pinTo(ri, null);
        apply();
      };
      wireTerms(tr, ri);   // full-table formula terms
    }
    const tip = document.createElement('div'); tip.className = 'ptip';
    root.append(tip);
    let tipPin = false;
    // cursor-anchored (offsetLeft/Top of inline spans in table cells lands
    // far from the pointer): position from the mouse event, following it
    // while unpinned
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
        if (tr) pinTo(+tr.dataset.row, null);   // pin the highlights too
        apply();
      };
    }
    if (this._dismiss) document.removeEventListener('click', this._dismiss);
    this._dismiss = () => { tipPin = false; tip.style.display = 'none'; };
    document.addEventListener('click', this._dismiss);
    // the heading is the total/active toggle; switching rebuilds and clears
    for (const b of root.querySelectorAll('.mbtn')) {
      b.onclick = () => {
        if (this._mode === b.dataset.mode) return;
        this._mode = b.dataset.mode;
        const l = layer();
        if (l) {   // the diagram's parentheticals follow the toggle
          l.activeView = this._mode === 'active';
          l.highlightOps?.(null); l.render(); l.changed(true);
        }
        plan()?.highlightOps?.(null);
        this.build();
      };
    }
  }
}
customElements.define('dsv3-param-tally', Dsv3ParamTally);
