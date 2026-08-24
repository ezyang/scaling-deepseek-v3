// The anatomy composition: <dsv3-anatomy> = a vertical margin plan
// (<dsv3-anatomy-plan>, the top-level structure in the block diagram's own
// visual language) beside ONE full transformer block (<dsv3-layer kindtabs>),
// joined by a dashed expansion cone from the highlighted plan block to the
// diagram. Clicking a plan block or a tab flips the FFN column; the two stay
// in sync. The plan shows embedding → dense ×3 → MoE ×58 → final RMSNorm →
// lm head → softmax/loss with parameter counts — no MLA/FFN/residual
// internals (that's the block diagram's job).

import { DSV3 } from './model.js';
import { fmtP, tokensCss } from './viewer.js';

// per-component parameter counts, derived from the architecture
const A = DSV3;
const E = A.hidden * A.vocab;
const MLA = A.hidden * (A.qRank + A.kvRank + A.qkRope)
  + A.qRank * A.heads * (A.qkNope + A.qkRope)
  + A.kvRank * A.heads * (A.qkNope + A.vHead)
  + A.heads * A.vHead * A.hidden;
const DENSE = MLA + 3 * A.hidden * A.denseInter;
const MOE = MLA + (A.routedExperts + A.sharedExperts) * 3 * A.hidden * A.moeInter
  + A.hidden * A.routedExperts;

// the block diagram's visual-language tokens, plus the plan's own bits
const CSS = `
dsv3-anatomy-plan { display: block; }
.anp { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: #0b0b0b; }
.anp svg { display: block; max-width: 100%; height: auto; }
${tokensCss('.anp')}
.anp .box.on { fill: #fff8ea; stroke: #eda100; }
.anp [data-kind] { cursor: pointer; }
.anp [data-kind].on { cursor: default; }
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
    const op = (label, dims, h = 22) => {
      S.push(`<rect class="op" x="${BX}" y="${y}" width="${W}" height="${h}" rx="6"/>` +
        `<text class="oplabel" x="${BX + 9}" y="${y + 15}">${label}${dims ? ` <tspan class="dims">${dims}</tspan>` : ''}</text>`);
      y += h;
    };
    const blockBox = (k, label, dims) => {
      const on = kind === k;
      S.push(`<g data-kind="${k}" class="${on ? 'on' : ''}">` +
        `<rect class="box${on ? ' on' : ''}" x="${BX}" y="${y}" width="${W}" height="34" rx="4"/>` +
        `<text class="name" x="${BX + 8}" y="${y + 14}">${label}</text>` +
        `<text class="dims" x="${BX + 8}" y="${y + 27}">${dims}</text></g>`);
      y += 34;
    };
    op('embedding', `(${fmtP(E)})`);
    wire(24, `x · ${A.hidden}`);
    blockBox('dense', `dense block ×${A.denseLayers}`, `${fmtP(DENSE)} each`);
    wire(24, `x · ${A.hidden}`);
    blockBox('moe', `MoE block ×${A.layers - A.denseLayers}`, `${fmtP(MOE)} each`);
    wire(24, `x · ${A.hidden}`);
    op('final RMSNorm', `(${fmtP(A.hidden)})`);
    wire(24, `norm out · ${A.hidden}`);
    S.push(`<rect class="box" x="${BX}" y="${y}" width="${W}" height="34" rx="4"/>` +
      `<text class="name" x="${BX + 8}" y="${y + 14}">lm head</text>` +
      `<text class="dims" x="${BX + 8}" y="${y + 27}">${A.hidden} → ${A.vocab} (${fmtP(E)})</text>`);
    y += 34;
    wire(24, `logits · ${A.vocab}`);
    op('softmax / loss', null);
    y += 8;
    S.push(`<text class="grplabel" x="${BX}" y="${y + 12}">click a block kind — the</text>` +
      `<text class="grplabel" x="${BX}" y="${y + 24}">diagram flips to match;</text>` +
      `<text class="grplabel" x="${BX}" y="${y + 36}">MTP not shown</text>`);
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
    requestAnimationFrame(() => this.expansion());   // measure after layout settles
  }
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
    const plan = document.createElement('dsv3-anatomy-plan');
    plan.setAttribute('layer', lid);
    const layer = document.createElement('dsv3-layer');
    layer.id = lid;
    layer.setAttribute('kindtabs', '');
    layer.setAttribute('block-only', '');
    for (const a of FWD) if (this.hasAttribute(a)) layer.setAttribute(a, this.getAttribute(a));
    grid.append(plan, layer);
    this.append(style, grid);
  }
}
customElements.define('dsv3-anatomy', Dsv3Anatomy);
