// <dsv3-anatomy-plan layer="id">: EXPERIMENT — the top-level plan as a
// VERTICAL margin strip in the block diagram's own visual language (same
// box/op/wire/name/dims styling): embedding → dense block ×3 → MoE block ×58
// → final RMSNorm → lm head → softmax/loss. No MLA/FFN/residual internals —
// that's the block diagram's job. Clicking the dense or MoE box flips the
// linked <dsv3-layer>'s kind (active box wears the save-yellow highlight);
// the strip re-syncs if the layer is flipped from its own tabs.
// Standalone module so the experiment is easy to remove.

const E = 7168 * 129280;
const fmtP = (n) => n >= 1e9 ? (n / 1e9).toFixed(1) + 'B'
  : n >= 9.95e6 ? Math.round(n / 1e6) + 'M' : (n / 1e6).toFixed(1) + 'M';
const MLA = 7168 * (1536 + 512 + 64) + 1536 * 128 * 192 + 512 * 128 * 256 + 128 * 128 * 7168;
const DENSE = MLA + 3 * 7168 * 18432;
const MOE = MLA + 257 * 3 * 7168 * 2048 + 7168 * 256;

// same visual vocabulary as the block diagram (LAYER_CSS)
const CSS = `
dsv3-anatomy-plan { display: block; }
.anp { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: #0b0b0b; }
.anp svg { display: block; max-width: 100%; height: auto; }
.anp .wire { stroke: #898781; stroke-width: 1.2; fill: none; }
.anp .box { fill: #fff; stroke: #c3c2b7; }
.anp .box.on { fill: #fff8ea; stroke: #eda100; }
.anp .op { fill: #f3f2ee; stroke: #e1e0d9; }
.anp .name { font: 600 11px system-ui; fill: #0b0b0b; }
.anp .dims { font: 9px system-ui; fill: #898781; }
.anp .oplabel { font: 10.5px system-ui; fill: #52514e; }
.anp .grplabel { font: italic 10px system-ui; fill: #898781; }
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
    const L = (x1, y1, x2, y2) =>
      `<line x1="${x1 - h.left}" y1="${y1 - h.top}" x2="${x2 - h.left}" y2="${y2 - h.top}" ` +
      `stroke="#c3c2b7" stroke-width="1.2" stroke-dasharray="5 4"/>`;
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
    wire(24, 'x · 7168');
    blockBox('dense', 'dense block ×3', `${fmtP(DENSE)} each`);
    wire(24, 'x · 7168');
    blockBox('moe', 'MoE block ×58', `${fmtP(MOE)} each`);
    wire(24, 'x · 7168');
    op('final RMSNorm', '(7.2K)');
    wire(24, 'norm out · 7168');
    S.push(`<rect class="box" x="${BX}" y="${y}" width="${W}" height="34" rx="4"/>` +
      `<text class="name" x="${BX + 8}" y="${y + 14}">lm head</text>` +
      `<text class="dims" x="${BX + 8}" y="${y + 27}">7168 → 129280 (${fmtP(E)})</text>`);
    y += 34;
    wire(24, 'logits · 129280');
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
