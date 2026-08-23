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
    this.draw();
    // re-sync the highlight when the layer flips from its own tabs
    queueMicrotask(() => {
      this.layerEl()?.addEventListener('recipe', () => this.draw());
      this.draw();
    });
  }
  layerEl() { return document.getElementById(this.getAttribute('layer') ?? ''); }
  draw() {
    const kind = this.layerEl()?.kind ?? 'moe';
    const S = [];
    S.push(`<defs><marker id="planarr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">` +
      `<path d="M 0 0 L 8 4 L 0 8 z" fill="#898781"/></marker></defs>`);
    const SX = 14, BX = 26, W = 152;
    const wire = (y1, y2) => S.push(`<line class="wire" x1="${SX}" y1="${y1}" x2="${SX}" y2="${y2}" marker-end="url(#planarr)"/>`);
    let y = 14;
    S.push(`<text class="oplabel" x="${BX}" y="${y - 2}">tokens</text>`);
    wire(y, y + 12); y += 12;
    const op = (label, dims, h = 22) => {
      S.push(`<rect class="op" x="${BX}" y="${y}" width="${W}" height="${h}" rx="${h / 2 > 11 ? 6 : h / 2}"/>` +
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
    wire(y, y + 14); y += 14;
    blockBox('dense', 'dense block ×3', `${fmtP(DENSE)} each`);
    wire(y, y + 14); y += 14;
    blockBox('moe', 'MoE block ×58', `${fmtP(MOE)} each`);
    wire(y, y + 14); y += 14;
    op('final RMSNorm', '(7.2K)');
    wire(y, y + 12); y += 12;
    S.push(`<rect class="box" x="${BX}" y="${y}" width="${W}" height="34" rx="4"/>` +
      `<text class="name" x="${BX + 8}" y="${y + 14}">lm head</text>` +
      `<text class="dims" x="${BX + 8}" y="${y + 27}">7168 → 129280 (${fmtP(E)})</text>`);
    y += 34;
    wire(y, y + 12); y += 12;
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
  }
}
customElements.define('dsv3-anatomy-plan', Dsv3AnatomyPlan);
