// <dsv3-mmfig>: one token through a 5 × 5 weight matrix (the 6ND prep
// diagram in studies/03-roofline.html), then the expansion of y = xW, one
// term per weight cell. The element's text content becomes the caption.
// Hover dependencies across both halves (a cell and its term are twins:
// both light as the hovered one): a weight (i, j) reads x_i and adds into
// y_j, beside the rest of y_j's sum; x_i feeds row i and reaches every y;
// y_j sums column j and reads every x.
const K = 5, P = 56, S = 54;                        // cells, pitch, cell size
const WX = 322, WY = 16, XX = 20, VY = 316;         // W's and the x row's top-left corners
const SUB = (n) => '₀₁₂₃₄₅₆₇₈₉'[n];

const CSS = `
dsv3-mmfig { display: block; }
.mm { display: block; margin: 14px 0 22px; max-width: 100%; height: auto; font: 11px system-ui, -apple-system, "Segoe UI", sans-serif; }
.mm text { fill: var(--c-52514e); }
.mm g text { text-anchor: middle; font-size: 14px; fill: var(--c-1c1c1a); }
.mm .w rect { fill: var(--c-bcd8f3); }
.mm .w text { font-size: 12.5px; fill: var(--c-0b3d75); }
.mm .v rect { fill: var(--c-f3f2ee); stroke: var(--c-c3c2b7); }
dsv3-mmfig .w, dsv3-mmfig .v { cursor: default; }
.mm .ar path { stroke: var(--c-898781); marker-end: url(#mm-ah); }
#mm-ah path { fill: var(--c-898781); }
#mm-ah-on path { fill: var(--c-eda100); }
/* hitboxes tile the gaps (no dead zone flicker between neighbors): cells
   by an invisible rect the full pitch, chips by an ::after reaching
   halfway across the +/= and the grid gaps */
.mm g rect.hit { fill: transparent; stroke: none; }
.mmeq .w, .mmeq .v { position: relative; }
.mmeq .w::after, .mmeq .v::after { content: ''; position: absolute; inset: -4px -11px; }
/* the expansion of y = xW: one term per weight cell, same colors */
.mmeq { width: 620px; box-sizing: border-box; display: grid; grid-template-columns: repeat(${2 * K + 1}, auto); justify-content: center; align-items: center; gap: 5px 6px;
  margin: -6px 0 0; padding: 10px 12px; border: 1px solid var(--c-e1e0d9); border-radius: 6px; background: var(--c-fcfcfb);
  font: 14px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-1c1c1a); text-align: center; }
.mmeq .w, .mmeq .v { padding: 3px 6px; border-radius: 3px; }
.mmeq .w { background: var(--c-bcd8f3); color: var(--c-0b3d75); font-size: 12.5px; }
.mmeq .v { background: var(--c-f3f2ee); box-shadow: inset 0 0 0 1px var(--c-c3c2b7); }
.mmeq .op { color: var(--c-898781); }
.mmcap { width: 620px; max-width: 100%; margin: 10px 0 22px; text-align: center; font: 11px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-52514e); }
/* hover: the hovered element (and its twin) solid amber, what it reads /
   feeds outlined, the rest of its sum pale */
.mm .dep rect { fill: var(--c-fdeab5); stroke: var(--c-eda100); stroke-width: 1.5; }
.mm .src rect { fill: var(--c-f6cd74); stroke: var(--c-eda100); stroke-width: 1.5; }
.mm .sib rect { fill: var(--c-fff3d1); }
.mm .ar .dep { stroke: var(--c-eda100); stroke-width: 1.5; marker-end: url(#mm-ah-on); }
.mmeq .dep { background: var(--c-fdeab5); box-shadow: inset 0 0 0 1.5px var(--c-eda100); }
.mmeq .src { background: var(--c-f6cd74); box-shadow: inset 0 0 0 1.5px var(--c-eda100); }
.mmeq .sib { background: var(--c-fff3d1); }
`;

const ARIA = 'One token through a 5 by 5 weight matrix W, in the standard layout: the token x on the left, W on top, the output y at the bottom right under W. Every weight cell holds exactly one term, plus w times x: one multiply and one add. Each column of W sums into the output below it. So a forward pass costs 2 FLOPs per weight per token. Hovering a weight highlights the input it reads and the output it adds into.';
const marker = (id, sz) => `<marker id="${id}" viewBox="0 0 8 8" refX="8" refY="4" markerWidth="${sz}" markerHeight="${sz}" orient="auto"><path d="M0 0 L8 4 L0 8 Z"/></marker>`;
// one cell: the hitbox the full pitch, the visible rect inset 1px
const cell = (cls, data, x, y, label, ty) => `<g class="${cls}" ${data}><rect class="hit" x="${x - 1}" y="${y - 1}" width="${P}" height="${P}"/>` +
  `<rect x="${x}" y="${y}" width="${S}" height="${S}"/><text x="${x + S / 2}" y="${y + ty}">${label}</text></g>`;
const term = (i, j) => `w${SUB(i + 1)}${SUB(j + 1)}x${SUB(i + 1)}`;
const idx = [...Array(K).keys()];

function svg() {
  const H = VY + S + 26, colX = (j) => WX + j * P + S / 2;
  return `<svg class="mm" width="620" height="${H}" viewBox="0 0 620 ${H}" role="img" aria-label="${ARIA}">` +
    `<defs>${marker('mm-ah', 7)}${marker('mm-ah-on', 6)}</defs>` +
    idx.map((i) => idx.map((j) => cell('w', `data-i="${i}" data-j="${j}"`, WX + j * P, WY + i * P, '+' + term(i, j), 31)).join('')).join('') +
    idx.map((i) => cell('v', `data-i="${i}"`, XX + i * P, VY, `x${SUB(i + 1)}`, 32)).join('') +
    idx.map((j) => cell('v', `data-j="${j}"`, WX + j * P, VY, `y${SUB(j + 1)}`, 32)).join('') +
    `<g class="ar">${idx.map((j) => `<path data-j="${j}" d="M${colX(j)} ${WY + K * P - 2} V${VY}"/>`).join('')}</g>` +
    `<text x="${WX - 22}" y="${WY + K * P / 2}" text-anchor="end">weights W</text>` +
    `<text x="${XX + K * P / 2 - 1}" y="${H - 8}" text-anchor="middle">input x</text>` +
    `<text x="${colX((K - 1) / 2)}" y="${H - 8}" text-anchor="middle">output y = xW</text></svg>`;
}
const eq = () => `<div class="mmeq">${idx.map((j) => `<span class="v" data-j="${j}">y${SUB(j + 1)}</span><span class="op">=</span>` +
  idx.map((i) => `<span class="w" data-i="${i}" data-j="${j}">${term(i, j)}</span>`).join('<span class="op">+</span>')).join('')}</div>`;

class Dsv3Mmfig extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    const cap = this.textContent.trim();
    this.innerHTML = `<style>${CSS}</style>${svg()}${eq()}` + (cap ? `<p class="mmcap">${cap}</p>` : '');
    const els = [...this.querySelectorAll('[data-i], [data-j]')];
    const kind = (e) => (e.classList.contains('w') ? 'w' : e.dataset.j === undefined ? 'x' : e.tagName === 'path' ? 'a' : 'y');
    const clear = () => els.forEach((e) => e.classList.remove('src', 'dep', 'sib'));
    this.addEventListener('mouseover', (ev) => {
      clear();
      const t = ev.target.closest('.w, .v');
      if (!t) return;
      const k = kind(t), { i, j } = t.dataset;
      for (const e of els) {
        const ke = kind(e), { i: ei, j: ej } = e.dataset;
        const c = ke === k && ei === i && ej === j ? 'src'
          : k === 'w' ? (ke === 'w' ? ej === j && 'sib' : ke === 'x' ? ei === i && 'dep' : ej === j && 'dep')
          : k === 'x' ? (ke === 'w' ? ei === i && 'dep' : ke === 'y' && 'sib')
          : ke === 'w' || ke === 'a' ? ej === j && 'dep' : ke === 'x' && 'sib';
        if (c) e.classList.add(c);
      }
    });
    this.addEventListener('mouseleave', clear);
  }
}
if (typeof customElements !== 'undefined' && !customElements.get('dsv3-mmfig')) customElements.define('dsv3-mmfig', Dsv3Mmfig);
