// <dsv3-anatomy-plan>: EXPERIMENT — the high-level plan of the model, vertical:
// embedding -> dense block x3 -> MoE block x58 -> final norm -> lm head, with
// each block showing its two halves (MLA + FFN) as color-coded slots. The
// slots' colors match the component diagrams drawn below the plan (the real
// <dsv3-layer only="mla|ffn"> renderings), so the plan is the table of
// contents and the components are drawn exactly once each.
// Standalone module so the experiment is easy to remove.

const P = {
  embed: 7168 * 129280,
  mla: 7168 * (1536 + 512 + 64) + 1536 * 128 * 192 + 512 * 128 * 256 + 128 * 128 * 7168,
  dense: 3 * 7168 * 18432,
  moeFfn: (3 * 7168 * 2048) * 257 + 7168 * 256,
};
const fmtP = (n) => n >= 1e9 ? (n / 1e9).toFixed(1) + 'B'
  : n >= 9.95e6 ? Math.round(n / 1e6) + 'M' : (n / 1e6).toFixed(1) + 'M';

// component accent colors — shared with the page's component headings
export const ACC = { mla: '#a06a00', dense: '#2a78d6', moe: '#6b5bd2' };

const CSS = `
dsv3-anatomy-plan { display: block; margin: 14px 0; }
.anp { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: #0b0b0b;
  border: 1px solid #e1e0d9; border-radius: 6px; background: #fcfcfb; padding: 10px 12px;
  display: inline-block; }
.anp svg { display: block; max-width: 100%; height: auto; }
.anp .op { fill: #f3f2ee; stroke: #e1e0d9; }
.anp .card { fill: #fcfcfb; stroke: #d8d6cb; }
.anp .slot { fill: #fff; }
.anp .oplabel { font: 10px system-ui; fill: #52514e; }
.anp .dims { font: 9px system-ui; fill: #898781; }
.anp .title { font: 600 11px system-ui; }
.anp .grplabel { font: italic 10px system-ui; fill: #898781; }
.anp .wire { stroke: #898781; stroke-width: 1.2; fill: none; }
`;

export class Dsv3AnatomyPlan extends HTMLElement {
  connectedCallback() {
    const S = [];
    S.push(`<defs><marker id="anparr" markerWidth="7" markerHeight="7" refX="5.5" refY="3" orient="auto">
      <path d="M0,0.5 L5.5,3 L0,5.5" fill="none" stroke="#898781" stroke-width="1.1"/></marker></defs>`);
    const LX = 8, LW = 262;
    const wire = (y1, y2) => S.push(`<line class="wire" x1="${LX + LW / 2}" y1="${y1}" x2="${LX + LW / 2}" y2="${y2}" marker-end="url(#anparr)"/>`);
    const pill = (y, label, dims) => {
      S.push(`<rect class="op" x="${LX}" y="${y}" width="${LW}" height="24" rx="12"/>`
        + `<text class="oplabel" x="${LX + 14}" y="${y + 15.5}">${label}${dims ? ` <tspan class="dims">${dims}</tspan>` : ''}</text>`);
      return y + 24;
    };
    // a block: title + its two halves stacked vertically, color-coded to the
    // component diagrams below the plan
    const block = (y, title, count, ffnKey, ffnLabel, total) => {
      const h = 100;
      S.push(`<rect class="card" x="${LX}" y="${y}" width="${LW}" height="${h}" rx="8"/>`
        + `<text class="title" x="${LX + 12}" y="${y + 17}">${title} <tspan class="dims">×${count} · ${total}</tspan></text>`);
      const slot = (sy, key, label, p) => S.push(
        `<rect class="slot" x="${LX + 12}" y="${sy}" width="${LW - 24}" height="26" rx="5" stroke="${ACC[key]}"/>`
        + `<text class="oplabel" x="${LX + 21}" y="${sy + 17}" fill="${ACC[key]}">${label}</text>`
        + `<text class="dims" x="${LX + LW - 21}" y="${sy + 17}" text-anchor="end">${p}</text>`);
      slot(y + 26, 'mla', 'MLA (attention)', fmtP(P.mla));
      wire(y + 52, y + 62);
      slot(y + 62, ffnKey, ffnLabel, ffnKey === 'dense' ? fmtP(P.dense) : fmtP(P.moeFfn));
      return y + h;
    };
    let y = 10;
    y = pill(y, 'embedding', `7168 × 129280 (${fmtP(P.embed)})`);
    wire(y, y + 14); y += 14;
    y = block(y, 'dense block', 3, 'dense', 'dense FFN', fmtP(3 * (P.mla + P.dense)));
    wire(y, y + 14); y += 14;
    y = block(y, 'MoE block', 58, 'moe', 'MoE FFN', fmtP(58 * (P.mla + P.moeFfn)));
    wire(y, y + 14); y += 14;
    y = pill(y, 'final RMSNorm', null);
    wire(y, y + 12); y += 12;
    y = pill(y, 'lm head', `7168 → 129280 (${fmtP(P.embed)})`);
    S.push(`<text class="grplabel" x="${LX}" y="${y + 18}">residual stream (7168) runs top to bottom;</text>`
      + `<text class="grplabel" x="${LX}" y="${y + 31}">MTP not shown</text>`);
    const H = y + 40;
    const style = document.createElement('style'); style.textContent = CSS;
    const root = document.createElement('div'); root.className = 'anp';
    root.innerHTML = `<svg viewBox="0 0 ${LX + LW + 8} ${H}" width="${LX + LW + 8}" height="${H}">${S.join('')}</svg>`;
    this.append(style, root);
  }
}
customElements.define('dsv3-anatomy-plan', Dsv3AnatomyPlan);
