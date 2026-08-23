// <dsv3-anatomy-plan layer="id">: EXPERIMENT — a very schematic top-level
// strip: embedding → dense block ×3 → MoE block ×58 → final RMSNorm → lm head.
// No MLA/FFN/residual internals here — that's the block diagram's job.
// Clicking the dense or MoE chip flips the linked <dsv3-layer>'s kind (and
// the strip re-syncs if the layer is flipped from its own tabs).
// Standalone module so the experiment is easy to remove.

const E = 7168 * 129280;
const fmtP = (n) => n >= 1e9 ? (n / 1e9).toFixed(1) + 'B'
  : n >= 9.95e6 ? Math.round(n / 1e6) + 'M' : (n / 1e6).toFixed(1) + 'M';
const MLA = 7168 * (1536 + 512 + 64) + 1536 * 128 * 192 + 512 * 128 * 256 + 128 * 128 * 7168;
const DENSE = MLA + 3 * 7168 * 18432;
const MOE = MLA + 257 * 3 * 7168 * 2048 + 7168 * 256;

const CSS = `
dsv3-anatomy-plan { display: block; margin: 14px 0; }
.anp { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: #0b0b0b;
  border: 1px solid #e1e0d9; border-radius: 6px; background: #fcfcfb; padding: 10px 12px;
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.anp .chip { border: 1px solid #c3c2b7; border-radius: 8px; background: #fff; padding: 5px 12px;
  color: #0b0b0b; font: 12px system-ui; text-align: left; }
.anp .chip small { display: block; color: #898781; font-size: 10.5px; }
.anp button.chip { cursor: pointer; }
.anp button.chip b { font-weight: 600; }
.anp button.chip.on { border-color: #eda100; background: #fff8ea; box-shadow: 0 1px 4px rgba(237,161,0,0.25); }
.anp .arr { color: #898781; }
.anp .note { color: #898781; font-size: 10.5px; flex-basis: 100%; margin: 2px 0 0; }
`;

export class Dsv3AnatomyPlan extends HTMLElement {
  connectedCallback() {
    const style = document.createElement('style'); style.textContent = CSS;
    const root = document.createElement('div'); root.className = 'anp';
    const chip = (html) => { const s = document.createElement('span'); s.className = 'chip'; s.innerHTML = html; return s; };
    const arr = () => { const s = document.createElement('span'); s.className = 'arr'; s.textContent = '→'; return s; };
    const btn = (kind, html) => {
      const b = document.createElement('button'); b.className = 'chip'; b.dataset.kind = kind; b.innerHTML = html;
      b.title = 'show this block kind in the diagram below';
      b.onclick = () => {
        const l = this.layerEl();
        if (!l || l.kind === kind) return;
        l.kind = kind; l.render(); l.changed(true);
        this.sync();
      };
      return b;
    };
    root.append(
      chip(`embedding<small>${fmtP(E)}</small>`), arr(),
      btn('dense', `<b>dense block</b> ×3<small>${fmtP(DENSE)} each</small>`), arr(),
      btn('moe', `<b>MoE block</b> ×58<small>${fmtP(MOE)} each</small>`), arr(),
      chip(`final RMSNorm<small>7.2K</small>`), arr(),
      chip(`lm head<small>${fmtP(E)}</small>`),
    );
    const note = document.createElement('span'); note.className = 'note';
    note.textContent = 'click a block kind to flip the diagram below · residual stream (7168) runs through all 61 blocks · MTP not shown';
    root.append(note);
    this.append(style, root);
    // re-sync the highlight when the layer flips from its own tabs
    queueMicrotask(() => {
      this.layerEl()?.addEventListener('recipe', () => this.sync());
      this.sync();
    });
  }
  layerEl() { return document.getElementById(this.getAttribute('layer') ?? ''); }
  sync() {
    const kind = this.layerEl()?.kind ?? 'moe';
    for (const b of this.querySelectorAll('button.chip'))
      b.classList.toggle('on', b.dataset.kind === kind);
  }
}
customElements.define('dsv3-anatomy-plan', Dsv3AnatomyPlan);
