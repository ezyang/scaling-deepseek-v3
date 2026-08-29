// Visual audit: the audit keys on the VISUAL LANGUAGE, never on the model.
// Whenever the chart draws a pattern, the pattern IMPLIES arithmetic, and the
// implication is checked against the exact values (data-true / data-pin)
// linked behind the rendered numbers. The vocabulary (see the semantic-
// implications section of docs/diagram-grammar.md):
//
//   1. a rendered value IS its exact value, rounded: text == fmtBytes(data-true)
//   2. a bar with a value at its end: the rightmost solid edge on the row
//      sits at px(value) — one rule covers plain AND stacked bars (a stack's
//      top segment must end at the total it claims)
//   3. a dashed twin (ghost) implies a saved baseline: its edge sits at
//      px(data-pin), and the row's ▲/▼×N badge equals the exact ratio
//   4. an indented '· name' row beneath a row is a DECOMPOSITION: the
//      children sum EXACTLY to their parent (rendered digits can't be
//      summed — correctly-rounded parts don't add up to the rounded parent)
//   5. a distances-ruler tick labeled ×N sits exactly log₂(N) doublings
//      from the origin cap — the legend's spans really are those factors
//
// Deliberately absent: any re-derivation of the model. Model identities (the
// stage split partitions the checkpoint-exact total; params vs the released
// checkpoint) live in scripts/sanity.mjs. The audit proves the RENDERING
// tells one coherent story; sanity proves the model tells the true one;
// data-true is the bridge between them.
import { fmtBytes, facNum, BAR_GEO } from './viewer.js';

export function auditFitCharts(root = document) {
  const out = [];
  const px = (b) => BAR_GEO.x0
    + Math.max(0, Math.min(1, (Math.log2(Math.max(b, 1)) - BAR_GEO.lo) / (BAR_GEO.hi - BAR_GEO.lo))) * BAR_GEO.bw;
  let charts = 0;
  for (const host of root.querySelectorAll('dsv3-layer')) {
    const svg = host.querySelector('.lv-bar svg');
    if (!svg) continue;
    charts++;
    const id = host.id || host.getAttribute('solo') || host.getAttribute('comps')
      || (host.hasAttribute('parts') ? 'parts' : `chart${charts}`);
    const bad = (msg) => out.push(`${id}: ${msg}`);

    const vals = [...svg.querySelectorAll('text[data-role^="val:"]')];
    const bars = [...svg.querySelectorAll('rect')].filter((r) => {
      const h = +r.getAttribute('height');
      return (h === 8 || h === 5) && +r.getAttribute('width') > 0;   // bars, not hitboxes/shading
    });
    const onRow = (t, r) => Math.abs((+r.getAttribute('y') + +r.getAttribute('height') / 2)
      - (+t.getAttribute('y') - 2.5)) < 5;

    for (const t of vals) {
      const b = +t.dataset.true, pinB = +(t.dataset.pin || 0);
      // 1) the rendered number is its exact value, rounded
      if (!t.textContent.startsWith(fmtBytes(b)))
        bad(`${t.dataset.role} shows "${t.textContent}" but ${b} rounds to "${fmtBytes(b)}"`);
      // 3a) the ▲/▼ badge is the exact ratio vs the saved value
      const badge = t.textContent.slice(fmtBytes(b).length).trim();
      const want = !pinB || !b || Math.abs(Math.log2(b / pinB)) < 0.05 ? ''
        : b > pinB ? `▲×${facNum(b / pinB)}` : `▼×${facNum(pinB / b)}`;
      if (badge !== want) bad(`${t.dataset.role} badge "${badge}" ≠ recomputed "${want}"`);
      // pair this value with the bars drawn on ITS row — pure geometry
      const row = bars.filter((r) => onRow(t, r));
      const solid = row.filter((r) => !r.getAttribute('stroke-dasharray'));
      const dashed = row.filter((r) => r.getAttribute('stroke-dasharray'));
      // 2) the rightmost solid edge sits at px(value) — plain or stacked
      if (solid.length && b > 0) {
        const edge = Math.max(...solid.map((r) => +r.getAttribute('x') + +r.getAttribute('width')));
        if (Math.abs(edge - px(b)) > 1.6)   // stacked segments carry ±1px seams
          bad(`${t.dataset.role} bar edge ${edge.toFixed(1)} ≠ px(${b}) = ${px(b).toFixed(1)}`);
      }
      // 3b) a dashed twin means a saved baseline drawn at px(saved)
      for (const g of dashed) {
        const edge = +g.getAttribute('x') + +g.getAttribute('width');
        if (!pinB) bad(`${t.dataset.role} has a ghost but no saved value to imply`);
        else if (Math.abs(edge - px(pinB)) > 0.6)
          bad(`${t.dataset.role} ghost edge ${edge.toFixed(1)} ≠ px(saved ${pinB}) = ${px(pinB).toFixed(1)}`);
      }
    }

    // 4) decomposition: the '· name' rows below a row sum exactly to it —
    // the indent IS the claim; which rows are children comes from layout
    const names = [...svg.querySelectorAll('text[data-role^="name:"]')]
      .sort((a2, b2) => +a2.getAttribute('y') - +b2.getAttribute('y'));
    const parts = vals.filter((t) => t.dataset.role.startsWith('val:part:'));
    for (const [ni, nm] of names.entries()) {
      const y = +nm.getAttribute('y');
      const nextY = +(names[ni + 1]?.getAttribute('y') ?? Infinity);
      const kids = parts.filter((t) => +t.getAttribute('y') > y && +t.getAttribute('y') < nextY);
      if (!kids.length) continue;
      const parent = +nm.dataset.true;
      const sum = kids.reduce((a2, t) => a2 + +t.dataset.true, 0);
      if (Math.abs(sum - parent) > parent * 1e-9)
        bad(`decomposition under "${nm.textContent}": Σ children ${sum} ≠ ${parent}`);
    }

    // 5) the distances ruler: a tick claiming ×N sits at exactly log₂(N)
    // doublings along the axis (the map legend must be to scale)
    for (const tk of svg.querySelectorAll('line[data-fac]')) {
      const f = +tk.dataset.fac;
      const want = BAR_GEO.x0 + Math.log2(f) / (BAR_GEO.hi - BAR_GEO.lo) * BAR_GEO.bw;
      if (Math.abs(+tk.getAttribute('x1') - want) > 0.15)
        bad(`distances tick ×${f} at ${tk.getAttribute('x1')} ≠ ${want.toFixed(1)}`);
    }
  }
  return { charts, findings: out };
}
