// Visual audit: what the fit chart SHOWS must be re-derivable from what the
// model COMPUTED. Every rendered number carries its exact value (data-true,
// plus data-pin for badges) and a role; every plain bar and ghost carries the
// value its pixels encode. This module replays the rendering rules — rounding
// (fmtBytes), badge factors (facNum), sums, and log-axis geometry — and
// reports every disagreement. Displayed digits alone CANNOT be audited: the
// rounded components legitimately don't sum to the rounded total (1.22 + 2.44
// + 4.88 TiB + 106.4 GiB reads 8.64, the true total rounds to 8.65), so the
// audit runs on the exact values and checks the rounding separately.
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

    // 1) rounding + 2) badges: the label is fmtBytes(true) + the exact factor
    for (const t of svg.querySelectorAll('text[data-role^="val:"]')) {
      const b = +t.dataset.true, pinB = +(t.dataset.pin || 0);
      const shown = t.textContent;
      if (!shown.startsWith(fmtBytes(b)))
        bad(`${t.dataset.role} shows "${shown}" but ${b} rounds to "${fmtBytes(b)}"`);
      const badge = shown.slice(fmtBytes(b).length).trim();
      const want = !pinB || !b || Math.abs(Math.log2(b / pinB)) < 0.05 ? ''
        : b > pinB ? `▲×${facNum(b / pinB)}` : `▼×${facNum(pinB / b)}`;
      if (badge !== want) bad(`${t.dataset.role} badge "${badge}" ≠ recomputed "${want}"`);
    }

    // 3) sums, on the EXACT values: total = Σ components (gutter names carry
    // every row's value, visible or dimmed), component = Σ its open parts
    const nameOf = (r) => svg.querySelector(`text[data-role="name:${r}"]`);
    if (nameOf('total')) {
      const comps = [0, 1, 2, 3].map((i) => +(nameOf(i)?.dataset.true ?? 0));
      const total = +nameOf('total').dataset.true;
      const sum = comps.reduce((a, b) => a + b, 0);
      if (Math.abs(sum - total) > total * 1e-9)
        bad(`total ${total} ≠ Σ components ${sum}`);
      for (let i = 0; i < 4; i++) {
        const parts = [...svg.querySelectorAll(`text[data-role^="val:part:${i}:"]`)];
        if (!parts.length) continue;
        const psum = parts.reduce((a, t) => a + +t.dataset.true, 0);
        // parts render only when > 0; a skipped part contributes exactly 0
        if (Math.abs(psum - comps[i]) > comps[i] * 1e-9)
          bad(`comp ${i} = ${comps[i]} ≠ Σ open parts ${psum}`);
      }
    }

    // 4) geometry: a bar's pixels must encode the same number as its label
    // (0.15 = the .toFixed(1) print grid; 0.5 = the min sliver for tiny values)
    for (const r of svg.querySelectorAll('rect[data-bar], rect[data-ghost]')) {
      const b = +r.dataset.true;
      const wantW = Math.max(0.5, px(b) - BAR_GEO.x0);
      if (Math.abs(+r.getAttribute('width') - wantW) > 0.15)
        bad(`${r.dataset.bar ?? 'ghost:' + r.dataset.ghost} bar width ${r.getAttribute('width')} ≠ log₂(${b}) → ${wantW.toFixed(1)}`);
    }
  }
  return { charts, findings: out };
}
