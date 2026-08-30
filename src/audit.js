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

// Besides pass/fail (findings), the audit narrates WHAT it verified: each
// chart's report lists every checked implication with its exact arithmetic
// (the overlay mode renders these; sel points at the pattern's element).
export function auditFitCharts(root = document) {
  const out = [];
  const reports = [];
  const px = (b) => BAR_GEO.x0
    + Math.max(0, Math.min(1, (Math.log2(Math.max(b, 1)) - BAR_GEO.lo) / (BAR_GEO.hi - BAR_GEO.lo))) * BAR_GEO.bw;
  const B = (n) => `${Math.round(n).toLocaleString('en-US')} B`;
  let charts = 0;
  const hosts = root.matches?.('dsv3-layer') ? [root] : [...root.querySelectorAll('dsv3-layer')];
  for (const host of hosts) {
    const svg = host.querySelector('.lv-bar svg');
    if (!svg) continue;
    charts++;
    const id = host.id || host.getAttribute('solo') || host.getAttribute('comps')
      || (host.hasAttribute('parts') ? 'parts' : `chart${charts}`);
    const report = { host, checks: [], findings: [] };
    reports.push(report);
    const bad = (msg) => { out.push(`${id}: ${msg}`); report.findings.push(msg); };
    const ok = (sel, msg) => report.checks.push({ sel, msg });

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
      else ok(`[data-role="${t.dataset.role}"]`, `“${fmtBytes(b)}” is exactly ${B(b)}, rounded`);
      // 3a) the ▲/▼ badge is the exact ratio vs the saved value
      const badge = t.textContent.slice(fmtBytes(b).length).trim();
      const want = !pinB || !b || Math.abs(Math.log2(b / pinB)) < 0.05 ? ''
        : b > pinB ? `▲×${facNum(b / pinB)}` : `▼×${facNum(pinB / b)}`;
      if (badge !== want) bad(`${t.dataset.role} badge "${badge}" ≠ recomputed "${want}"`);
      else if (want) ok(`[data-role="${t.dataset.role}"]`, `badge ${want} is the exact live-vs-saved ratio (${B(b)} vs ${B(pinB)})`);
      // pair this value with the bars drawn on ITS row — pure geometry
      const row = bars.filter((r) => onRow(t, r));
      const solid = row.filter((r) => !r.getAttribute('stroke-dasharray'));
      const dashed = row.filter((r) => r.getAttribute('stroke-dasharray'));
      // 2) the rightmost solid edge sits at px(value) — plain or stacked
      if (solid.length && b > 0) {
        const edge = Math.max(...solid.map((r) => +r.getAttribute('x') + +r.getAttribute('width')));
        if (Math.abs(edge - px(b)) > 1.6)   // stacked segments carry ±1px seams
          bad(`${t.dataset.role} bar edge ${edge.toFixed(1)} ≠ px(${b}) = ${px(b).toFixed(1)}`);
        else ok(`[data-role="${t.dataset.role}"]`, `rightmost solid edge sits at px(${B(b)}) on the log₂ axis`);
      }
      // 3b) a dashed twin means a saved baseline drawn at px(saved)
      for (const g of dashed) {
        const edge = +g.getAttribute('x') + +g.getAttribute('width');
        if (!pinB) bad(`${t.dataset.role} has a ghost but no saved value to imply`);
        else if (Math.abs(edge - px(pinB)) > 0.6)
          bad(`${t.dataset.role} ghost edge ${edge.toFixed(1)} ≠ px(saved ${pinB}) = ${px(pinB).toFixed(1)}`);
        else ok(`[data-role="${t.dataset.role}"]`, `dashed ghost ends at px(saved ${B(pinB)}) — the twin IS the baseline`);
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
      else ok(`[data-role="${nm.dataset.role}"]`,
        `the ${kids.length} “·” rows sum EXACTLY to ${nm.textContent} (Σ = ${B(parent)}; rounded digits wouldn't add up)`);
    }

    // 5) the distances ruler: a tick claiming ×N sits at exactly log₂(N)
    // doublings along the axis (the map legend must be to scale)
    for (const tk of svg.querySelectorAll('line[data-fac]')) {
      const f = +tk.dataset.fac;
      const want = BAR_GEO.x0 + Math.log2(f) / (BAR_GEO.hi - BAR_GEO.lo) * BAR_GEO.bw;
      if (Math.abs(+tk.getAttribute('x1') - want) > 0.15)
        bad(`distances tick ×${f} at ${tk.getAttribute('x1')} ≠ ${want.toFixed(1)}`);
      else if (f > 1) ok(`line[data-fac="${f}"]`, `ruler tick ×${f} sits exactly ${Math.log2(f)} doublings from the origin`);
    }
  }
  return { charts, findings: out, reports };
}

// ---- the page-wide audit overlay (hidden debug mode, Alt+A) ----------------
// The visual language, verified IN PLACE with the same code the battery
// runs: every fit chart grows a floating chip — ✓ N implications verified
// (✗ N in alarm red if any lie survives) — and clicking it drops the
// chart's report as an overlay: one line per checked implication with its
// exact arithmetic; hovering a line lights the pattern it re-derived and
// dims the rest. Nothing reflows: chips and panels are absolutely
// positioned inside .lv-bar. Charts re-render wholesale (innerHTML) on
// every knob turn / slide / tween frame, so a MutationObserver re-audits
// after the page has been QUIET for a beat — mid-blend geometry is
// deliberately between two truths, and auditing it would cry wolf.
const AUD_CSS = `
.aud-chip { position: absolute; top: -6px; right: 0; z-index: 30; cursor: pointer;
  font: 600 10px ui-monospace, monospace; color: #1a7a43; background: #f0faf4;
  border: 1px solid #1baf7a; border-radius: 9px; padding: 1px 7px; opacity: 0.92; }
.aud-chip.bad { color: #d03b3b; background: #fdf1f1; border-color: #d03b3b; }
.aud-panel { display: none; position: absolute; top: 14px; right: 0; z-index: 31;
  max-width: 620px; max-height: 260px; overflow: auto; overscroll-behavior: contain;
  font: 11px ui-monospace, monospace; color: #52514e; text-align: left;
  background: #fff; border: 1px solid #c3c2b7; border-radius: 6px;
  box-shadow: 0 2px 10px rgba(11, 11, 11, 0.12); padding: 6px 10px; }
.aud-panel.open { display: block; }
.aud-panel .aud-hd { color: #898781; font-style: italic; margin: 2px 0; }
.aud-panel .aud-ln { line-height: 1.55; white-space: nowrap; }
.aud-panel .aud-ln:hover { color: #0b0b0b; }
.aud-panel .aud-ln.bad { color: #d03b3b; }`;
let audObs = null;
const audOpen = new Set();   // chart ids whose report is pinned open — the
                             // charts re-render wholesale, so DOM can't hold this
export function auditOverlay(on) {
  const sweep = () => document.querySelectorAll('.aud-chip, .aud-panel').forEach((n) => n.remove());
  audObs?.disconnect(); audObs = null;
  sweep(); audOpen.clear();
  if (!on) return;
  if (!document.getElementById('aud-style')) {
    const st = document.createElement('style');
    st.id = 'aud-style'; st.textContent = AUD_CSS;
    document.head.append(st);
  }
  const decorate = () => {
    sweep();
    for (const rep of auditFitCharts(document).reports) {
      const slot = rep.host.querySelector('.lv-bar');
      if (!slot) continue;
      const chip = document.createElement('div');
      chip.className = 'aud-chip' + (rep.findings.length ? ' bad' : '');
      chip.textContent = rep.findings.length ? `✗ ${rep.findings.length}` : `✓ ${rep.checks.length}`;
      chip.title = 'visual-language audit (Alt+A toggles) — click for this chart\u2019s verification report';
      const panel = document.createElement('div');
      panel.className = 'aud-panel' + (rep.host.id && audOpen.has(rep.host.id) ? ' open' : '');
      const hd = document.createElement('div');
      hd.className = 'aud-hd';
      hd.textContent = rep.findings.length
        ? `the audit found ${rep.findings.length} lie(s) on this chart:`
        : `${rep.checks.length} implications verified — every number and pattern re-derived from its exact value:`;
      panel.append(hd);
      for (const f of rep.findings) {
        const d = document.createElement('div');
        d.className = 'aud-ln bad'; d.textContent = `✗ ${f}`; panel.append(d);
      }
      for (const c of rep.checks) {
        const d = document.createElement('div');
        d.className = 'aud-ln'; d.textContent = `✓ ${c.msg}`; d.dataset.sel = c.sel; panel.append(d);
      }
      // hover a line → light exactly the pattern it verified (plus the
      // row's bar/ghost twins when the line anchors on a value text)
      const marks = () => [...slot.querySelectorAll('svg [data-role], svg [data-bar], svg [data-ghost], svg [data-fac]')];
      panel.onmouseover = (ev) => {
        const sel = ev.target.closest?.('.aud-ln')?.dataset.sel;
        const svg = slot.querySelector('svg');
        if (!sel || !svg) return;
        const m = /^\[data-role="val:(.+)"\]$/.exec(sel);
        const full = m ? `${sel}, [data-bar="${m[1]}"], [data-ghost="${m[1]}"]` : sel;
        const hit = new Set(svg.querySelectorAll(full));
        for (const n of marks()) n.style.opacity = hit.has(n) ? '' : 0.2;
      };
      panel.onmouseout = () => { for (const n of marks()) n.style.opacity = ''; };
      chip.onclick = () => {
        panel.classList.toggle('open');
        if (rep.host.id) audOpen[panel.classList.contains('open') ? 'add' : 'delete'](rep.host.id);
      };
      slot.append(chip, panel);
    }
  };
  decorate();
  let t = null;
  const ours = (n) => n.nodeType === 1 && (n.classList?.contains('aud-chip') || n.classList?.contains('aud-panel'));
  audObs = new MutationObserver((muts) => {
    // ignore our own decoration (else decorate() would trigger itself forever)
    if (muts.every((m2) => [...m2.addedNodes, ...m2.removedNodes].every(ours))) return;
    clearTimeout(t); t = setTimeout(decorate, 250);
  });
  audObs.observe(document.body, { childList: true, subtree: true });
}
