// Diagram lint driver: serves the repo, renders scripts/diagramlint.html in
// headless Chrome, and reports visual-grammar violations (see
// docs/diagram-grammar.md): arrowheads in text, wires through labels, label
// collisions, clipped/border-cut text, overlapping op boxes.
// Known-acceptable findings live in scripts/diagramlint-allow.json (substring
// match against the JSON of a finding). Exit 1 on any unallowed finding.
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { launch, serve, root } from './cdp.mjs';

const [srv, browser] = await Promise.all([serve(), launch()]);
const p = await browser.open(srv.url('scripts/diagramlint.html'), { width: 1500, height: 4000, budget: 12000 });
await p.until();
const out = await p.eval(`document.getElementById('lint-out')?.textContent ?? null`);
await p.close(); browser.close(); srv.close();

if (out == null) { console.error('lint page produced no output (widgets failed to render?)'); process.exit(2); }
const findings = JSON.parse(out || '[]');

let allow = [];
try { allow = JSON.parse(await readFile(join(root, 'scripts/diagramlint-allow.json'), 'utf8')); } catch {}
const allowed = (f) => { const s = JSON.stringify(f); return allow.some(a => s.includes(a)); };

const bad = findings.filter(f => !allowed(f));
const ok = findings.length - bad.length;
const byCheck = {};
for (const f of bad) (byCheck[f.check] ??= []).push(f);
for (const [check, fs] of Object.entries(byCheck)) {
  console.log(`\n${check} (${fs.length}):`);
  for (const f of fs.slice(0, 25)) console.log('  ' + JSON.stringify(f));
  if (fs.length > 25) console.log(`  … +${fs.length - 25} more`);
}
console.log(`\ndiagramlint: ${bad.length} finding(s)${ok ? ` (+${ok} allowlisted)` : ''} across the widget matrix`);
process.exit(bad.length ? 1 : 0);
