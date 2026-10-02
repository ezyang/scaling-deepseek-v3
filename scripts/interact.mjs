// Interaction-test driver: run a scenario script against a page in headless
// Chrome and report assertions. Complements diagramlint (static geometry)
// with sequenced clicks/hovers/state probes.
//
//   node scripts/interact.mjs <page-path> <scenario-file> [--width N] [--shot out.png]
//
// The scenario file is plain JS, injected as a module after the page's own
// scripts, with a tiny harness `T` in scope:
//   T.click(sel)          dispatch a click on querySelector(sel)
//   T.hover(sel) / T.unhover(sel)   mouseenter / mouseleave
//   T.text(sel)           textContent (trimmed) or null
//   T.el(sel)             the element
//   T.check(name, cond, detail?)    record an assertion
//   T.log(name, value)    record a value for the report
//   await T.tick(ms?)     let the page settle (default 120 ms)
//   T.done()              finish (writes the report; REQUIRED at the end)
// Scenarios run inside an async IIFE, so top-level await works.
import { readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { launch, serve, root } from './cdp.mjs';

const HARNESS = `
const T = {
  el: (sel) => document.querySelector(sel),
  text: (sel) => document.querySelector(sel)?.textContent.trim() ?? null,
  click: (sel) => document.querySelector(sel)?.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true })),   // real clicks are cancelable (preventDefault must work)
  hover: (sel) => document.querySelector(sel)?.dispatchEvent(new MouseEvent('mouseenter')),
  unhover: (sel) => document.querySelector(sel)?.dispatchEvent(new MouseEvent('mouseleave')),
  tick: (ms = 120) => new Promise(r => setTimeout(r, ms)),
  _out: [],
  check: (name, cond, detail = '') => T._out.push({ check: name, ok: !!cond, detail: String(detail) }),
  log: (name, value) => T._out.push({ log: name, value: String(value) }),
  done: () => { document.title = 'INTERACT-DONE'; __interactDone(JSON.stringify(T._out)); },
};
`;

// Run one scenario in a fresh browser context of `browser`; resolves with
// the report text and its tallies. The battery calls this in-process.
export async function runScenario(browser, srv, page, scenarioFile, { width = 1500, shot = null } = {}) {
  const scenario = await readFile(resolve(root, scenarioFile), 'utf8');
  const pageHtml = await readFile(join(root, page), 'utf8');
  const job = srv.inject(page, pageHtml.replace('</body>',
    `<script type="module">${HARNESS}\n(async () => {\n await T.tick(500);\n${scenario}\n})().catch(e => { T._out.push({ error: String(e) }); T.done(); });</script></body>`));
  const p = await browser.open(job.url, { width, height: 4000, bindings: ['__interactDone'] });
  try {
    const payload = await p.until('__interactDone');
    if (shot) await writeFile(shot, await p.screenshot());
    if (payload == null) return { text: 'scenario produced no output (did it call T.done()?)', checks: 0, fails: 1 };
    const out = JSON.parse(payload), lines = [];
    let fails = 0;
    for (const o of out) {
      if (o.error) { lines.push('ERROR ' + o.error); fails++; }
      else if (o.log !== undefined) lines.push(`  log  ${o.log} = ${o.value}`);
      else { lines.push(`${o.ok ? 'PASS' : 'FAIL'}  ${o.check}${o.detail ? '  (' + o.detail + ')' : ''}`); if (!o.ok) fails++; }
    }
    const checks = out.filter(o => o.check).length;
    lines.push(`\ninteract: ${checks} checks, ${fails} failure(s)${shot ? ` · shot: ${shot}` : ''}`);
    return { text: lines.join('\n'), checks, fails };
  } finally { await p.close(); job.done(); }
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  const [page, scenarioFile, ...rest] = process.argv.slice(2);
  if (!page || !scenarioFile) { console.error('usage: interact.mjs <page> <scenario.js> [--shot out.png] [--width N]'); process.exit(2); }
  const shot = rest.includes('--shot') ? rest[rest.indexOf('--shot') + 1] : null;
  const width = rest.includes('--width') ? parseInt(rest[rest.indexOf('--width') + 1], 10) : 1500;   // viewport width (mobile checks)
  const [srv, browser] = await Promise.all([serve(), launch()]);
  const r = await runScenario(browser, srv, page, resolve(scenarioFile), { width, shot: shot && resolve(shot) });
  console.log(r.text);
  browser.close(); srv.close();
  process.exit(r.fails ? 1 : 0);
}
