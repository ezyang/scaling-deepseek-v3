// Interaction-test driver: run a scenario script against a page in headless
// Chrome and report assertions. Complements diagramlint (static geometry)
// with sequenced clicks/hovers/state probes.
//
//   node scripts/interact.mjs <page-path> <scenario-file> [--shot out.png]
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
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { extname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const [page, scenarioFile, ...rest] = process.argv.slice(2);
if (!page || !scenarioFile) { console.error('usage: interact.mjs <page> <scenario.js> [--shot out.png]'); process.exit(2); }
const shot = rest[rest.indexOf('--shot') + 1] && rest.includes('--shot') ? rest[rest.indexOf('--shot') + 1] : null;
const width = rest.includes('--width') ? parseInt(rest[rest.indexOf('--width') + 1], 10) : 1500;   // viewport width (mobile checks)

const root = fileURLToPath(new URL('..', import.meta.url));
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.png': 'image/png' };
const CHROME = process.env.CHROME ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

const HARNESS = `
const T = {
  el: (sel) => document.querySelector(sel),
  text: (sel) => document.querySelector(sel)?.textContent.trim() ?? null,
  click: (sel) => document.querySelector(sel)?.dispatchEvent(new MouseEvent('click', { bubbles: true })),
  hover: (sel) => document.querySelector(sel)?.dispatchEvent(new MouseEvent('mouseenter')),
  unhover: (sel) => document.querySelector(sel)?.dispatchEvent(new MouseEvent('mouseleave')),
  tick: (ms = 120) => new Promise(r => setTimeout(r, ms)),
  _out: [],
  check: (name, cond, detail = '') => T._out.push({ check: name, ok: !!cond, detail: String(detail) }),
  log: (name, value) => T._out.push({ log: name, value: String(value) }),
  done: () => {
    const pre = document.createElement('pre'); pre.id = 'interact-out';
    pre.textContent = JSON.stringify(T._out, null, 1);
    document.body.append(pre);
    document.title = 'INTERACT-DONE';
  },
};
`;

const scenario = await readFile(resolve(scenarioFile), 'utf8');
const pageHtml = await readFile(join(root, page), 'utf8');
const injected = pageHtml.replace('</body>',
  `<script type="module">${HARNESS}\n(async () => {\n await T.tick(500);\n${scenario}\n})().catch(e => { T._out.push({ error: String(e) }); T.done(); });</script></body>`);

const srv = createServer(async (req, res) => {
  try {
    const path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    if (path === '/__page__.html' || path === '/' + page) {
      res.writeHead(200, { 'content-type': 'text/html' }); res.end(injected); return;
    }
    const body = await readFile(join(root, path));
    res.writeHead(200, { 'content-type': MIME[extname(path)] ?? 'application/octet-stream' });
    res.end(body);
  } catch { res.writeHead(404); res.end(); }
});
await new Promise(r => srv.listen(0, r));
const url = `http://localhost:${srv.address().port}/${page}`;

const args = ['--headless', '--disable-gpu', '--hide-scrollbars', '--virtual-time-budget=12000', `--window-size=${width},4000`];
const dom = await new Promise((res2, rej) => execFile(CHROME, [...args, '--dump-dom', url],
  { maxBuffer: 64 * 1024 * 1024 }, (err, stdout) => err ? rej(err) : res2(stdout)));
if (shot) await new Promise((res2, rej) => execFile(CHROME, [...args, `--screenshot=${shot}`, url],
  { maxBuffer: 64 * 1024 * 1024 }, (err) => err ? rej(err) : res2()));
srv.close();

const m = dom.match(/<pre id="interact-out">([\s\S]*?)<\/pre>/);
if (!m) { console.error('scenario produced no output (did it call T.done()?)'); process.exit(2); }
const unesc = (s) => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&amp;/g, '&');
const out = JSON.parse(unesc(m[1]));
let fails = 0;
for (const o of out) {
  if (o.error) { console.log('ERROR ' + o.error); fails++; }
  else if (o.log !== undefined) console.log(`  log  ${o.log} = ${o.value}`);
  else { console.log(`${o.ok ? 'PASS' : 'FAIL'}  ${o.check}${o.detail ? '  (' + o.detail + ')' : ''}`); if (!o.ok) fails++; }
}
console.log(`\ninteract: ${out.filter(o => o.check).length} checks, ${fails} failure(s)${shot ? ` · shot: ${shot}` : ''}`);
process.exit(fails ? 1 : 0);
