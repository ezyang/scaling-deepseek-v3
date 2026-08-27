// Diagram lint driver: serves the repo, renders scripts/diagramlint.html in
// headless Chrome, and reports visual-grammar violations (see
// docs/diagram-grammar.md): arrowheads in text, wires through labels, label
// collisions, clipped/border-cut text, overlapping op boxes.
// Known-acceptable findings live in scripts/diagramlint-allow.json (substring
// match against the JSON of a finding). Exit 1 on any unallowed finding.
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromePath } from './chromepath.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.png': 'image/png' };
const CHROME = chromePath();

const srv = createServer(async (req, res) => {
  try {
    const path = join(root, decodeURIComponent(new URL(req.url, 'http://x').pathname));
    const body = await readFile(path);
    res.writeHead(200, { 'content-type': MIME[extname(path)] ?? 'application/octet-stream' });
    res.end(body);
  } catch { res.writeHead(404); res.end(); }
});
await new Promise(r => srv.listen(0, r));
const url = `http://localhost:${srv.address().port}/scripts/diagramlint.html`;

const dom = await new Promise((resolve, reject) => {
  execFile(CHROME, ['--headless', '--disable-gpu', '--hide-scrollbars', '--virtual-time-budget=12000',
    '--window-size=1500,4000', '--dump-dom', url], { maxBuffer: 64 * 1024 * 1024 },
    (err, stdout) => err ? reject(err) : resolve(stdout));
});
srv.close();

const m = dom.match(/<pre id="lint-out">([\s\S]*?)<\/pre>/);
if (!m) { console.error('lint page produced no output (widgets failed to render?)'); process.exit(2); }
const unescape = (s) => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&quot;/g, '"').replace(/&amp;/g, '&');
const findings = JSON.parse(unescape(m[1]) || '[]');

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
