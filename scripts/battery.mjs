// The whole test battery, in parallel: sanity + diagramlint + every
// interaction scenario in tests/ (each declares its page in a `// @page …`
// header; optional `// @args --width N` passes extra interact.mjs flags).
//
//   node scripts/battery.mjs [name-substring …]   # filter scenarios by name
//
// Scenarios run in-process, each in a fresh browser context of a small pool
// of long-lived browsers (scripts/cdp.mjs); the scripts that aren't
// scenarios run as subprocesses. Everything runs concurrently, longest
// first, one report line per job.
import { readdir, readFile } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { availableParallelism } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { launch, serve } from './cdp.mjs';
import { runScenario } from './interact.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));
const filters = process.argv.slice(2);
const pick = (name) => !filters.length || filters.some(f => name.includes(f));

const jobs = [];
if (pick('sanity')) jobs.push({ name: 'sanity', args: ['scripts/sanity.mjs'] });
if (pick('stamp')) jobs.push({ name: 'stamp', args: ['scripts/stamp.mjs', '--check'] });
if (pick('goldens')) jobs.push({ name: 'goldens', args: ['scripts/goldens.mjs'] });
if (pick('pixel')) jobs.push({ name: 'pixel', args: ['scripts/pixelgold.mjs'] });
if (pick('diagramlint')) jobs.push({ name: 'diagramlint', args: ['scripts/diagramlint.mjs'] });
for (const f of (await readdir(join(root, 'tests'))).filter(f => f.endsWith('.js')).sort()) {
  const name = f.replace(/\.js$/, '');
  if (!pick(name)) continue;
  const src = await readFile(join(root, 'tests', f), 'utf8');
  const page = src.match(/^\/\/ @page (\S+)/m)?.[1];
  if (!page) { console.error(`SKIP ${name}: no "// @page" header`); process.exitCode = 1; continue; }
  const extra = src.match(/^\/\/ @args (.+)$/m)?.[1].trim().split(/\s+/) ?? [];
  const width = extra.includes('--width') ? parseInt(extra[extra.indexOf('--width') + 1], 10) : undefined;
  jobs.push({ name, page, file: join('tests', f), width, size: src.length });
}
jobs.sort((a, b) => (b.size ?? Infinity) - (a.size ?? Infinity));   // longest scenarios first: they set the critical path

const limit = Math.max(2, availableParallelism() - 2);
const POOL = 4;   // browsers shared by the scenario workers
const t0 = performance.now();
const [srv, ...browsers] = jobs.some(j => j.file) ? await Promise.all([serve(), ...Array.from({ length: POOL }, () => launch())]) : [];
let next = 0, failed = 0;
const sub = (job) => new Promise((res) => execFile('node', job.args, { cwd: root, maxBuffer: 16 * 1024 * 1024, timeout: 120_000 },
  (err, stdout, stderr) => res({ ok: !err, out: stdout + stderr })));
const scenario = (job, browser) => runScenario(browser, srv, job.page, job.file, { width: job.width })
  .then(r => ({ ok: !r.fails, out: r.text }), e => ({ ok: false, out: 'ERROR ' + e.message }));
const run = async (job, w) => {
  const t = performance.now();
  const { ok, out } = await (job.file ? scenario(job, browsers[w % POOL]) : sub(job));
  const secs = ((performance.now() - t) / 1000).toFixed(1);
  const tally = out.match(/interact: (\d+) checks/)?.[1]
    ?? out.match(/sanity: (\d+)/)?.[1]
    ?? out.match(/(\d+) finding/)?.[1];
  if (!ok) {
    failed++;
    console.log(`FAIL  ${job.name}  (${secs}s)`);
    console.log(out.split('\n').filter(l => /FAIL|ERROR|error/.test(l) || !out.includes('interact:')).slice(-15).map(l => '      ' + l).join('\n'));
  } else {
    console.log(`pass  ${job.name}  (${tally ? tally + ' checks, ' : ''}${secs}s)`);
  }
};
await Promise.all(Array.from({ length: limit }, async (_, w) => {
  while (next < jobs.length) await run(jobs[next++], w);
}));
for (const b of browsers) b.close();
srv?.close();
console.log(`\nbattery: ${jobs.length} jobs, ${failed} failure(s), ${((performance.now() - t0) / 1000).toFixed(1)}s`);
process.exit(failed || process.exitCode ? 1 : 0);
