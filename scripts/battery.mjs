// The whole test battery, in parallel: sanity + diagramlint + every
// interaction scenario in tests/ (each declares its page in a `// @page …`
// header; optional `// @args --width N` passes extra interact.mjs flags).
//
//   node scripts/battery.mjs [name-substring …]   # filter scenarios by name
//
// Chrome startup dominates a scenario's wall time and the runs are fully
// independent (each interact.mjs spawns its own server + browser), so the
// battery runs them concurrently and reports one line per job.
import { readdir, readFile } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { availableParallelism } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const filters = process.argv.slice(2);
const pick = (name) => !filters.length || filters.some(f => name.includes(f));

const jobs = [];
if (pick('sanity')) jobs.push({ name: 'sanity', args: ['scripts/sanity.mjs'] });
if (pick('diagramlint')) jobs.push({ name: 'diagramlint', args: ['scripts/diagramlint.mjs'] });
for (const f of (await readdir(join(root, 'tests'))).filter(f => f.endsWith('.js')).sort()) {
  const name = f.replace(/\.js$/, '');
  if (!pick(name)) continue;
  const src = await readFile(join(root, 'tests', f), 'utf8');
  const page = src.match(/^\/\/ @page (\S+)/m)?.[1];
  if (!page) { console.error(`SKIP ${name}: no "// @page" header`); process.exitCode = 1; continue; }
  const extra = src.match(/^\/\/ @args (.+)$/m)?.[1].trim().split(/\s+/) ?? [];
  jobs.push({ name, args: ['scripts/interact.mjs', page, join('tests', f), ...extra] });
}

const limit = Math.max(2, Math.min(8, availableParallelism() - 2));
const t0 = performance.now();
let next = 0, failed = 0;
const run = (job) => new Promise((res) => {
  const t = performance.now();
  execFile('node', job.args, { cwd: root, maxBuffer: 16 * 1024 * 1024 }, (err, stdout, stderr) => {
    const secs = ((performance.now() - t) / 1000).toFixed(1);
    const out = stdout + stderr;
    const tally = out.match(/interact: (\d+) checks/)?.[1]
      ?? out.match(/sanity: (\d+)/)?.[1]
      ?? out.match(/(\d+) finding/)?.[1];
    if (err) {
      failed++;
      console.log(`FAIL  ${job.name}  (${secs}s)`);
      console.log(out.split('\n').filter(l => /FAIL|ERROR|error/.test(l) || !out.includes('interact:')).slice(-15).map(l => '      ' + l).join('\n'));
    } else {
      console.log(`pass  ${job.name}  (${tally ? tally + ' checks, ' : ''}${secs}s)`);
    }
    res();
  });
});
await Promise.all(Array.from({ length: limit }, async () => {
  while (next < jobs.length) await run(jobs[next++]);
}));
console.log(`\nbattery: ${jobs.length} jobs, ${failed} failure(s), ${((performance.now() - t0) / 1000).toFixed(1)}s`);
process.exit(failed || process.exitCode ? 1 : 0);
