// Compare a real profiler trace (PyTorch/Kineto Chrome JSON, .json or .json.gz)
// against the simulator, category by category.
//
// Usage:
//   node scripts/compare.mjs real_trace.json [--config '{"hardware":"h800","dtype":"mxfp8","microbatches":120}'] [--level N]
//   node scripts/compare.mjs simA.json simB.json     # compare two trace files
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
import { simulate } from '../src/sim.js';
import { toChromeTrace } from '../src/trace.js';
import { summarize, diff } from '../src/compare.js';

const args = process.argv.slice(2);
const files = args.filter(a => !a.startsWith('--') && !args[args.indexOf(a) - 1]?.startsWith('--'));
const flag = (name, dflt) => {
  const i = args.indexOf('--' + name);
  return i >= 0 ? args[i + 1] : dflt;
};
if (files.length === 0) {
  console.error('usage: node scripts/compare.mjs <real.json[.gz]> [sim.json] [--config JSON] [--level N]');
  process.exit(1);
}
const load = (f) => JSON.parse(f.endsWith('.gz') ? gunzipSync(readFileSync(f)) : readFileSync(f, 'utf8'));

const real = summarize(load(files[0]), { label: files[0] });
let simSum;
if (files[1]) {
  simSum = summarize(load(files[1]), { label: files[1] });
} else {
  const overrides = JSON.parse(flag('config', '{}'));
  if (flag('level')) overrides.level = +flag('level');
  overrides.granularity = 'op'; // exact category attribution
  overrides.level ??= 6;
  const { trace, cfg } = simulate(overrides);
  simSum = summarize(toChromeTrace(trace), { label: `sim L${cfg.level} ${cfg.hardware}/${cfg.dtype}` });
}
console.log(`${files[0]}: ${real.nGpu} GPU pid(s), ` +
  `${real.usedProfilerSteps ? 'window = median ProfilerStep' : 'window = full trace'}, ` +
  `busy ${(real.busyFrac * 100).toFixed(1)}%`);
console.table(diff(real, simSum));
