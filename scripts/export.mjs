// Emit a simulated trace as Chrome trace JSON (opens in ui.perfetto.dev).
// Usage: node scripts/export.mjs [--level N] [--out trace.json] [--config '{"hardware":"gb200"}']
import { writeFileSync } from 'node:fs';
import { simulate } from '../src/sim.js';
import { toChromeTrace } from '../src/trace.js';
import { fmtUs } from '../src/model.js';

const args = process.argv.slice(2);
const flag = (name, dflt) => {
  const i = args.indexOf('--' + name);
  return i >= 0 ? args[i + 1] : dflt;
};
const overrides = JSON.parse(flag('config', '{}'));
if (flag('level')) overrides.level = +flag('level');
const out = flag('out', 'dsv3-sim.json');

const { trace, stats, cfg } = simulate(overrides);
writeFileSync(out, JSON.stringify(toChromeTrace(trace, out.replace(/\.json$/, ''))));
console.log(`wrote ${out}: level ${cfg.level}, ${cfg.hardware}/${cfg.dtype}, ` +
  `step ${fmtUs(stats.stepUs)}, MFU ${(stats.mfu * 100).toFixed(1)}%`);
console.log('view: open https://ui.perfetto.dev and drag the file in');
