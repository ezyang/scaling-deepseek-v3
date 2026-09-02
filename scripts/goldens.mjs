// Numeric goldens: the modeling layer's outputs, pinned in tests/goldens.json.
// Changing a number is allowed — invisibly changing one is not: this gate
// fails on any drift, and `--update` + the git diff of the golden file is the
// review artifact. Taste: sanity.mjs already PROVES the shard-assembly math
// (96-config independent replay, ===), so goldens pin what that proof takes
// as input — per-token stash rates, FLOP counts, sim step times, the memory
// model — plus a few configs with narrative meaning in 02, not the
// combinatorial product.
//
//   node scripts/goldens.mjs           # compare (battery job)
//   node scripts/goldens.mjs --update  # rewrite tests/goldens.json
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { simulate, LEVELS } from '../src/sim.js';
import { modelFlopsPerToken, DSV3, HARDWARE } from '../src/model.js';
import { PARAMS } from '../src/params.js';
import { memoryUsage } from '../src/memory.js';
import { resolveMatmuls, RECIPES } from '../src/recipes.js';
import { blockGraph, analyze, RECOMPUTE_PRESETS, DTYPE_BYTES } from '../src/blockgraph.js';
import { defaultConfig } from '../src/sim.js';
import { buildCells, cellsEnv } from '../src/cells.js';

const FILE = fileURLToPath(new URL('../tests/goldens.json', import.meta.url));
const G = {};

// architecture + rate tables (literals in source; pinned against typo edits)
G.arch = DSV3;
G.dtypeBytes = DTYPE_BYTES;
G.hardware = HARDWARE;
G.flopsPerToken = { seq4096: modelFlopsPerToken(DSV3, 4096), seq8192: modelFlopsPerToken(DSV3, 8192) };

// per-token stash rates: every kind × recipe × policy (× fp8ᵀ dual) — these
// are the blockgraph outputs the cell graph and fit charts price from
G.stash = {};
for (const kind of ['moe', 'dense'])
  for (const recipe of Object.keys(RECIPES))
    for (const policy of Object.keys(RECOMPUTE_PRESETS))
      for (const T of [false, true]) {
        const a = analyze(blockGraph(kind, DSV3, resolveMatmuls({ recipe }), 4096), RECOMPUTE_PRESETS[policy], T);
        G.stash[`${kind}·${recipe}·${policy}${T ? '·T' : ''}`] =
          { savedBytes: a.savedBytes, replayFrac: a.replayFrac, ...(T ? {} : { buckets: a.buckets }) };
      }

// trace sim: step time + MFU per refinement level (seeded — deterministic),
// plus the two calibration anchors
G.sim = {};
for (let level = 0; level < LEVELS.length; level++) {
  const { stats } = simulate({ level });
  G.sim[`L${level}-${LEVELS[level].id}`] = { stepUs: stats.stepUs, mfu: stats.mfu };
}
for (const [name, cfg] of [
  ['anchor-dsv3-h800', { level: 6, dtype: 'mxfp8', microbatches: 120, gpus: 2048, ep: 64 }],
  ['anchor-nemo-gb300', { level: 5, hardware: 'gb300', dtype: 'mxfp8', gpus: 256, pp: 2, ep: 32, microbatches: 32 }],
]) {
  const { stats } = simulate(cfg);
  G.sim[name] = { stepUs: stats.stepUs, tokPerSecPerGpu: stats.tokPerSecPerGpu };
}

// memoryUsage (the sim's memory model — a separate code path from the cells)
G.memory = {};
for (const [name, cfg] of [
  ['gb300-ddp', { hardware: 'gb300', pp: 1, ep: 1, zero: 0, recompute: 'none' }],
  ['gb300-dsv3', { hardware: 'gb300', pp: 16, ep: 16, zero: 1, recompute: 'selective' }],
  ['gb300-mbs4-norecompute', { hardware: 'gb300', pp: 16, ep: 16, zero: 1, recompute: 'none', mbs: 4 }],
  ['h100-prod', { hardware: 'h100', gpus: 2048, pp: 8, ep: 64, zero: 1, recompute: 'attn-replay', recipe: 'dsv3-fp8' }],
  ['h100-prod-T', { hardware: 'h100', gpus: 2048, pp: 8, ep: 64, zero: 1, recompute: 'attn-replay', recipe: 'dsv3-fp8', transposedStash: true }],
]) {
  const m = memoryUsage(defaultConfig(cfg));
  G.memory[name] = { worstGiB: m.worst.total, fits: m.fits, stage0GiB: m.perStage[0].total };
}

// the cell graph at story configs (assembly identity is sanity's job; these
// pin the assembled headline numbers for configs 02's narrative stands on).
// cellsEnv IS the production assembly — the same code the widget and sheet run.
const envFor = ({ pp, ep, zero, stage, fp8p = false, recipe, policy }) => {
  const mm = resolveMatmuls({ recipe });
  const [anaM, anaD, fM, fD] = [['moe', policy], ['dense', policy], ['moe', 'none'], ['dense', 'none']]
    .map(([k, p]) => analyze(blockGraph(k, DSV3, mm, 4096), RECOMPUTE_PRESETS[p], false));
  return cellsEnv({ pp, ep, zero, stage, fp8p, vpp: pp > 1 ? 2 : 1, fold: 'reflect' }, anaM, anaD, fM, fD);
};
const EP = { pp: 8, ep: 64, zero: 1, stage: 1, recipe: 'dsv3-fp8', policy: 'dsv3' };
G.cells = {};
for (const [name, cfg] of [
  ['bf16-none-peak', { ...EP, recipe: 'bf16', policy: 'none' }],
  ['endpoint', EP],
  ['endpoint-fp8p', { ...EP, fp8p: true }],
  ['endpoint-attn-replay', { ...EP, policy: 'attn-replay' }],
  ['endpoint-all-fp8', { ...EP, recipe: 'all-fp8' }],
  ['endpoint-zero3', { ...EP, zero: 3 }],
  ['endpoint-stage0', { ...EP, stage: 0 }],
  ['no-parallelism', { ...EP, pp: 1, ep: 1, zero: 0, stage: 0 }],
]) {
  const { get } = buildCells(envFor(cfg));
  G.cells[name] = Object.fromEntries(['T1', 'W1', 'G1', 'O1', 'A1'].map((id) => [id, get(id)]));
}

// ---- the sheet as TEXT (tests/goldens-sheet.txt) ---------------------------
// One line per cell at the essay's endpoint config: id · label · exact value
// · unit · formula (or input kind). Formulas are SHAPE-STABLE across configs
// (the stability rule), so one config pins every formula string — this file
// is the reviewable git diff for sheet-restructuring commits (the pixel
// goldens defeat git diff; this doesn't). Values are dyadic: String() is
// exact and round-trips.
const SHEET_FILE = fileURLToPath(new URL('../tests/goldens-sheet.txt', import.meta.url));
const sheetText = () => buildCells(envFor(EP)).cells.map((c) => {
  const label = '  '.repeat(c.depth ?? 0) + c.label;
  return (c.id.padEnd(5) + label.padEnd(56) + String(c.value).padStart(15)
    + ' ' + (c.unit ?? '').padEnd(6) + (c.expr ? '= ' + c.expr : (c.note ?? '(model input)'))).trimEnd();
}).join('\n') + '\n';

// ---- compare / update -----------------------------------------------------
if (process.argv.includes('--update')) {
  writeFileSync(FILE, JSON.stringify(G, null, 2) + '\n');
  writeFileSync(SHEET_FILE, sheetText());
  console.log(`goldens: wrote ${Object.keys(G).map((k) => `${k}(${Object.keys(G[k]).length})`).join(' ')} + sheet text`);
  process.exit(0);
}
let old;
try { old = JSON.parse(readFileSync(FILE, 'utf8')); }
catch { console.error('goldens: no tests/goldens.json — run `node scripts/goldens.mjs --update`'); process.exit(1); }
const diffs = [];
const walk = (a, b, path) => {
  if (typeof a === 'number' && typeof b === 'number') {
    const ok = Number.isInteger(a) && Number.isInteger(b)
      ? a === b
      : Math.abs(a - b) <= 1e-9 * Math.max(Math.abs(a), Math.abs(b));
    if (!ok) diffs.push(`${path}: ${b} → ${a}`);
  } else if (a && b && typeof a === 'object' && typeof b === 'object') {
    for (const k of new Set([...Object.keys(a), ...Object.keys(b)]))
      if (!(k in a)) diffs.push(`${path}.${k}: removed (was ${JSON.stringify(b[k])})`);
      else if (!(k in b)) diffs.push(`${path}.${k}: new (${JSON.stringify(a[k])})`);
      else walk(a[k], b[k], `${path}.${k}`);
  } else if (JSON.stringify(a) !== JSON.stringify(b)) diffs.push(`${path}: ${JSON.stringify(b)} → ${JSON.stringify(a)}`);
};
walk(G, old, '');
// the sheet text compares line-wise (the git diff of the file is the review)
try {
  const oldTxt = readFileSync(SHEET_FILE, 'utf8'), newTxt = sheetText();
  if (oldTxt !== newTxt) {
    const A = oldTxt.split('\n'), B = newTxt.split('\n');
    for (let i = 0; i < Math.max(A.length, B.length); i++)
      if (A[i] !== B[i]) { diffs.push(`sheet-text line ${i + 1}: ${JSON.stringify(A[i] ?? '∅')} → ${JSON.stringify(B[i] ?? '∅')}`); if (diffs.length > 40) break; }
  }
} catch { diffs.push('sheet-text: tests/goldens-sheet.txt missing — run --update'); }
for (const d of diffs) console.log('DRIFT ' + d);
if (diffs.length) console.log(`\ngoldens: ${diffs.length} drift(s) — intentional? \`node scripts/goldens.mjs --update\` and review the git diff`);
else console.log(`goldens: match (${Object.entries(G).reduce((n, [, v]) => n + Object.keys(v).length, 0)} pinned groups)`);
process.exit(diffs.length ? 1 : 0);
