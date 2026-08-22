// Sanity-check the simulator: step time by refinement level, cross-checked
// against analytic expectations. Run: node scripts/sanity.mjs
import { simulate, LEVELS, defaultConfig } from '../src/sim.js';
import { toChromeTrace } from '../src/trace.js';
import { summarize } from '../src/compare.js';
import { fmtUs, modelFlopsPerToken, DSV3, HARDWARE, peakFlops } from '../src/model.js';

const base = {}; // default config: H800, bf16, pp16/ep64/dp128, m=24, seq 4096
let prev = null;
const rows = [];
for (let level = 0; level < LEVELS.length; level++) {
  const { stats, cfg } = simulate({ ...base, level });
  rows.push({
    level: `${level} ${LEVELS[level].id}`,
    step: fmtUs(stats.stepUs),
    'Δ vs prev': prev ? '+' + fmtUs(Math.max(0, stats.stepUs - prev)) : '',
    'MFU %': (stats.mfu * 100).toFixed(1),
    'tok/s/GPU': Math.round(stats.tokPerSecPerGpu),
    'idle %': (stats.bubbleFrac * 100).toFixed(1),
  });
  prev = stats.stepUs;
}
console.table(rows);

// analytic roofline check
const cfg = defaultConfig(base);
const hw = HARDWARE[cfg.hardware];
const flops = modelFlopsPerToken(DSV3, cfg.seqLen) * cfg.dp * cfg.microbatches * cfg.mbs * cfg.seqLen;
const roofUs = flops / (cfg.pp * cfg.dp) / peakFlops(hw, cfg.dtype) * 1e6;
const l0 = simulate({ ...base, level: 0 }).stats.stepUs;
check('L0 == analytic roofline', Math.abs(l0 - roofUs) / roofUs < 1e-9, `${fmtUs(l0)} vs ${fmtUs(roofUs)}`);

// 1F1B bubble fraction ≈ (p-1)/(m+p-1) (approximate: stages are not uniform)
const l2 = simulate({ ...base, level: 2 }).stats;
const analytic = (cfg.pp - 1) / (cfg.microbatches + cfg.pp - 1);
check('L2 bubble ≈ (p-1)/(m+p-1)',
  Math.abs(l2.bubbleFrac - analytic) < 0.5 * analytic + 0.03,
  `measured ${(l2.bubbleFrac * 100).toFixed(1)}% vs analytic ${(analytic * 100).toFixed(1)}%`);

// step time should be monotone in level (each refinement adds cost) up to jitter
const steps = rows.map(r => r.step);
let mono = true;
for (let i = 2; i < LEVELS.length - 1; i++)
  if (simulate({ ...base, level: i }).stats.stepUs + 1 < simulate({ ...base, level: i - 1 }).stats.stepUs) mono = false;
check('step time monotone in level (2..5)', mono, steps.join(' → '));

// round-trip: chrome export of the sim summarizes to the same step time
const { trace, stats } = simulate({ ...base, level: 4 });
const sum = summarize(toChromeTrace(trace));
check('chrome round-trip step time', Math.abs(sum.stepUs - stats.stepUs) < 1, `${fmtUs(sum.stepUs)} vs ${fmtUs(stats.stepUs)}`);
console.log('\nper-GPU busy by category at level 4 (per step):');
console.table(Object.fromEntries(Object.entries(sum.byCat).map(([k, v]) => [k, fmtUs(v)])));

// Calibration anchors (sim should land BELOW both: no DualPipe / no interleaved-VP overlap yet).
// DeepSeek-V3: ~2.788M H800-hours for 14.8T tokens ≈ 1475 tok/s/GPU (2048 GPUs, FP8, DualPipe).
const real = simulate({ level: 6, dtype: 'mxfp8', microbatches: 120, gpus: 2048, ep: 64 }).stats;
console.log(`anchor DSv3/H800 (mxfp8 m=120 pp16 ep64): ${Math.round(real.tokPerSecPerGpu)} tok/s/GPU (real ≈ 1475)`);
// NVIDIA NeMo/Megatron-Bridge on 256x GB300: TP1/PP2/VP8/EP32, mbs1, GBS 4096
// -> 6338 tok/s/GPU = 1648 model TFLOP/s/GPU = 10.30 s/step (scaling-puzzles SOURCES.md).
const nemo = simulate({ level: 5, hardware: 'gb300', dtype: 'mxfp8', gpus: 256, pp: 2, ep: 32, microbatches: 32 }).stats;
console.log(`anchor NeMo/GB300 (mxfp8 m=32 pp2 ep32): ${Math.round(nemo.tokPerSecPerGpu)} tok/s/GPU, ` +
  `step ${fmtUs(nemo.stepUs)} (real 6338 tok/s/GPU, 10.30 s/step — sim runs optimistic here: ` +
  `gemmEff/comm assumptions unvalidated; VP8 interleave unmodeled)`);

// ---- memory model (GB300 focus) ----------------------------------------------
const { memoryUsage, actBreakdownPerToken, resolveMatmuls } = await import('../src/memory.js');
const mem = (o) => memoryUsage(defaultConfig({ hardware: 'gb300', ...o }));
const ddp = mem({ pp: 1, ep: 1, zero: 0, recompute: 'none' }); // dp = gpus/pp = 256
check('DDP replication never fits', !ddp.fits && ddp.worst.total > 4000, `${(ddp.worst.total / 1024).toFixed(1)} TiB`);
const dsv3 = mem({ pp: 16, ep: 16, zero: 1, recompute: 'selective' });
check('PP16/EP16/ZeRO-1 selective fits 256xGB300', dsv3.fits, `${dsv3.worst.total.toFixed(0)}/${dsv3.capacityGB} GiB`);
const fat = mem({ pp: 16, ep: 16, zero: 1, recompute: 'none', mbs: 4 });
check('mbs=4 without recompute is OOM', !fat.fits, `${fat.worst.total.toFixed(0)} GiB`);
const mmRecipe = resolveMatmuls({ recipe: 'nv-mxfp8' });
const acts = ['none', 'selective', 'full'].map(r =>
  Object.values(actBreakdownPerToken('moe', DSV3, r, mmRecipe)).reduce((x, y) => x + y, 0));
check('recompute strictly shrinks activations', acts[0] > acts[1] && acts[1] > acts[2],
  acts.map(b => (b / 1024).toFixed(0) + 'KiB/tok').join(' > '));
const actsBf16 = Object.values(actBreakdownPerToken('moe', DSV3, 'selective', resolveMatmuls({ recipe: 'bf16' }))).reduce((x, y) => x + y, 0);
check('fp8 recipe shrinks stashes vs bf16', acts[1] < actsBf16,
  `${(acts[1] / 1024).toFixed(0)} < ${(actsBf16 / 1024).toFixed(0)} KiB/tok`);

// ---- block-graph invariants (attn-replay policy, router state, vocab split) -----
const { blockGraph, analyze, RECOMPUTE_PRESETS } = await import('../src/blockgraph.js');
const mmFp8 = resolveMatmuls({ recipe: 'dsv3-fp8' });
const ar = analyze(blockGraph('moe', DSV3, mmFp8, 4096), RECOMPUTE_PRESETS['attn-replay']);
check('attn-replay stashes only {x0, norm2, dispatch, gate_up, router}',
  [...ar.neededSaved].sort().join(',') === 'dispatch,gate_up,norm2,router,x0',
  [...ar.neededSaved].sort().join(','));
check('attn-replay replays the whole MLA path (incl. residual add)',
  ['norm1', 'qkv_down', 'q_up', 'kv_up', 'attn', 'o_proj', 'x1'].every(id => ar.replayed.has(id)),
  `replay +${(ar.replayFrac * 100).toFixed(0)}% of fwd FLOPs`);
const routerNode = blockGraph('moe', DSV3, mmFp8, 4096).find(n => n.id === 'router');
check('router stash is the full retention set (~2.1 KiB/tok)',
  routerNode.outBytes === 4 * (2 * DSV3.routedExperts + 4 * DSV3.topk),
  `${(routerNode.outBytes / 1024).toFixed(2)} KiB/tok`);
const arT = analyze(blockGraph('moe', DSV3, mmFp8, 4096), RECOMPUTE_PRESETS['attn-replay'], true);
check('fp8ᵀ dual stash: wgrad-read fp8 only (dispatch, norm2; not gate_up/x0/router)',
  arT.dual.has('dispatch') && arT.dual.has('norm2') && !arT.dual.has('gate_up') && !arT.dual.has('x0') && !arT.dual.has('router'),
  [...arT.dual].sort().join(','));
check('fp8ᵀ adds exactly the dual payloads',
  Math.abs((arT.savedBytes - ar.savedBytes) - (ar.byId.dispatch.outBytes + ar.byId.norm2.outBytes)) < 1,
  `+${((arT.savedBytes - ar.savedBytes) / 1024).toFixed(1)} KiB/tok`);
const noT = mem({ hardware: 'h100', gpus: 2048, pp: 8, ep: 64, zero: 1, recompute: 'attn-replay', recipe: 'dsv3-fp8' });
const withT = mem({ hardware: 'h100', gpus: 2048, pp: 8, ep: 64, zero: 1, recompute: 'attn-replay', recipe: 'dsv3-fp8', transposedStash: true });
check('transposedStash grows the H100 watermark by GiBs',
  withT.worst.total > noT.worst.total + 3,
  `${noT.worst.total.toFixed(1)} -> ${withT.worst.total.toFixed(1)} GiB`);
const vs = mem({ pp: 4, ep: 16, zero: 1, recompute: 'selective' }).perStage;
check('vocab weights split: pp0/last only, grads follow',
  vs[0].vocab > 0 && vs[1].vocab === 0 && vs.at(-1).vocab > 0 && vs[0].vocabGrads > 0 && vs[1].vocabGrads === 0,
  `pp0 ${vs[0].vocab.toFixed(1)} + g${vs[0].vocabGrads.toFixed(1)} GiB, mid 0`);

// ---- individual refinement toggles & ZeRO variants ------------------------------
const z1 = simulate({ level: 4, zero: 1 });
const names = new Set();
for (const r of z1.trace.ranks) for (const t of r.tracks) for (const s of t.slices) names.add(s.name.split(' ')[0]);
check('ZeRO-1 all-reduces instead of AG/RS', names.has('allreduce') && !names.has('allgather'), [...names].filter(n => /all|reduce/.test(n)).join(','));
const solo = simulate({ level: 1, refinements: { epComm: true } });
check('a2a toggles on without pipeline', solo.trace.ranks.some(r => r.tracks.some(t => t.name === 'a2a')),
  `step ${fmtUs(solo.stats.stepUs)}`);

function check(name, ok, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  (${detail})`);
  if (!ok) process.exitCode = 1;
}
