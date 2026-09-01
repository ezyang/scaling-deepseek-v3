// Sanity-check the simulator: step time by refinement level, cross-checked
// against analytic expectations. Run: node scripts/sanity.mjs
import { simulate, LEVELS, defaultConfig } from '../src/sim.js';
import { toChromeTrace } from '../src/trace.js';
import { summarize } from '../src/compare.js';
import { fmtUs, modelFlopsPerToken, DSV3, HARDWARE, peakFlops } from '../src/model.js';
import { PARAMS } from '../src/params.js';

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

// Exact checkpoint-header audit (model tensors only; FP8 scale metadata excluded).
check('DSv3 exact main-model parameter count', PARAMS.total === 671026419200,
  PARAMS.total.toLocaleString('en-US'));
// DeepSeek's corrected convention (README_WEIGHTS.md): input embedding not
// counted, output head counted in full — "36.6B", advertised as 37B
check('DSv3 exact activated parameter count', PARAMS.activeTotal === 36625618432,
  PARAMS.activeTotal.toLocaleString('en-US'));

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
const { memoryUsage, actBreakdownPerToken, resolveMatmuls, totalParams } = await import('../src/memory.js');
check('memory inventory matches exact main-model parameter count',
  totalParams(DSV3, 1, 1) === PARAMS.total,
  totalParams(DSV3, 1, 1).toLocaleString('en-US'));
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
// ---- the production H100 config (notes.txt): attention checkpointed
// (= attn-replay) + every linear fp8 (= all-fp8) + fp8ᵀ dual stash. Under
// attn-replay the o_proj stash never materializes, so all-fp8's BYTES equal
// dsv3-fp8's — the recipes differ only in the o_proj GEMM's compute pricing.
// The absolute numbers are pinned as regression anchors for the external
// crosscheck (Haziza's analysis).
const amaia = analyze(blockGraph('moe', DSV3, resolveMatmuls({ recipe: 'all-fp8' }), 4096), RECOMPUTE_PRESETS['attn-replay'], true);
check('all-fp8 ≡ dsv3-fp8 bytes under attn-replay (o_proj replayed, stash moot)',
  Math.abs(amaia.savedBytes - arT.savedBytes) < 1e-6, `${(amaia.savedBytes / 1024).toFixed(3)} KiB/tok`);
check('production H100 config stash: 118.2 KiB/tok (183.2 with fp8ᵀ)',
  Math.abs(ar.savedBytes / 1024 - 118.223) < 0.01 && Math.abs(amaia.savedBytes / 1024 - 183.191) < 0.01,
  `${(ar.savedBytes / 1024).toFixed(3)} / ${(amaia.savedBytes / 1024).toFixed(3)} KiB/tok`);
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

// conservation: the local-lens stage split (viewer.js ppStage — the same
// decomposition every fit chart renders from) must PARTITION the
// checkpoint-exact parameter total for every schedule geometry: nothing
// dropped (final norm, block norms), nothing doubled (DualPipeV puts embed
// AND head on rank 0). The visual audit (src/audit.js) deliberately never
// re-derives the model — this is where that identity is enforced.
globalThis.HTMLElement = class {};   // viewer.js defines custom elements; the math exports are DOM-free
const { ppStage, inflightOf } = await import('../src/viewer.js');
{
  const moeExp = PARAMS.expert * DSV3.routedExperts;
  let worst = null;
  for (const pp of [1, 2, 4, 8, 16, 32, 64])
    for (const vpp of [1, 2, 4])
      for (const fold of ['reflect', 'wrap']) {
        let t = 0;
        for (let s2 = 0; s2 < pp; s2++) {
          const g = ppStage(s2, pp, vpp, fold);
          t += g.dense * PARAMS.denseBlock + g.moe * (PARAMS.moeBlock - moeExp) + g.moe * moeExp
            + ((g.emb ? 1 : 0) + (g.head ? 1 : 0)) * PARAMS.embed + (g.head ? DSV3.hidden : 0);
        }
        if (t !== PARAMS.total) worst = `pp${pp}·vpp${vpp}·${fold}: ${t}`;
      }
  check('stage split partitions the exact total (42 geometries)', worst === null, worst ?? PARAMS.total.toLocaleString('en-US'));
}

// The CELL GRAPH (src/cells.js): the formula strings the fit chart's
// tooltips and the sheet display are what the chart itself evaluates. An
// independent replay of the shard math must agree EXACTLY (===, no epsilon):
// every divisor is a power of two and every dtype rate a dyadic rational on
// integer counts ≪ 2^53, so IEEE doubles carry the byte counts with zero
// rounding — bytes are exact, not float-ish.
{
  const { buildCells } = await import('../src/cells.js');
  const { actBucketsOf, ACT_BUCKETS } = await import('../src/viewer.js');
  const moeExp = PARAMS.expert * DSV3.routedExperts;
  const mmS = resolveMatmuls({ recipe: 'dsv3-fp8' });
  const anaM = analyze(blockGraph('moe', DSV3, mmS, 4096), RECOMPUTE_PRESETS.dsv3, false);
  const anaD = analyze(blockGraph('dense', DSV3, mmS, 4096), RECOMPUTE_PRESETS.dsv3, false);
  const aM = anaM.savedBytes, aD = anaD.savedBytes;
  const bM = actBucketsOf(anaM), bD = actBucketsOf(anaD);
  // the save-everything rates at the same recipe (the R• factorization)
  const bMF = actBucketsOf(analyze(blockGraph('moe', DSV3, mmS, 4096), RECOMPUTE_PRESETS.none, false));
  const bDF = actBucketsOf(analyze(blockGraph('dense', DSV3, mmS, 4096), RECOMPUTE_PRESETS.none, false));
  const envOf = (S) => ({
    world: 2048, pp: S.pp, ep: S.ep, zero: S.zero, sched: '1f1b', fp8p: !!S.fp8p,
    g: ppStage(Math.min(S.stage, S.pp - 1), S.pp, S.pp > 1 ? 2 : 1, 'reflect'),
    aM, aD, bM, bD, bMF, bDF, bLabels: ACT_BUCKETS.map((b) => b.label),
    N: { restLayer: PARAMS.moeBlock - moeExp, denseLayer: PARAMS.denseBlock },
  });
  // the essay's endpoint (step 6, the peak rank): pin the exact integer
  const c6 = buildCells(envOf({ pp: 8, ep: 64, zero: 1, stage: 1 }));
  check('cells: step-6 endpoint total = exactly 66,296,545,344 B (an integer)',
    c6.get('T1') === 66296545344, String(c6.get('T1')));
  let worst = null;
  for (const pp of [1, 8]) for (const ep of [1, 4, 64]) for (const zero of [0, 1, 2, 3])
    for (const stage of [0, 1]) for (const fp8p of [false, true]) {
      const S = { pp, ep, zero, stage, fp8p };
      const env = envOf(S), { get } = buildCells(env);
      const g = env.g, dp = 2048 / pp, edp = dp / ep;
      const q = {
        e: g.moe * moeExp / ep,
        d: g.dense * PARAMS.denseBlock + g.moe * (PARAMS.moeBlock - moeExp),
        v: ((g.emb ? 1 : 0) + (g.head ? 1 : 0)) * PARAMS.embed + (g.head ? DSV3.hidden : 0),
      };
      const comp = (bpp, zt, w = 1) => {
        const se = zero >= zt ? bpp / edp : bpp, sd = zero >= zt ? bpp / dp : bpp;
        return q.e * se * w + q.d * sd * w + q.v * sd;
      };
      const IF = inflightOf('1f1b', stage, pp, pp > 1 ? 2 : 1, 'reflect');
      const acts = (g.dense * aD + g.moe * aM) * 4096 * IF
        + ((g.emb ? 2 * DSV3.hidden : 0) + (g.head ? 6 * DSV3.vocab : 0)) * 4096;
      const want = [comp(2, 3, fp8p ? 2.0625 / 2 : 1), comp(4, 2), comp(8, 1), acts];
      const got = ['W1', 'G1', 'O1', 'A1'].map(get);
      if (!got.every((v, i) => v === want[i])) { worst = `${JSON.stringify(S)}: ${got} ≠ ${want}`; }
      if (get('P6') !== IF) worst = `${JSON.stringify(S)}: P6 ${get('P6')} ≠ inflightOf ${IF}`;
      if (get('N1') !== moeExp || get('N4') !== PARAMS.embed) worst = `${JSON.stringify(S)}: N-cells drifted from PARAMS`;
      // the accordion sub-cells: per-class components and per-bucket stashes
      // (their sums ARE the parents' formulas — verified against the
      // aggregate math above)
      const clsWant = (bpp, zt, w = 1) => {
        const se = zero >= zt ? bpp / edp : bpp, sd = zero >= zt ? bpp / dp : bpp;
        return [q.e * se * w, q.d * sd * w, q.v * sd];
      };
      const partWant = [clsWant(2, 3, fp8p ? 2.0625 / 2 : 1), clsWant(4, 2), clsWant(8, 1)];
      const partIds = [['W2', 'W3', 'W4'], ['G2', 'G3', 'G4'], ['O2', 'O3', 'O4']];
      for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++)
        if (get(partIds[j][k]) !== partWant[j][k]) worst = `${JSON.stringify(S)}: ${partIds[j][k]} ${get(partIds[j][k])} ≠ ${partWant[j][k]}`;
      const vocab = ((g.emb ? 2 * DSV3.hidden : 0) + (g.head ? 6 * DSV3.vocab : 0)) * 4096;
      for (let k = 0; k < bM.length; k++) {
        const bw = (g.moe * bM[k] + g.dense * bD[k]) * 4096 * IF + (k === bM.length - 1 ? vocab : 0);
        if (get(`A${k + 2}`) !== bw) worst = `${JSON.stringify(S)}: A${k + 2} ${get(`A${k + 2}`)} ≠ ${bw}`;
      }
    }
  check('cells ≡ independent shard math EXACTLY, sub-cells included (96 configs, ===)', worst === null, worst ?? '');
}

function check(name, ok, detail) {
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${name}  (${detail})`);
  if (!ok) process.exitCode = 1;
}
