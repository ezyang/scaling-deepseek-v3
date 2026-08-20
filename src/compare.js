// Compare a (real or simulated) Chrome trace against a simulated one at the
// category level: per-GPU busy time by kernel category, plus step walltime.
// Real traces = PyTorch profiler / Kineto exports (cat: 'kernel' on GPU tids).

const SIM_CATS = ['gemm', 'attn', 'vector', 'a2a', 'fsdp', 'p2p', 'optimizer', 'stall'];

// name -> category rules for real kernel names (order matters)
export const CATEGORY_RULES = [
  ['a2a', /all[_ ]?to[_ ]?all|a2a|alltoall|dispatch|combine|moe.*(send|recv)/i],
  ['fsdp', /all[_ ]?gather|reduce[_ ]?scatter|all[_ ]?reduce/i],
  ['p2p', /sendrecv|send|recv|p2p|memcpy/i],
  ['attn', /flash|fmha|attention|attn|softmax.*dropout/i],
  ['optimizer', /adam|optimizer|multi_tensor|foreach/i],
  ['gemm', /gemm|matmul|cutlass|nvjet|wgrad|dgrad|grouped|_mm_|^nchw|s\d{4,}|tf32|f8f8|bf16/i],
  ['vector', /elementwise|vectorized|norm|softmax|rope|rotary|cast|copy|fill|reduce|triton|cat_|index|embedding|cross_entropy/i],
];

export function categorize(e) {
  if (SIM_CATS.includes(e.cat)) return e.cat;
  for (const [cat, re] of CATEGORY_RULES) if (re.test(e.name)) return cat;
  return 'other';
}

export function summarize(chromeTrace, opts = {}) {
  const evs = (chromeTrace.traceEvents ?? chromeTrace).filter(e => e.ph === 'X' && e.dur > 0);
  const isReal = evs.some(e => e.cat === 'kernel');
  const gpu = evs.filter(e => isReal
    ? ['kernel', 'gpu_memcpy', 'gpu_memset', 'gpu_user_annotation'].includes(e.cat)
    : SIM_CATS.includes(e.cat));

  // step window: prefer ProfilerStep annotations (real traces), else full span
  let w0 = Math.min(...gpu.map(e => e.ts)), w1 = Math.max(...gpu.map(e => e.ts + e.dur));
  const steps = evs.filter(e => /ProfilerStep\s*#?\d+/.test(e.name)).sort((a, b) => a.dur - b.dur);
  if (steps.length) {
    const step = steps[Math.floor(steps.length / 2)]; // median-duration step
    w0 = step.ts; w1 = step.ts + step.dur;
  }
  const win = gpu.filter(e => e.ts < w1 && e.ts + e.dur > w0);

  // self time per (pid,tid), attributed by category
  const byTid = new Map();
  for (const e of win) {
    const k = `${e.pid}|${e.tid}`;
    if (!byTid.has(k)) byTid.set(k, []);
    byTid.get(k).push(e);
  }
  const byCat = {};
  const pids = new Set(win.map(e => e.pid));
  for (const evs of byTid.values()) {
    evs.sort((a, b) => a.ts - b.ts || b.dur - a.dur);
    const stack = [];
    const pop = () => { const s = stack.pop(); byCat[s.cat] = (byCat[s.cat] ?? 0) + Math.max(0, s.dur - s.child); };
    for (const e of evs) {
      while (stack.length && stack.at(-1).end <= e.ts + 1e-6) pop();
      const clipped = Math.min(e.ts + e.dur, w1) - Math.max(e.ts, w0);
      if (stack.length) stack.at(-1).child += clipped;
      stack.push({ end: e.ts + e.dur, dur: clipped, child: 0, cat: categorize(e) });
    }
    while (stack.length) pop();
  }
  const nGpu = Math.max(1, pids.size);
  for (const k of Object.keys(byCat)) byCat[k] /= nGpu; // per-GPU average
  const busyUs = Object.values(byCat).reduce((a, b) => a + b, 0);
  return {
    label: opts.label, isReal, nGpu,
    stepUs: w1 - w0, busyUs, busyFrac: busyUs / (w1 - w0), byCat,
    usedProfilerSteps: steps.length > 0,
  };
}

export function diff(real, sim) {
  const cats = [...new Set([...Object.keys(real.byCat), ...Object.keys(sim.byCat)])].sort(
    (a, b) => (sim.byCat[b] ?? 0) + (real.byCat[b] ?? 0) - (sim.byCat[a] ?? 0) - (real.byCat[a] ?? 0));
  const row = (name, r, s) => ({
    what: name,
    [real.label ?? 'real']: (r / 1e3).toFixed(1) + ' ms',
    [sim.label ?? 'sim']: (s / 1e3).toFixed(1) + ' ms',
    'sim/real': r > 0 ? (s / r).toFixed(2) + 'x' : '—',
  });
  return [
    row('step walltime', real.stepUs, sim.stepUs),
    row('gpu busy (per GPU)', real.busyUs, sim.busyUs),
    ...cats.map(c => row('  ' + c, real.byCat[c] ?? 0, sim.byCat[c] ?? 0)),
  ];
}
