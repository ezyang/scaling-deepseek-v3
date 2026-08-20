# scaling-dsv3 — a profile estimator for DeepSeek-V3 training

Simulates what a Perfetto profile of a DeepSeek-V3 pretraining step *should*
look like under a given set of assumptions, as a pedagogical tool: the model
starts at the roofline and adds refinements one at a time (kernel costs → 1F1B
pipeline bubbles → expert-parallel all-to-all → FSDP/pipeline comm → launch
overhead → jitter/GC stragglers), each ordered by impact on step walltime.
The simulated traces render inline in a small Perfetto-style viewer and export
to real Perfetto, and can be compared category-by-category against real
PyTorch profiler traces.

Plain HTML/JS, zero dependencies, no build step.

## Run

```sh
python3 -m http.server 8000        # any static server; ES modules need http
open http://localhost:8000         # the essay, with live embedded traces
```

CLI (Node ≥ 18):

```sh
node scripts/sanity.mjs            # step time per refinement level + invariant checks
node scripts/export.mjs --level 4 --out trace.json            # Chrome trace JSON → ui.perfetto.dev
node scripts/compare.mjs real.json --config '{"dtype":"mxfp8","microbatches":120}'
```

## Layout

| file | what |
|---|---|
| `src/model.js` | DSv3 architecture (MLA, 256-expert MoE, 61 layers), per-op FLOP/byte counts, hardware specs (H800/H100/GB200/GB300 incl. `memGB`), BF16 vs MXFP8 rates, comm formulas. **All assumptions live here.** |
| `src/blockgraph.js` | The transformer block as an explicit op DAG: per-op save/recompute marks, backward-need rules (matmul/norm/swiglu need inputs, flash needs its own output, adds need nothing, combine needs expert-outs+probs), transitive replay closure. Memory = saved∧needed outputs; recompute = +1× fwd per replayed op (charged in the trace's backward). Presets: none/dsv3/selective/full; per-op overrides via `saved: {x1: false}`. |
| `src/memory.js` | High-watermark memory model (1F1B steady-state moment; fwd temporaries out of scope): per-GPU/per-stage weights, grads, optimizer (ZeRO-0/1/3), activations broken down mla/moe/residual/logits under a recompute policy, working buffers. Precision is **per matmul**: named recipes (`bf16`, `dsv3-fp8`, `nv-mxfp8`) + `matmuls` overrides; a stash is stored at the dtype of the matmul that reads it. Every sim's `stats.mem` carries the verdict. |
| `src/sim.js` | Refinement levels; op-graph generation (1F1B schedule, a2a, ZeRO-aware DP comm — FSDP AG/RS or ZeRO-0/1 all-reduce — p2p, GC); event-driven scheduler with cross-rank collective barriers; slice building at `phase`/`layer`/`op` granularity. |
| `src/trace.js` | Chrome Trace Event export, download, `ui.perfetto.dev` postMessage hand-off. |
| `src/viewer.js` | Canvas timeline viewer (`TraceViewer`) + `<dsv3-trace>` custom element. Perfetto keys: W/S/A/D, F, M, 0, arrows; wheel/drag; hover/click. |
| `src/compare.js` | Category-level summarization of any Chrome trace (real Kineto or simulated) + diff. |
| `index.html` + `src/essay.js` | The progressive-disclosure essay with embedded widgets, impact table, playground. |

## Embedding a trace

```html
<script type="module" src="./src/viewer.js"></script>
<dsv3-trace level="3" config='{"hardware":"gb200","microbatches":8}' height="300"></dsv3-trace>
```

Any `defaultConfig` key goes in `config`, including `refinements` to toggle
individual effects independently of the cumulative `level` (e.g.
`"refinements":{"jitter":true,"epComm":false}`). The element exposes `.result`
(`{cfg, trace, stats}`) and fires a `sim` event after each simulation.

Widgets link through `for="id1 id2"`: `<dsv3-layer>` (top-down SVG block
schematic, per-matmul dtype selects — the single source of truth for
precision) and `<dsv3-controls>` (knob strip; `toggles` adds per-refinement
checkboxes) patch the configs of any listed `<dsv3-memory>`/`<dsv3-trace>`,
so one set of choices drives memory and profile together. Memory rows may
target different hardware — each row draws its own capacity line:

```html
<dsv3-layer for="mem trace" recipe="nv-mxfp8"></dsv3-layer>
<dsv3-controls for="mem trace" fields="hardware,gpus,pp,ep,zero,recompute,mbs,seqLen"></dsv3-controls>
<dsv3-memory id="mem" configs='[
  {"label":"PP16 · EP16 · ZeRO-1","hardware":"gb300","pp":16,"ep":16,"zero":1,"recipe":"nv-mxfp8"},
  {"label":"same on H800","hardware":"h800","pp":16,"ep":16,"zero":1,"recipe":"nv-mxfp8"}
]'></dsv3-memory>
<dsv3-trace id="trace" level="4"></dsv3-trace>
```

Cluster size is `gpus` (default 256 — the scale we have real traces at);
`dp = gpus/pp` is derived, `ep` is clamped to dp. An explicit `dp` override
still wins for back-compat.

## Design notes / caveats

- **Representative ranks**: within a pipeline stage, DP/EP peers are symmetric
  until jitter is enabled, so we simulate `dpRanksToSim` ranks per stage
  (default 1) rather than all 2048. Collectives barrier across the simulated
  peers, so one rank's GC pause visibly convoys the others.
- **Categories** (gemm / attn / vector / a2a / fsdp / p2p / optimizer / stall)
  are the comparison currency between sim and real traces; `compare.mjs`
  regex-maps real kernel names onto them and computes per-GPU self-time within
  the median `ProfilerStep` window. Use `granularity: "op"` for exact category
  attribution (merged layer slices take their dominant category).
- **Not modeled yet**: DualPipe-style a2a/compute overlap (the fix for the
  biggest correction term), output-form RMSNorm backward (the one-residual
  trick), MTP, interleaved PP, congestion, critical-path stitching across
  all ranks.
- **Planned restructure**: refinements re-ordered by impact on the headline
  number (relative vs. absolute TBD); one essay per hardware config, each
  working through the parallelism search — every refinement should *rule
  configs out* (the GB300 memory section is the prototype of that style).
- Hardware numbers (`src/model.js`) are approximate public specs plus
  efficiency assumptions; they are inputs, not claims. H800 = H100 compute
  with NVLink cut to 400 GB/s; MXFP8 on Hopper pays a recipe tax, on
  Blackwell it is native.
- Calibration anchor: DeepSeek reported 2.788M H800-hours for 14.8T tokens
  ≈ 1475 tok/s/GPU; this sim at full refinement (mxfp8, m=120, no DualPipe
  overlap) lands ~1235 tok/s/GPU — appropriately *below* the real number,
  since the real run overlaps a2a and we deliberately don't (yet).
