# Widget attribute API

The custom elements and their attributes, grouped by concern. Conventions:
**binding** attributes are ids of other elements (`layer` = the `<dsv3-layer>`
an element is bound to; `for` = space-separated ids this element pushes config
patches to); **model state** is user-mutable and persisted in the URL hash
(keyed by the element's `id`); **view composition** is authored per instance
and never changes at runtime. Every widget instance on a real page should
carry a measured `style="min-height:…px"` placeholder (scroll restoration);
`node scripts/diagramlint.mjs` reports the value to paste and flags drift.

## `<dsv3-layer>` — the transformer-block diagram

| attribute | kind | values (default first) | meaning |
|---|---|---|---|
| `scope` | view | `model` · `block` · `mla` · `ffn` | how much to draw: block + ×61/head row · the block alone · one column (with hand-off labels to the block wiring) |
| `controls` | view | `full` · `static` · `marks` · `dtype` | progressive-disclosure tier: which knobs render (static = structure only: no quantities, marks, or tooltips) |
| `detail` | view | boolean | also draw the elided kernels (latent norms, RoPE, router internals, shared expert, top-k rail) |
| `lens` | view | absent · `params` · `param-bytes` | display lens: `params` hides intermediates/dims/aux and unparenthesizes the parameter counts; `param-bytes` additionally restates every number as bf16 bytes, always multiplied out (the sizes toggle is hidden — factored ×256 byte chains pull no weight), adds blue weight-size strips (largest op per block = one row of 32; a square = largestOp/32 · 2 B = 448 MiB), and wears a static bf16 tag on each weight-bearing GEMM. Boxes reserve their in-box strip band only in tiers/lenses that can fill it (dtype tiers, the bytes lens) — static/params boxes are compact |
| `tabs` | view | boolean | dense/MoE flip tabs (with per-block FFN tallies) above the FFN column, fused into a scoped enclosure |
| `nocaption` | view | boolean | suppress the foot caption (the page explains the diagram itself) |
| `recipe` | state | `nv-mxfp8` · `bf16` · `dsv3-fp8` | per-matmul dtype preset |
| `recompute` | state | `dsv3` · `none` · `attn-replay` · `selective` · `full` | save/recompute marks preset |
| `kind` | state | `moe` · `dense` | which FFN column variant (flip-stable layout; the MLA column is shared) |
| `transposed` | state | boolean | Hopper fp8ᵀ dual-orientation stashes |
| `cumulative` | state | boolean | parameter parentheticals ×(selected kind's block count), always multiplied out; tabs + enclosure hide. No double-multiplication cues: whenever a number is already multiplied out, the ×N leaves its label (the plan's ×3/×58 tags in cumulative; the experts group's ×256 under multiplied sizes) |
| `xlayers`, `xinflight` | state | numbers (61, 1) | combined-view multipliers (× layers × in-flight × 4096 tokens) |
| `optim` / `consolidated` | view | boolean | param-bytes only: stack byte-component squares under the blue weight squares, in the memory-bars stacked order with the memory-bars segment colors (`BYTE_COMPS`: weights #2a78d6 2 B/param · fp32 gradients #eb6834 4 B/param · optimizer states #1baf7a 8 B/param). `optim` = weights + optimizer; `consolidated` = all three + amber saved-activation chips ON THE WIRES (each saved tensor: name + bytes × 4096 tokens + squares, mostly hollow traces; no total line — the squares live at the chips, no double count; pair with `recipe="bf16"` so stashes price at 2 B). Under ×N the chips scale too and narrow fork chips go two-line. Same global unit, so the ratios are the picture; pinned per block. Component visibility (URL state `showWeights`/`showGrads`/`showOptim`/`showActs`) toggles via the legend (head checkboxes, or margin rows under anatomy): numbers always total exactly the squares shown (nothing visible = no number), and toggling animates per-block-tween style — the color's squares pour in/out and boxes reflow smoothly (numbers snap) |
| `local` | view | boolean | param-bytes only, implies `consolidated`: what ONE GPU holds under the fiat parallelism (`LOCAL_PAR`: 2048 GPUs) with mini-head knobs (URL state): EP and PP as ± steppers over powers of two 1–64 (contiguous floor split of the 61 layers), PP stage (each option names its layer assignment, tags the peak, defaults to it; empty PP64 stages render true zeros; PP changes jump to the new peak; GPUs stepper 128–16384), and a segmented ZeRO-(off|1|2|3) level picker (zthresh per component). EVERY knob change pours the squares between the old and new configuration (the generic `_vtween` factor lerp — same feel as per block ↔ ×N; numbers snap; kind-flipping changes snap). Amber wire-chip squares wrap at 16/row (8 in the fork columns) and the wire gaps grow with their rows. Expert weights divide by EP; with ZeRO-1 on, optimizer states shard over each parameter's replication group (dense /DP=world/pp, expert /expert-DP=DP/EP; the EP share and the smaller group cancel, so per-rank optimizer bytes are a uniform 8 B/DP per global param, EP-independent — matches memory.js); the kind and ×N follow the stage (0 = embedding + dense blocks; last = + final norm/lm head; the plan doubles as the stage map, gating those rows); `cumulative` toggles one block ↔ the stage total (tweened; tabs stay hidden in both). Activations follow the schedule, decomposed into three knobs: admission (`sched`, 1F1B/×1 mb), VPP degree (`vpp`, stepper 1/2/4 — chunks per rank over a VPP·PP-deep virtual chain), and chunk placement (`fold`: wrap = Megatron interleaving, virtual stage v on rank v mod PP; reflect/V = the zigzag, so even VPP puts embed AND head on rank 0). DualPipeV ≡ VPP2 + reflect. The law (`inflightOf`): each virtual stage v holds D−v chunk-stashes (1F1B admission, D = VPP·PP); a rank's residency = the sum over its `vstagesOf` ÷ VPP. VPP1 → the pp−s staircase; VPP2+reflect → a UNIFORM PP+½ on every rank (8.5 at PP8, the reference-spreadsheet convention, vs the DSv3 paper's coarse PP+1); wrap → PP(V+1)/2 − s, concentrated at rank 0. `ppStage(s, pp, vpp, fold)` returns the hosted segments plus `emb`/`head` flags (all vocab attribution flows through them), multiplying the amber chips/gaps and included in the peak-stage pick. Knobs carry `data-knob` names (gpus/pp/stage/sched/vpp/fold/ep/zero) for tests. Head layout: parallelism row, then the fit chart, then the misc row. Fit chart: UNSTACKED per-component log₂ bars + a grey total row on a FIXED axis (256 MiB–16 TiB; gridlines = ×2; the 80 GiB capacity line labels itself ON TOP, leaving the bottom axis to the power-of-two labels). The row labels ARE the legend (name + absolute bytes; whole-gutter hitbox; mousedown SOLOS the component — the filter that matters — and soloing again restores all; dimmed = off). Under a solo the TOTAL row keeps its full extent as a stacked bar: grey =other= base + the highlighted component on top, so the colored width IS the factor available from optimizing just that component and bar ends carry the ×N/÷N factor vs the 80 GiB boundary; a measuring cursor armed only inside the bars: click = point (value + vs-capacity factor), drag = span (any span on a log axis is a factor: ×N with endpoints), click it to clear; the SAVE cluster (top right, visually apart): 'save' locks in the config (re-save to lock in changes), 'reset' returns TO the save, 'reset all' is factory (PP1/EP1/ZeRO-off/bf16/none — the whole-model one-GPU view). A save renders as dotted GHOST bars (main rows and sub-rows; value labels always ride the live bar end) and annotates bar ends + the DIAGRAM numbers with each class's exact change (each sharding class moves uniformly). CHANGE factors wear alert badges: red ▲×N grew / bold ▼×N shrank (direction in the arrow, magnitude always ≥1; red is unused by the components). Soloing a param component ACCORDIONS its breakdown sub-rows (experts / non-expert / vocab) open beneath it (animated reflow, per author ruling), each with its own save badge + ghost, so e.g. only the experts sub-bar moves under an EP change. Sub-rows are clickable: selecting a part (`partSel` — NOT `part`, which collides with the native Element.part API) filters the DIAGRAM to that part's ops and gives the soloed row the stacked grey+color treatment, one level down from the total's. Stash knobs (precision/recompute/fp8ᵀ/per-op buttons) animate through the same generic tween (the snapshot carries the pre-change analysis so chips and the acts bar lerp bytes). With recompute none, x0 loses its 🔒 (a checkpoint anchor means something only when a replay exists) — the pinned-label line is height-reserved so pinning never reflows; bar motion interpolates geometrically (lerp the exponent). Local instances have NO margin legend (the chart carries it); optim/consolidated anatomy instances keep the margin rows (`marginlegend` delegation; standalone instances keep head checkboxes). Strip squares FLOOR per component; one hollow trace only where an op would otherwise show nothing (largest remainder's color). Half boxes wrap strips at 21/row. Fixed 448 MiB unit |
| `strips` | view | absent · `absolute` | byte-strip spacing profile (param-bytes lens): default renormalizes per view (largest op = one row, unit legend shows the rescale); `absolute` fixes ONE unit and reserves worst-case (cumulative) rows, so toggling ×N grows strips with zero rescale/reflow — costs vertical space; dense/MoE flips may reflow. Hollow square = nonzero but sub-square |
| `for` | binding | ids | linked widgets receiving recipe/marks patches |

Runtime-only properties (driven by other widgets, not attributes):
`activeView` (the tally's active/token toggle: fired-expert counts),
`highlightOps(ids)` (`null` clears, `[]` fades everything, else the listed
`data-op` cells stay full-contrast and the rest fades).

## `<dsv3-anatomy>` — the shipped composition (margin plan + expansion cone + tabbed block)

| attribute | kind | meaning |
|---|---|---|
| `layer` | binding | id given to the inner `<dsv3-layer tabs scope="block">` (URL state and page scripts address it) |
| `tally` | view | mount the compact parameter tally in the margin below the plan |
| *(forwarded)* | | `controls recipe recompute detail transposed for nocaption kind xlayers xinflight lens strips optim consolidated local cumulative` pass through to the inner layer |

Narrow viewports (≤860px): the anatomy grid stacks (plan above the diagram,
expansion cone hidden) and diagrams stop scaling down — they render at natural
size and scroll horizontally inside their container instead.
`scripts/interact.mjs` takes `--width N` to test this.

## `<dsv3-anatomy-plan layer=…>` — the vertical margin plan
Bound to a layer: clicking a block kind flips it; re-syncs via the layer's
`recipe` event; follows `activeView` and the `param-bytes` lens (byte values +
weight strips on the embedding / final norm / lm head — NEVER on the block
boxes, whose bytes are shown expanded on the right: no byte is ever
double-counted anywhere in the figure); supports `highlightOps`.

## `<dsv3-param-tally layer=…>` — the parameter count, computed from the diagram

| attribute | meaning |
|---|---|
| `layer` | the diagram whose cells rows/terms highlight (hover previews, click pins; hidden-kind pins flip the diagram; hidden-kind hovers show only kind-shared cells) |
| `compact` | narrow two-column margin form with the fixed equation slot |
| `mode` | `total` (default) · `active` — initial toggle position |
| `units` | absent · `bytes` — bf16 memory framing: values in binary bytes, the total/active toggle hidden (activation doesn't change resident bytes). `<dsv3-anatomy lens="param-bytes" tally>` sets this automatically |

## `<dsv3-pp-schedule layer=…>` — the pipeline-schedule strip

One row per PP stage, time flowing right; F cells one slot, B cells two
(backward ≈ 2× forward FLOPs), every cell numbered with its microbatch.
Colors follow the byte language (F stashes activations = amber; B consumes
them into gradients = orange). Bound to a local-lens layer it follows the
layer's PP / schedule / stage (the selected stage's row is tinted) AND wears
its own replica of the pipeline knob group — PP stepper, stage select, and the
sched/VPP/fold knobs — whose changes drive the layer (`layer.setLocal`), so
either widget's controls move both, plus a strip-local `mb` knob for how many
microbatches to DRAW (default 64, a real step's worth; 'auto' = depth+4, just
enough to reach steady state — the memory model needs no m, its law assumes
m ≥ pp; smaller m shows a pipeline that never fills). ×1 mb draws the
single-microbatch wave. VPP>1 draws 1F1B admission over the VPP·pp-deep
virtual chain with `vstagesOf` placement (wrap or reflect), each rank
interleaving its chunk queues greedily one-op-at-a-time (earliest-ready;
ties backward-first, then the deeper chunk — DualPipeV's official overlapped
F+B blocks are NOT modeled, disclosed in the header); later passes wear
deeper shades. The drawn peak residency reproduces the modeled law exactly
(2pp+1 half-rank chunks under VPP2+reflect), and cells carry data-v/t0/t1 so
tests count it. Below the schedule, an IN-FLIGHT section (same svg, so the
horizontal scroll is shared) shows the selected stage's stashes as lifetime
lanes — the F cell that stashes a microbatch, an amber tail while it's held,
and the B cell that frees it (data-stash) — so the braid's thickness IS the
in-flight count; a dashed line marks the modeled peak, labeled in
microbatches (text[data-peak]): the number the memory bars charge. Cell
microbatch numbers shrink to a 6px font when double digits meet a 1-slot
cell (no wide convention exists; papers widen cells instead) and hide past
PP32 (cell budget).
PP/sched changes tween the strip's height (12-frame ease-out, deterministic).
No state of its own; unbound instances read `pp`/`sched`/`stage` attributes.

## Other elements (unchanged conventions)
- `<dsv3-trace level height title config>` — the canvas trace viewer over the
  simulator (not yet on a published page; the timing posts' widget).
