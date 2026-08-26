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
| `optim` / `consolidated` | view | boolean | param-bytes only: stack byte-component squares under the blue weight squares, in the memory-bars stacked order with the memory-bars segment colors (`BYTE_COMPS`: weights #2a78d6 2 B/param · fp32 gradients #eb6834 4 B/param · optimizer states #1baf7a 8 B/param). `optim` = weights + optimizer; `consolidated` = all three + amber saved-activation chips ON THE WIRES (each saved tensor: name + bytes × 4096 tokens + squares, mostly hollow traces; a text-only total line at the bottom — the squares live at the chips, no double count; pair with `recipe="bf16"` so stashes price at 2 B). Under ×N the chips scale too and narrow fork chips go two-line. Same global unit, so the ratios are the picture; pinned per block. The legend entries are visibility checkboxes (URL state `showWeights`/`showGrads`/`showOptim`/`showActs`): numbers always total exactly the squares shown (nothing visible = no number), and toggling animates per-block-tween style — the color's squares pour in/out and boxes/band reflow smoothly (numbers snap) |
| `local` | view | boolean | param-bytes only, implies `consolidated`: what ONE GPU holds under the fiat parallelism (`LOCAL_PAR`: 2048 GPUs) with mini-head knobs (URL state): EP (off/32/64), PP degree (powers of two, contiguous floor split of the 61 layers), PP stage (each option names its layer assignment; empty PP64 stages render true zeros), and a ZeRO-1 checkbox. Expert weights divide by EP; with ZeRO-1 on, optimizer states shard over each parameter's replication group (dense /DP=world/pp, expert /expert-DP=world/pp/EP — why expert green squares survive; EP off makes the classes coincide); the kind and ×N follow the stage (0 = embedding + dense blocks; last = + final norm/lm head; the plan doubles as the stage map, gating those rows); `cumulative` toggles one block ↔ the stage total (tweened; tabs stay hidden in both). Activations = one microbatch per stage layer (schedule in-flight deferred). Fixed 448 MiB unit |
| `strips` | view | absent · `absolute` | byte-strip spacing profile (param-bytes lens): default renormalizes per view (largest op = one row, unit legend shows the rescale); `absolute` fixes ONE unit and reserves worst-case (cumulative) rows, so toggling ×N grows strips with zero rescale/reflow — costs vertical space; dense/MoE flips may reflow. Hollow square = nonzero but sub-square |
| `for` | binding | ids | linked widgets receiving recipe/marks patches |

Runtime-only properties (driven by other widgets, not attributes):
`activeView` (the tally's active/token toggle: fired-expert counts),
`highlightOps(ids)` (`null` clears, `[]` fades everything, else the listed
`data-op` cells stay full-contrast and the rest fades).

## `<dsv3-anatomy>` — the shipped composition (margin plan + expansion cone + tabbed block)

| attribute | kind | meaning |
|---|---|---|
| `layer` | binding | id given to the inner `<dsv3-layer tabs scope="block">` (URL state, `dsv3-controls layer=` links, page scripts all address it) |
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

## Other elements (unchanged conventions)
- `<dsv3-stack arch dense choices for>` — layer-structure strip; the dense-count
  toggle patches linked widgets' `arch`.
- `<dsv3-controls for layer fields values bundles toggles>` — config knobs;
  `layer` links the block diagram for bundle presets.
- `<dsv3-memory configs>` · `<dsv3-trace level height title config>`.
