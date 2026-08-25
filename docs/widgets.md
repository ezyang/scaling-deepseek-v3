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
| `lens` | view | absent · `params` | display lens: `params` hides intermediates/dims/aux and unparenthesizes the parameter counts (spacing unchanged) |
| `tabs` | view | boolean | dense/MoE flip tabs (with per-block FFN tallies) above the FFN column, fused into a scoped enclosure |
| `nocaption` | view | boolean | suppress the foot caption (the page explains the diagram itself) |
| `recipe` | state | `nv-mxfp8` · `bf16` · `dsv3-fp8` | per-matmul dtype preset |
| `recompute` | state | `dsv3` · `none` · `attn-replay` · `selective` · `full` | save/recompute marks preset |
| `kind` | state | `moe` · `dense` | which FFN column variant (flip-stable layout; the MLA column is shared) |
| `transposed` | state | boolean | Hopper fp8ᵀ dual-orientation stashes |
| `cumulative` | state | boolean | parameter parentheticals ×(selected kind's block count), always multiplied out; tabs + enclosure hide |
| `xlayers`, `xinflight` | state | numbers (61, 1) | combined-view multipliers (× layers × in-flight × 4096 tokens) |
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
| *(forwarded)* | | `controls recipe recompute detail transposed for nocaption kind xlayers xinflight lens` pass through to the inner layer |

## `<dsv3-anatomy-plan layer=…>` — the vertical margin plan
Bound to a layer: clicking a block kind flips it; re-syncs via the layer's
`recipe` event; follows `activeView`; supports `highlightOps`.

## `<dsv3-param-tally layer=…>` — the parameter count, computed from the diagram

| attribute | meaning |
|---|---|
| `layer` | the diagram whose cells rows/terms highlight (hover previews, click pins; hidden-kind pins flip the diagram; hidden-kind hovers show only kind-shared cells) |
| `compact` | narrow two-column margin form with the fixed equation slot |
| `mode` | `total` (default) · `active` — initial toggle position |

## Other elements (unchanged conventions)
- `<dsv3-stack arch dense choices for>` — layer-structure strip; the dense-count
  toggle patches linked widgets' `arch`.
- `<dsv3-controls for layer fields values bundles toggles>` — config knobs;
  `layer` links the block diagram for bundle presets.
- `<dsv3-memory configs>` · `<dsv3-trace level height title config>`.
