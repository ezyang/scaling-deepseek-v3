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
| `controls` | view | `full` · `static` · `marks` · `dtype` | progressive-disclosure tier: which knobs render (static = structure only: no quantities, marks, or tooltips). The `marks`/`dtype` tiers wear the HOUSE knob row (pargrp/stp segments, like the local head) instead of the legacy select head: one preset segment (`data-knob="recompute"` over `RECOMPUTE_PRESETS`, or `data-knob="recipe"` over `RECIPES` + TWO stash-format checkboxes — e4m3ᵀ dual stash and 'E5M6 attn-out stash' — plus the SwiGLU pill's 'in:' button (`data-dt="swiglu_in"`: the SwiGLU input's save format is FREE-FLOATING — its only backward reader is the elementwise SwiGLU backward, no GEMM forces it; the paper caches it fp8, §3.3.3; it counts toward recipe recognition like every stash channel) (writes `mm.o_proj` e5m6 ⇄ bf16; greys out when the recipe pins e4m3/mxfp8 there). BOTH checkboxes count toward recipe recognition: each recipe is a full stash-policy bundle (`RECIPE_T` carries the canonical ᵀ state — all-fp8 ships ᵀ on, dsv3-fp8 off since DeepSeek requantizes; E5M6 rides mm) — picking a chip sets them, flipping one drops to custom; the dtype tier ALSO wears the recompute segment in the marks tier's slot — LIVE but PRESETS-ONLY (`noCustom`: no custom chip, since no per-op marks render below the marks tier a custom policy can't arise; the `recompute` attr seeds it)) plus an always-reserved disabled `custom` chip that lights up when per-op edits leave every preset (reserved so a custom state never reflows the row), then the `ctx` readout, sizes button and reset — NO elided-kernels toggle: these tiers are ALWAYS the detail view (one canonical diagram), and the KIND is pinned to MoE (the section analyzes the peak rank, which is all-MoE): the dense/MoE tab flaps are gone; the enclosure stays, wearing a static 'MoE FFN ×58' label and the REGION TOGGLE — a tristate ↻ none / ↻ all / mixed that sets every MoE-FFN mark at once (router, dispatch, gate/up, swiglu, ffn down, combine), where 'mixed' both displays an in-between state and REMEMBERS the last one (the custom-chip pattern). Marks are LITERAL, torch_remat-style: a ↻ mark replays whether or not any backward consumer demands the output — recompute-everything costs exactly 1× fwd, and a useless mark can honestly increase the stash (its inputs must be available); demand still decides what is stashed. A ↻ whose output nothing reads wears a ⚠ (tooltip: pointless recompute — torch_remat does what you said; a demand-driven planner would skip it). Even 'full' carries one ⚠: the routed+shared sum at the replay tail is read by nobody. EVERY op shows its recompute state as a BOOLEAN ↻ button (live ↻ = replayed; struck ↻ = kept — whether a kept output is STASHED is demand's call, which is why there is no 💾 face): the MLA latent RMSNorms' micro pills carry real boolean ↻ buttons (they are graph nodes); the shared-expert boxes and the ungated-SwiGLU pill carry MIRRORED buttons (the shared expert lives inside the grouped node — one mark, two buttons, both re-render); RoPE is a REAL node pair (rope_q/rope_kv, zero-byte marks: attention reads the ROTATED q/k, the same size as pre-RoPE, so flipping moves the stash across the rotation without changing a byte — every preset except none replays them; in the quant tiers the pre-RoPE outputs are chips too, so the ledger is visible wherever it lands). The tally ends with the honesty line: the bandwidth-bound vector work the current policy re-runs unmetered (dynamic per marks). These tiers also DROP the prose caption (the page's own text explains the diagram); their foot is just the full-width tally svg. Quant tiers price compute as PICKETS (2×5 thin rects at 3px pitch, dtype-colored): a tally of time quanta — ONE fixed unit of TIME at H100 peak, ≈ 41 µs per 4096-token microbatch (= 10 MFLOP/token at bf16's 989 TFLOP/s) — so dtype flips visibly grow/shrink the COUNT instead of renormalizing the scale (the fp8 flavors pack 2×; fp32 runs on CUDA cores at 67 TFLOP/s, ≈15× bf16 time per FLOP — the pinned router is a visible brick run, not a fig leaf); sub-picket GEMMs (router, kv down-proj half) wear the hollow trace, the last picket is a partial-width sliver (area exact), a run that outgrows its box wraps and grows rows. Norms/SwiGLU keep an unpriced hollow-dashed fig-leaf (bandwidth-bound; the model deliberately prices only the quadratic GEMM terms — no epilogue-fusion modeling, so no false bandwidth precision). Byte stashes stay amber SQUARES (4 KiB/token each), single-line ALWAYS; in dtype-bearing tiers a saved sub-bf16 stash wears a DASHED PHANTOM tail out to its bf16-equivalent edge (elems × 2 B — dtype-independent, so dtype flips pour the solid fill inside a fixed dashed silhouette; a dual fp8ᵀ stash that exceeds bf16 gets no phantom), and the pure dtype tier DROPS the ↻ hollow counterfactual grids (the AC story's device); the shared-expert boxes carry compact dtype MIRRORS in the mark slot (one matmul, two buttons — the params parenthetical yields its spot, the count stays in the tooltip); ctx'd (fixed-parallelism) instances drop the ×61/×58 enclosure multipliers (the readout already sums the layers — no double-claim) (wrapped grids crossed the wire routes; the reserved chip band is fixed-height); narrow fork/shared-column chips (`ov.short`) go two-line — name over value, grid beside the value line — the CONS chips' pattern. The tally = the readout header (`stashed for backward: N KiB/token·layer × layers × in-flight × 4096 tokens = X GiB`) + the CONSOLIDATED STASH as a SOLID LINEAR amber bar over a unit RULER (minor tick = 1 GiB at 6px, major = 8 GiB, red 80 GiB card line; ghost = dashed amber OVERHANG only, with an exact ▲/▼×N badge; its anchor baselines THIS tier's lever at do-nothing, HOLDING the other lever: the dtype tier ghosts all-bf16 at the CURRENT recompute policy (it moves with the policy segment — the overhang isolates precision's win), other tiers ghost the untreated anchor — recompute none · all-bf16 · no fp8ᵀ; the anchor edge lerps like the bar) + fwd/bwd(2×)/replay picket ribbons (dtype-sorted runs, per-op counts Bresenham'd over the exact cumulative; the fwd ribbon = the boxes' pickets laid end to end; foreign-currency costs ride the ribbons as PILLS, by the fusion rule (a pill only where there is NO producer to fuse into): violet a2a comm on the replay ribbon; bronze HBM-traffic on bwd when ᵀ is OFF — the backward requant round trips (≈2 B/elem over the would-dual set, µs at HBM peak in the tooltip; a COLD stash, nothing to fuse into). ᵀ ON pays no pill: the second orientation rides the forward quantize epilogue — fusable, same unpriced class as every stash write; its cost is the GiB on the bar. Pill presence EASES with the toggle tween (scales about its anchor — bar physics, no pop)) over their own TIME ruler (ticks in ms per mb·layer at H100 peak; minor = 0.5 ms, labels every 1 ms). Per-op dtype buttons TOGGLE bf16 ⇄ the instance's fp8 flavor — `e4m3` (Hopper tile-scaled) on the dsv3 recipes, `mxfp8` on the Blackwell one; same 1+1/32 B/elem, distinct keys so labels carry provenance, both pink — dtype colors are the warm magenta PRECISION family, clear of the byte-component blue/orange/green; recompute pickets/ribbons reuse the forward dtype color LIGHTENED (fill-opacity 0.55) rather than a foreign hue (no fp32 stop — the article anchors on bf16) — EXCEPT attn-out and the router, whose tags are pinned readouts: the router follows the recipe (production runs it fp32) and attn-out shows its COMPUTE dtype (the E5M6 save format is the head checkbox). EVERY stash knob (preset segments, per-op 💾/↻ and dtype buttons, fp8ᵀ) animates through the standard 12-frame tween (`_tweenQuant`): picket counts and colors lerp (prev dtypes captured in the snapshot), ribbons pour (replay membership grows from/drains to zero), saved-tensor chips dissolve between their text forms while their squares pour between old and new bytes, aux artifacts (rstd/lse) dissolve too, and the stash bar's width lerps (grammar rule 9); numbers snap |
| `detail` | view | boolean | also draw the elided kernels (latent norms, RoPE, router internals, shared expert, top-k rail) |
| `lens` | view | absent · `params` · `param-bytes` | display lens: `params` hides intermediates/dims/aux and unparenthesizes the parameter counts; `param-bytes` additionally restates every number as bf16 bytes, always multiplied out (the sizes toggle is hidden — factored ×256 byte chains pull no weight), adds blue weight-size strips (largest op per block = one row of 32; a square = largestOp/32 · 2 B = 448 MiB), and wears a static bf16 tag on each weight-bearing GEMM. Boxes reserve their in-box strip band only in tiers/lenses that can fill it (dtype tiers, the bytes lens) — static/params boxes are compact |
| `tabs` | view | boolean | dense/MoE flip tabs (with per-block FFN tallies) above the FFN column, fused into a scoped enclosure |
| `nocaption` | view | boolean | suppress the foot caption (the page explains the diagram itself) |
| `recipe` | state | `nv-mxfp8` · `bf16` · `dsv3-fp8` · `all-fp8` | per-matmul dtype preset. Even `bf16` pins `router: 'fp32'` (gating is never a precision choice in the Hopper story — the pinned router tag renders in EVERY quant tier incl. marks, so the label never flips between sections; nv-mxfp8 keeps router bf16, NVIDIA's actual choice). `dsv3-fp8` follows the paper exactly, including `o_proj: 'e5m6'` — the customized 12-bit stash format for the attention output (1.5 B/elem, purple; §3.3.3). e5m6 names the STASH: box tags, pickets and ribbons speak COMPUTE dtype (`COMPUTE_DT` maps e5m6 → e4m3 — o_proj's tag is a pinned e4m3 readout, its pickets pink; purple appears only on the attn-out CHIP, and the save format is the head checkbox), flopEq prices the GEMM at fp8 rate, and the fp8ᵀ dual never applies to it (the transpose problem is 1×128 tile scales; the dual threshold is < 1.2 B/elem). `all-fp8` = every linear's stash e4m3 incl. attn-out — the production H100 variant in notes.txt; byte-identical to dsv3-fp8 under attn-replay, pinned in sanity |
| `recipes` | view | comma list of recipe keys | curates which recipe chips/options an instance offers (both the house segment and the legacy select); absent = all. The Hopper article drops nv-mxfp8 (the Blackwell post's recipe) |
| `recompute` | state | `dsv3` · `none` · `attn-replay` · `selective` · `full` | save/recompute marks preset |
| `kind` | state | `moe` · `dense` | which FFN column variant (flip-stable layout; the MLA column is shared) |
| `transposed` | state | boolean | Hopper e4m3ᵀ dual-orientation stashes; part of recipe recognition (canonical per recipe via `RECIPE_T`). Labeled '(expert inputs)': under realistic recompute policies the dual set is only the MoE-FFN inputs (norm2 out + dispatched tokens) — attention-side candidates are replayed or E5M6. Mechanics stay general (any fp8 stash a wgrad reads duals) |
| `cumulative` | state | boolean | parameter parentheticals ×(selected kind's block count), always multiplied out; tabs + enclosure hide. No double-multiplication cues: whenever a number is already multiplied out, the ×N leaves its label (the plan's ×3/×58 tags in cumulative; the experts group's ×256 under multiplied sizes). The `local` variant is PINNED cumulative (no per-block toggle: the fit bar totals the rank, so a per-block diagram would disagree with the chart it explains) |
| `xlayers`, `xinflight` | state | numbers (61, 1) | combined-view multipliers (× layers × in-flight × 4096 tokens). 02's AC/fp8 instances pin the EP64·PP8·DualPipeV context: `xlayers="8" xinflight="8.5"` (an interior rank's 8 MoE layers × the PP+½ law), so the stash readout tallies byte-exact with the fit chart/deck. `xtag` (text) is appended to the readout in parens to name that context. `ctx` (JSON of key→value chips, e.g. `{"GPUs":"2048","PP":"8",…}`) renders the section's FIXED parallelism as locked knob chips in the house style — readouts, not levers |
| `optim` / `consolidated` | view | boolean | param-bytes only: stack byte-component squares under the blue weight squares, in the memory-bars stacked order with the memory-bars segment colors (`BYTE_COMPS`: weights #2a78d6 2 B/param · fp32 gradients #eb6834 4 B/param · optimizer states #1baf7a 8 B/param). `optim` = weights + optimizer; `consolidated` = all three + amber saved-activation chips ON THE WIRES (each saved tensor: name + bytes × 4096 tokens + squares, mostly hollow traces; no total line — the squares live at the chips, no double count; pair with `recipe="bf16"` so stashes price at 2 B). Under ×N the chips scale too and narrow fork chips go two-line. Same global unit, so the ratios are the picture; pinned per block. Component visibility (URL state `showWeights`/`showGrads`/`showOptim`/`showActs`) toggles via the legend (head checkboxes, or margin rows under anatomy): numbers always total exactly the squares shown (nothing visible = no number), and toggling animates per-block-tween style — the color's squares pour in/out and boxes reflow smoothly (numbers snap) |
| `local` | view | boolean | param-bytes only, implies `consolidated`: what ONE GPU holds under the fiat parallelism. Its diagram chips carry the UNION of the quant tiers' stash language at rank scale — dtype tags (dtOf + ᵀ×2), state tooltips, bf16 phantom tails, hollow ↻ counterfactual grids, idle wire-precision tags, and the fp32 aux artifacts (lse/rstd) — everything except the flop pickets (space); the kv fork widens to 200 like the AC tiers, and chipSpace reserves the worst case over dual/phantom edges (`LOCAL_PAR`: 2048 GPUs) with mini-head knobs (URL state): EP as a ± stepper over powers of two 1–64, PP over `PP_CHOICES` = {1, 8} ONLY (PP8 × 2 DualPipeV chunks = DSv3's published 16 virtual stages; intermediate depths are jettisoned — a good zero-bubble partition is sensitive to the front-of-pipe imbalance, so "the PP4 schedule" isn't well-defined without choices this essay won't defend), the RANK picker as a two-way segment (`data-knob="rank"`: 'r0 · emb+head' vs 'r1–7 · peak' — under the slot split every interior rank holds the same 8 MoE layers, so only two choices matter; the · peak tag follows the truth; hidden at PP1; user-visible copy says RANK, DualPipeV's stages being the 2·pp chunks; the slot split itself: emb+3 · fourteen ×4 · 2+head, emb/head each costing a layer's worth), GPUs stepper 128–16384 (stepper buttons walk the OPTION LIST by index, not ×2), and a segmented ZeRO-(off|1|2|3) level picker (zthresh per component), plus the 'e4m3+ᵀ params' checkbox (fp8-RESIDENT parameters, AMAIA-style: weights = the fp8 copies in BOTH orientations at 1×128-scale cost, ×2.0625/2 on the e/d classes; emb/head stay bf16, fp32 master stays in the optimizer bar, norms/router ≈0.1% booked fp8 as disclosed noise; disabled under the bf16 recipe; the effective flag is snapshotted so tweens lerp it). EVERY knob change pours the squares between the old and new configuration (the diagram squares via the `_vtween` factor lerp; the fit chart via the layout blend, diagram-grammar rule 9 — numbers snap; kind-flipping changes snap). Amber wire-chip squares wrap at 16/row (8 in the fork columns) and the wire gaps grow with their rows. Expert weights divide by EP; with ZeRO-1 on, optimizer states shard over each parameter's replication group (dense /DP=world/pp, expert /expert-DP=DP/EP; the EP share and the smaller group cancel, so per-rank optimizer bytes are a uniform 8 B/DP per global param, EP-independent — matches memory.js); the kind and ×N follow the stage (0 = embedding + dense blocks; last = + final norm/lm head; the plan doubles as the stage map, gating those rows); `cumulative` toggles one block ↔ the stage total (tweened; tabs stay hidden in both). THE SCHEDULE IS DualPipeV: `vpp`/`fold` are DERIVED, never knobs (pp > 1 → VPP2·reflect, the V; pp 1 → one trivial chunk), and the only schedule knob left is admission (`sched`: DualPipeV steady state / ×1 mb). Authored configs' vpp/fold keys are ignored. The default stage is the peak. The law (`inflightOf`): each virtual stage v holds D−v chunk-stashes (1F1B admission, D = VPP·PP); a rank's residency = the sum over its `vstagesOf` ÷ VPP. under the V the law is a UNIFORM PP+½ on every rank (8.5 at PP8, the reference-spreadsheet convention, vs the DSv3 paper's coarse PP+1); the law functions stay general (VPP1 → the pp−s staircase; wrap → PP(V+1)/2 − s, rank-0-concentrated) for a later Megatron-schedules post, but no UI reaches them. `ppStage(s, pp, vpp, fold)` returns the hosted segments plus `emb`/`head` flags (all vocab attribution flows through them), multiplying the amber chips/gaps and included in the peak-stage pick. Knobs carry `data-knob` names (gpus/pp/stage/sched/ep/zero) for tests. Head layout: parallelism row, then the fit chart, then the misc row. Fit chart: UNSTACKED per-component log₂ bars + a grey total row on a FIXED axis (256 MiB–16 TiB; gridlines = ×2; the 80 GiB capacity line labels itself ON TOP, leaving the bottom axis to the power-of-two labels). The row labels ARE the legend (name + absolute bytes; whole-gutter hitbox; mousedown SOLOS the component — the filter that matters — and soloing again restores all; dimmed = off). Under a solo the TOTAL row keeps its full extent as a stacked bar: grey =other= base + the highlighted component on top, so the colored width IS the factor available from optimizing just that component and bar ends carry the ×N/÷N factor vs the 80 GiB boundary; a measuring cursor armed only inside the bars: click = point (value + vs-capacity factor), drag = span (any span on a log axis is a factor: ×N with endpoints), click it to clear; the SAVE cluster (top right, visually apart): 'save' locks in the config (re-save to lock in changes), 'reset' returns TO the save, 'reset all' is factory (PP1/EP1/ZeRO-off/bf16/none — the whole-model one-GPU view). A save renders as dotted GHOST bars (main rows and sub-rows; value labels always ride the live bar end) and annotates bar ends + the DIAGRAM numbers with each class's exact change (each sharding class moves uniformly). CHANGE factors wear alert badges: red ▲×N grew / bold ▼×N shrank (direction in the arrow, magnitude always ≥1; red is unused by the components). Soloing a param component ACCORDIONS its breakdown sub-rows (experts / non-expert / emb + lm head — the last also carries the final RMSNorm) open beneath it; soloing ACTIVATIONS opens a PER-OP breakdown instead (`ACT_BUCKETS`, Haziza-style: residuals, norm outs, mla latents, q·k,v, attention out, router state, dispatched tokens, gate/up, swiglu out, + an 'other' catch-all so the list always partitions savedBytes — per-tensor bytes come from analyze()'s savedById, so fp8ᵀ dual copies and aux artifacts price correctly; acts sub-rows are display-only, no part filter; mixed-kind ranks price the dense front at the DENSE layer's stash rate — no router state or dispatched tokens — via a parallel dense analysis at the same marks/recipe) (animated reflow, per author ruling), each with its own save badge + ghost, so e.g. only the experts sub-bar moves under an EP change. Sub-rows are clickable: selecting a part (`partSel` — NOT `part`, which collides with the native Element.part API) filters the DIAGRAM to that part's ops and gives the soloed row the stacked grey+color treatment, one level down from the total's. Stash knobs (precision/recompute/fp8ᵀ/per-op buttons) animate through the same generic tween (the snapshot carries the pre-change analysis so chips and the acts bar lerp bytes). With recompute none, x0 loses its 🔒 (a checkpoint anchor means something only when a replay exists) — the pinned-label line is height-reserved so pinning never reflows; bar motion is the layout blend (linear pixel lerp on the log axis = geometric byte motion; an interrupted tween retargets from the on-screen layout). Local instances have NO margin legend (the chart carries it); optim/consolidated anatomy instances keep the margin rows (`marginlegend` delegation; standalone instances keep head checkboxes). Strip squares FLOOR per component; one hollow trace only where an op would otherwise show nothing (largest remainder's color). Half boxes wrap strips at 21/row. Fixed 448 MiB unit |
| `barsonly` / `knobs` | view | boolean · comma list | article views over the local lens: `barsonly` mounts the head knobs + the fit chart and never mounts the block diagram (it's still computed — the chart derives from it); `knobs="cluster,pipeline,mesh,zero,save,prec,blocks"` discloses a subset of the knob groups (absent = all). Instances keep their own id-keyed URL state, so each discussion pins its own scenario (see studies/scratch-bars.html; linking/sidebar TBD) |
| `snapshot` / `from` / `to` / `solo` / `sandbox` | view | boolean · JSON · JSON · comp name · layer id | a story beat over the local lens (implies barsonly; no knobs, no URL state). Chart interactions follow the ONE capability seam (`_wireBars`/`_barCaps`): every view gets MEASURE (the scrub cursor + drag ruler — reading a span as a factor); full/barsonly also get MUTATE (solo/part gutter clicks); snapshots are config-static so mutate is off (the gutter loses its pointer cursor). With several charts on one page, a drag on one chart dismisses any other chart's ruler. `from` is a FULL config by fiat (world/pp/ep/zero/sched/vpp/fold/stage/recipe/recompute; unlisted keys mean the neutral nothing-applied defaults — same rule as deck steps, so figures never drift when interactive defaults change; stage defaults to the peak) rendered as the saved baseline (ghost bars + the saved-label line), `to` is the change, applied on top, rendered live with ▲/▼ factor badges. `solo="weights|grads|optim|acts"` picks the component under discussion: its breakdown accordions open, other rows dim to name-only (explicitly unchanged/off), the total goes stacked grey+color. `comps="weights,grads,optim,acts"` instead shows a visible SUBSET for additive-tally beats. The total row is the full consolidated mass (stacked: grey "other" + the shown components) — context for what the story hasn't introduced yet, exactly like the sim; `nototal` drops it (intro beats, where the 80 GiB shading already says "doesn't fit"). Off components' rows don't render at all (a dimmed unclickable name is a dead affordance in a figure). Omitting `to` makes a single-config figure (no ghosts, no badges, no saved-label). `parts` opens EVERY visible param component's experts/non-expert/emb+lm head breakdown at once (the accordion machinery generalized from single-solo to a set of open rows — interactive widgets still open one, under the solo). Snapshot cards shrink-wrap their chart (width fit-content) and pinless snapshots drop the reserved saved-label line. `hypothetical` (optional text) marks a counterfactual beat — the card border goes dashed (the ghost language's not-real cue) with an italic tag ('hypothetical — not what DSv3 did' or the attribute's text). `sandbox="<layer-id>"` adds a "play with this scenario in the full widget" link that loads the story into the target (from as its save, to tweened in) and scrolls to it — the target resolves at click time (anatomy-wrapped layers upgrade late). Authoring an optimization story = a chain of snapshots where each `from` is the previous beat's config |
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
| *(forwarded)* | | `controls recipe recompute detail transposed for nocaption kind xlayers xinflight xtag lens strips optim consolidated local cumulative` pass through to the inner layer |

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

## `<dsv3-beat-deck>` — the optimization story as a slideshow

One fit chart, stepped explicitly. Each `<section>` child is a beat: a FULL
config in `data-config` (never a patch — stepping past a hypothetical
reverts it automatically), an optional `data-hypothetical` (the dashed
not-real card + authorable tag), and the step's caption as its HTML (author
prose lives in the page). The deck always shows every bar and sub-bar
(fixed focus: the story is one scene). A forward step
saves the last REAL step as the baseline (hypotheticals are parentheses —
never measured against) and pours the bars to the new config through the
layout blend (ghosts plant FIRST, then the bars move — diagram-grammar rule
9); ▲/▼ badges narrate the move. Every direction animates (forward, back,
jump). Nav buttons + arrow keys; the current step persists in the URL
(`d:<id>`). The deck renders as a framed CARD (study.css) and reserves its
tallest caption at mount so slides never change its height. The inner layer
is snapshot-mode but LIVE (`live` attr): the reader may fiddle the knobs —
a DETOUR, marked by an amber '✎ detour' nav tag + an '↩ back to the slide'
reset button and a dimmed caption — the reset pours back to the slide's
config in place, and stepping first rewinds the detour the same way, then
animates the step's own delta. The visual audit covers every step.

## `<dsv3-pp-fold ep=… view=…>` — the V-fold stage map

How the DualPipeV stage split distributes PARAMETERS over ranks (placed in
02's PP section; scratch host: studies/scratch-fold.html).
Two views of the same 16-chunk contiguous split (`ppStage(c, 16, 1,
'reflect')` — the exact split the memory model charges): `virtual` unrolls
the chunks as a 16-deep chain; `physical` folds chunk v onto rank
min(v, 15−v) beside its partner — the V. The toggle ANIMATES the fold
(frame-stepped; each chunk segment flies to its rank; physical rows are
double-pitch so total height is reserved — no reflow). Bars are SOLID and
LINEAR on a FIXED graduated scale — countability lives in the unit RULER
under them (minor tick = 128 MiB bf16, major = 1 GiB; labels in bytes),
per the site language: solid bar + ruler for memory comparison, squares
for byte piles, pickets for compute. Down-pass chunks light blue / up-pass
dark (the depth gradient), vocab shares (emb / head+norm) wear a dashed
outline — at EP64 they dominate rank 0, the fold's signature. A model-stack
MINIMAP on the left (emb cap · 61 layer cells, dense ×3 distinct · norm+head
cap) lights the hovered bar's layers — hovering s0 folded lights BOTH ends
of the model at once — with a reserved-height readout line naming the
hovered rank's composition and exact param count (`data-params` carries
exact values; rank totals = pairwise chunk sums, tested). The expert sharding is a KNOB on the figure (an
EP stepper, default 64 — honesty over fine print: rank-resident params
depend on it; bars tween between shardings on the FIXED scale — EP is a
length change, honestly — and exact values snap). `ep` seeds the default;
`view="physical"` opens folded. Attributes are figure-authoring only; no
URL state.

## `<dsv3-sheet layer=…>` — the full model's formula sheet

A spreadsheet-like readout of the CELL GRAPH (`src/cells.js`) that the bound
local widget prices from: one row per cell — coordinate name (`P1`/`Q2`/`W1`…),
a one-line label, the formula, the EXACT value (primary column; byte counts
are exact — every divisor is a power of two and every dtype rate a dyadic
rational on integer counts ≪ 2^53), and a rounded `≈` column. The formula
STRING shown is what the engine evaluates (`evalExpr`) — the chart, the
tooltips and the sheet cannot diverge, and `scripts/sanity.mjs` replays the
shard math independently and asserts `===` across a config matrix (totals
AND sub-cells). The ACCORDION computes the totals: `W1 = W2 + W3 + W4` (per
sharding class), `A1 = A2 + … + A11` (per stash bucket — the buckets
partition the op graph's savedBytes exactly), `T1 = W1 + G1 + O1 + A1`.
Formula-switching INPUTS get explicit rows: `Z1` (ZeRO level), `F1`
(fp8-resident params as a 0/1 that rides the weights formulas —
`(2 + F1 × 2 × 4/128) × Q1`), and per-bucket `R•` recompute choices (0/1)
multiplying the bucket's SAVE-EVERYTHING rates at the current recipe, so
flipping the policy flips the 0/1, not the formula (a partially-kept bucket
— the catch-all's aux, x1 under `full` — falls back to as-is rates, labeled
so). The rates themselves are DECOMPOSED per saved tensor — dims × B• where B•
is the bucket's PRECISION INPUT row (B/elem: 2 bf16 · 1.03125 e4m3+scales ·
1.5 e5m6 · 4 fp32; a ᵀ dual FOLDS into the value) + the fp32 aux artifacts
as literals — built from the op graph and VALIDATED (the string must
evaluate back to the exact rate, else the literal stands). FORMULA
STABILITY is the design rule: toggling a model input changes input VALUES,
never a formula's shape — ZeRO resolves to per-component shard-group cells
(S1–S6 — real formulas via the language's 0/1 indicator ≥: (Z1 ≥ 3) × (P5 - 1) + 1, value 1 when unsharded), emb/head presence to E1/H1 (0/1, L3 =
E1 + H1), fp8 params to F1, precision/ᵀ to B•. BREAKOUT buckets (residual, norm outs, mla latents, attention out, the
remainder — every bucket a preset can split) get per-TENSOR sub-cells
(`A3a`…) so each row stays a whole 0/1 `kept?` choice, with per-tensor
precision inputs (`B4a`… — a bucket may mix bf16 latents with e4m3-rate
norm outs) and the aux artifacts (lse, rstd) SPLIT OUT as their own rows.
The one known piecewise exception: P6 flips between '1' and 'P2 + 0.5'
with the schedule/PP. A `simplify` checkbox on the sheet drops the
negligible terms (the aux rows and the final norm in Q3) — allowed to
change formulas by design; its totals drift slightly from the (always
exact) chart, and a note says so. The sheet indents child
rows (sub-cells depth 1, per-bucket R•/B• depth 2), keeps labels to one
line (nowrap), and hovering any formula variable shows its cell card
(click = jump to its row). Rows carry JUMP affordances (click the
coordinate cell): a model input lands on its controlling knob (P1→GPUs,
Z1/S•→ZeRO, F1→the e4m3+ᵀ checkbox, L•/E1/H1→the rank picker, P6→sched)
and an activation row on its chip in the diagram (`data-chip` on every
local chip; aux labels are `<id>:aux`), pulsing amber on arrival. Model
inputs are also EDITABLE from the sheet, IN the exact-value cell: steppable
rows flank the number with − + (CSS pseudo-content, so copy/paste never
picks the glyphs up; generous padded hitboxes), toggle rows make the NUMBER
itself the button (dashed-underline button language, the whole cell is the
target). Edits fire on MOUSEDOWN so rapid presses never straddle a resync
and drop. EDITABLE value cells wear a tinted face (the button language —
the dashed underline was too subtle); a ± at its bound dims and inerts, and
a pinned toggle (x0's kept?, a recipe-pinned precision) renders as a plain
untinted cell — enabled-ness is READ from the widget's own controls, so the
sheet can't disagree with the diagram. The formula is the rightmost column. Every edit drives the widget's OWN control — steppers
step (P1/P2/P3), the ZeRO segment steps (Z1), two-chip segments flip
(P6→sched, L•/E1/H1→rank), checkboxes click (F1, o_proj's B via E5M6),
precision rows click their dtype button (B• via the node's `dtc` channel,
recorded in blockGraph where the rate is priced), and kept? rows click the
mark button — so bounds, tweens, URL state and the diagram stay linked
with no second mutation path (a kept? edit leaves the presets: pick a
recompute chip to return). In the
local diagram, chips report sizes consistently — saved chips their bytes,
recomputed chips (and replayed aux) their would-be size in parentheses —
and the saved-for-backward tooltip is BRIEF and attached to the chip's
NAME tspan only, so the byte value's raw-B hover never conflicts. Cells without a formula are model INPUTS (slot-split layer counts, the
op-graph stash rates D1/D2) — drill-down ends at their labels. Binds like
the pp-schedule strip (poll for the layer id, resync on its `recipe`
event), so rows update live as knobs move; `reveal(id)` scrolls to a row
and highlights it — the tooltip's bold coordinate is the jump affordance.

The same cells power the fit chart's FORMULA TOOLTIPS: chart value/name
labels carry `data-cell`; hovering shows that cell's entry (name · label,
`= formula = value (exact B)`), clicking PINS it, and clicking a coordinate
inside a pinned tip pushes that cell's entry below — a STACK growing
downward, one path through the graph at a time (clicking a name higher up
truncates the path there first). Clicking elsewhere unpins. Op-box tooltips
no longer carry hand-written FLOP expression strings (the divergence-prone
pattern the cells replace) — just the FLOP count, dims note and exact
parameter count.

## The visual audit (src/audit.js)

The audit keys on the VISUAL LANGUAGE, never the model: each pattern the
chart draws implies arithmetic (the semantic-implications table in
docs/diagram-grammar.md), detected from geometry and checked against the
exact values linked behind the rendered numbers (`data-true`, `data-pin`).
Rendered value == fmtBytes(true); a row's rightmost solid bar edge sits at
px(value) (covers stacked bars); a dashed ghost sits at px(saved) and the
▲/▼ badge is the exact ratio; indented `· name` rows sum exactly to their
parent (rounded digits can't be summed). Model identities are deliberately
NOT re-derived here — the stage-split-partitions-the-exact-total check
(catches dropped norms / doubled vocab across all 42 schedule geometries)
lives in scripts/sanity.mjs; `data-true` is the bridge. Besides findings,
`auditFitCharts` returns per-chart `reports` narrating every implication it
verified (message + the pattern's selector) — the page-wide OVERLAY renders
these: Alt+A (hidden debug mode, persisted in the URL hash key `audit`)
floats a ✓ N chip on EVERY fit chart (✗ N alarm-red if a lie survives);
clicking a chip drops that chart's report as an overlay — one line per
verification with its exact arithmetic — and hovering a line lights the
pattern it re-derived, dimming the rest. Nothing reflows: chips and panels
are absolutely positioned, and the mode costs no reserved height anywhere.
Charts re-render wholesale, so a MutationObserver re-audits after the page
has been quiet for a beat (never mid-tween: blended geometry is between two
truths); pinned-open reports survive re-audits. All overlay code lives in
src/audit.js next to the checks it renders — the battery and the overlay
run the SAME auditFitCharts.
Battery scenarios: tests/audit.js (02's charts + a corruption canary
proving the audit can fail), tests/auditbars.js (after saves, knob turns,
solos), tests/overlay.js (the overlay itself). Vision/LLM checks stay out
of the battery — rounding-tolerant OCR is strictly weaker than exact values
for arithmetic.

## `<dsv3-pp-schedule layer=…>` — the pipeline-schedule strip

One row per PP stage, time flowing right; F cells one slot, B cells two
(backward ≈ 2× forward FLOPs), every cell numbered with its microbatch.
Colors follow the byte language (F stashes activations = amber; B consumes
them into gradients = orange). Bound to a local-lens layer it follows the
layer's PP / schedule / stage (the selected stage's row is tinted). Its
only OWN control is the drawn-microbatch knob — the PP/sched replicas
stopped paying once PP became {1, 8} and the schedule became DualPipeV
(the ×1 mb wave still draws when the layer's sched knob says so). The sX gutter labels ARE the stage
picker (whole-gutter hit rects, `.stghit`; the selected row's label is bold
and its row tinted; after a click the strip holds focus so ↑/↓ walk the
stages, clamped at the ends) — no stage dropdown on the strip. Perfetto-
style panning is DIRECT: press and drag pans the timeline (a 4px horizontal
threshold splits pan from pick, and a pan suppresses the click that follows;
shift+drag pans from the first pixel and shift-click never picks); the grab
cursor parks whenever the strip overflows its column. The scroll
container sets overscroll-behavior-x: none so hitting the strip's edge
never triggers the browser's back-swipe. `noflight` drops the in-flight braid AND all the
stage machinery with it — no tint, no gutter hitboxes, keys inert (a pure-
schedule figure; with no braid there is nothing a selection would feed). Drawn-microbatch count is
attr-only (`mb`, default 64 — a real step's worth; 'auto' = depth+4, just
enough to reach steady state; the memory model needs no m, its law assumes
m ≥ pp) — the strip carries NO controls of its own now. ×1 mb draws the
single-microbatch wave. The strip ALWAYS draws DualPipeV: with m ≥ 2·PP
(and pp > 1) it is the OFFICIAL program, ported step-for-step from
deepseek-ai/DualPipe
(`_officialDPV`): eight phases per rank, the zero-bubble B/W split (B =
input grads, one slot; W = deferred weight grads, one slot, pale dashed
cells, FIFO), and fused F&B blocks in the steady state, drawn as two
full-height cells sharing an edge — contiguity is the fusion cue: every
ordinary op wears a 2.5px trailing gap, a fused pair touches (the F's result
is modeled available one slot in — the real kernels interleave). When the
official program can't run (m < 2·PP, the ×1 mb wave, PP1), the greedy
engine fills in on the same virtual chain — 1F1B admission per virtual
stage, each rank interleaving its chunk queues earliest-ready (ties
backward-first, then the deeper chunk); up-pass chunks wear deeper shades.
The official program's drawn peak residency is exactly 2pp+1 half-rank
chunks — the law survives the real schedule — and cells carry
data-v/t0/t1 so tests count it. Whenever a drawn peak differs from the
modeled law, the bracket label appends `— the model charges N (its 1F1B
law)`. (The textbook 1F1B/ZB-H1/GPipe families and the wrap fold existed
briefly and were removed for this post's pedagogical scope — version
control is the archive; the shared _resolveProg engine remains for
hand-built programs.) Below the schedule, an IN-FLIGHT section (same svg, so the
horizontal scroll is shared) shows the selected stage's stashes as lifetime
lanes — the F cell that stashes a microbatch, an amber tail while it's held,
and the B cell that frees it (data-stash) — so the braid's thickness IS the
in-flight count. Clicking a stash bar (whole-row-band hitbox) lights exactly
its two ops in the schedule — the F that made it, the B that frees it — and
dims everything else; click again to release (any redraw also clears it; the
selected bar's marks thicken). Hover is deliberately inert. A dimension bracket at the maximal steady-state moment
measures the braid (text[data-peak], pointer-transparent), labeled in
microbatches: the number the memory bars charge. Cell
microbatch numbers shrink to a 6px font when double digits meet a 1-slot
cell (no wide convention exists; papers widen cells instead) and hide past
PP32 (cell budget).
PP/sched changes tween the strip's height (12-frame ease-out, deterministic).
No state of its own; unbound instances read `pp`/`sched`/`stage` attributes.

## Other elements (unchanged conventions)
- `<dsv3-trace level height title config>` — the canvas trace viewer over the
  simulator (not yet on a published page; the timing posts' widget).
