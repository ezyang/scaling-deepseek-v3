# scratch: ideas for 02 (AI brainstorm — pick, twist, or discard)

Everything here is grounded in machinery that already exists unless marked
**[new widget]**. Numbers quoted are from our model as of today.

## The spine: the essay is "find ÷111"

8.65 TiB ÷ 80 GiB = **×110.7 over budget ≈ 6.8 doublings** on the chart where
every gridline is ×2. That's the whole post in one sentence: the opening tally
sets the debt, and every section pays some of it down. Two devices:

- Each lever section is a snapshot beat whose `from` is the previous beat's
  `to` — the chain we already support. The final beat's `to` is the DSv3
  recipe and the bar ducks under the shading for the first time. Give that
  moment room; it's the payoff of the whole series so far.
- **[new widget, cheap]** a *factor ledger*: a one-line strip after each beat
  — `so far: ÷16 · still need: ÷6.9` — rendered on the same log axis as a tiny
  bar from 8.65 TiB to the current total, with the 80 GiB line. It's the HUD
  that keeps a reader oriented across ten figures. Could literally be a
  snapshot with `comps=""` (total row only) if we allow that.

## Beat order that maximizes surprises

1. **Tally** (done): 8.65 TiB, accordion map.
2. **PP16** — the surprise beat: weights/grads/optim ▼×16, but solo the
   activations and the badge is **▼×~1**. PP divides the *layers* per rank and
   multiplies the *microbatches in flight* by almost exactly the same factor
   (1F1B: 61/16 layers × 16 in flight). One snapshot pair teaches the deepest
   fact in the post: **pipeline parallelism does not save activation memory** —
   it only smears the same activations across time. This is why the rest of
   the essay exists.
3. **EP64** — solo weights: experts ▼×64, non-expert ▼×1, blended ▼×~40.
   The accordion is the figure; the expert/non-expert decomposition earns its
   keep here, not in the tally.
4. **ZeRO-1** (the demo beat, done): non-expert ▼×128, experts ▼×2 (EDP),
   blended ▼×4.6 — the EP-cancellation twist.
5. **The schedule beats**: ×1 mb (the fiction) → 1F1B (acts ▲×15 at the peak
   stage — the price of keeping the pipe full) → DualPipeV (uniform PP+½;
   point at the strip widget's braid, don't re-explain). The in-flight braid
   with the dimension bracket is already the perfect companion figure.
6. **AC** (recompute=dsv3 / attn-replay): acts ▼; this is where the FLOPs
   trade-off gets its forward reference ("what does the replay cost? next
   post"). Also the one place the wire-chip diagram might be worth ONE
   appearance — activations are the only component whose *location* (on the
   wires) is pedagogically load-bearing.
7. **fp8** — stashes and weights shrink; note the fp8ᵀ dual-stash wrinkle as
   the "reality is fussier" beat.
8. **Fit** — the final config, one snapshot, bar inside the budget, sandbox
   link. State the headroom in GiB, not just the factor.

## The lever × component matrix **[new widget, medium]**

A compact generated table: rows = levers (PP, EP, ZeRO-1/2/3, sched, AC,
fp8), columns = components (experts / non-expert / vocab / grads / optim /
acts), each cell the factor that lever applies to that component (÷PP, ÷EP,
÷DP, ÷EDP, ×in-flight, ~1, —). The snapshots show *instances*; this is the
*map*, and it's derivable from the model (exactness preserved) rather than
hand-written. It also answers "which lever should I reach for" at a glance —
the practitioner takeaway. Could live near the end as the summary figure.

## Per-stage mini-profile **[new widget, cheap-ish]**

Sixteen tiny total-bars, one per PP stage, same log axis — the "skyline" of
the pipeline. Shows in one glance: why the peak is an early stage under 1F1B
(the staircase), and how DualPipeV flattens the skyline. We already compute
per-stage totals (peakStage iterates them); this is mostly rendering. Pairs
with the stage selector story without needing interactivity.

## "Why not X" sidebars (your stretch notes)

- **Why no TP**: MLA's per-head dims are small and the expert FFN is only
  2048 wide — TP would shard GEMMs below efficient tile sizes, and it demands
  per-layer collectives on the critical path. EP is MoE's natural sharding
  axis: whole experts, coarse-grained, and the a2a is already being paid.
  (DSv3 paper says this outright — citable.)
- **Why no CP**: 4096-token sequences; attention memory is linear in our
  stash accounting and tiny next to the FFN stashes. CP is a long-context
  tool; nothing here needs it.
- **ZeRO-3 warning** (your forward-ref note): the bars make ZeRO-3 look free
  (weights ▼×DP!) — the trap is that the chart shows *bytes*, not *time*; the
  all-gather it implies cannot hide behind compute at this arithmetic
  intensity. A one-line "the memory chart cannot see communication" warning,
  promised to the comms post. Maybe even a recurring icon for "this lever has
  an off-chart cost".

## Trust and reality-gap beats

- **Provenance callback**: 8.65 TiB decomposes into 671,026,419,200 × (2+4+8)
  bytes + 106.4 GiB of stashes — and 01 already cross-checked the param count
  against the released checkpoint. One sentence buys a lot of trust.
- **Fake PG payoff**: the intro promises fake-PG simulation; close the loop —
  end the fit section with "run it under fake PG and compare" and an
  expected-vs-measured table stub. Even with TODO numbers it tells readers
  the model is falsifiable.
- **What we don't model** (disclosure box): allocator fragmentation, NCCL and
  EP-dispatch buffers, CUDA context (~0.5–1 GiB), the MTP head (DSv3 trained
  with it; we model 61 layers without), logits/loss spike at the last stage,
  and that "80 GB" is 79.65 GiB. The headroom in the final beat should be
  read against this list.
- **Non-commutativity aside** (optional, subtle): component badges multiply
  across beats but the *total* badge doesn't — totals aren't a product of
  independent factors, which is exactly why the log chart shows components
  unstacked. One sentence where a reader might otherwise try to multiply
  ▼×16 · ▼×4.6 · … and get confused.

## Prose-shape suggestion

Per beat, four moves, ≤2 short paragraphs total: (1) the mechanism in one
sentence, (2) the figure, (3) what it did NOT change (the dimmed rows are
this sentence's evidence), (4) the cost that memory can't see (forward ref).
The bars carry all numbers; prose never repeats a number the bar already
shows — it interprets it.

## Small polish ideas

- The ZeRO beat's saved-label (`saved: EP64·PP16·…`) is the only place a
  snapshot names its full config — consider making that line a standard
  caption on every beat so no figure is "underdetermined" (your worry).
- Beat captions could carry the running chain marker: "beat 4/8 · config so
  far: PP16·EP64·ZeRO-1".
- A `comps=""`/total-only snapshot variant would make the ledger strip free.
