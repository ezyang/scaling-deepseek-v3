# The block diagram's visual grammar

The `<dsv3-layer>` renderer (and its satellites: the anatomy plan, the stack
strip) follows a consistent visual language. Today the rules live implicitly
in coordinate arithmetic; this file states them explicitly. `scripts/
diagramlint.mjs` mechanically enforces the geometric consequences (marked ✓);
the rest are conventions to uphold when editing the renderer.

## Structure

1. **Columns own a spine.** Each column's dataflow runs down a single vertical
   spine on the LEFT of its boxes (`SX1`, `SX2`, `shMid`). Boxes sit to the
   right of the spine; the spine threads box-to-box.
2. **Ordinary outputs continue downward** along the spine, annotated by a
   tensor chip beside the wire. **Aux backward artifacts (rstd, lse) exit
   RIGHT** out of their box with a short arrow.
3. **Bypass rails run OUTSIDE the column** (left of the spine): the residual
   rail (`RAIL1`, plan rails), the k_rope rail, the top-k-weights rail
   (right side, `gateX`). Rails turn with right angles only.
4. **Forks get a junction dot** (`r=2.5` filled circle) at the point where a
   tensor is consumed by two paths; a tensor drawn twice without a dot implies
   (incorrectly) that it is duplicated.
5. **Arrowheads appear only at destinations** — one `marker-end` per wire, at
   the point of consumption. No mid-wire arrowheads. ✓ (the linter flags
   arrowheads landing inside text)
6. **Related boxes align by row** across columns/kinds: the shared-expert
   boxes align with the grouped-GEMM rows; the dense FFN boxes align with
   their MoE counterparts (enforced by advancing through MoE row arithmetic
   via `gapM`); the residual adds clamp to `col1End` so both columns end
   together.
7. **Kind flips preserve anchors.** Flipping dense ↔ MoE must keep every
   surviving element at the same coordinates, leaving whitespace where the
   removed machinery sat. Same rule for detail ↔ terse: major anchors persist.
8. **Reserved, non-reflowing space.** Chip gaps are sized for the WORST case
   (`chipSpace`: saved, bf16, dual-orientation) so toggling marks/dtypes never
   moves the layout. Space is reserved only for content the INSTANCE can ever
   render: the in-box strip band exists in dtype tiers and the bytes lens, so
   static/params boxes are compact (32px, not 38/60). A toggle that must
   change height animates the reflow instead of jumping (the ×N strips tween).

## Text and labels

9. **Text never collides**: no text under an arrowhead ✓, no wire passing
   through a label ✓, no two labels overlapping ✓, no text crossing a box
   border or the SVG edge (clipping) ✓.
10. **Label placement**: tensor chips sit right of the spine (`sx + 14/16`);
    wire-side labels sit right of the wire; box param counts are grey
    parentheticals appended to the dims line; group labels sit inside the top
    edge of their group rect.
11. **Vocabulary**: op names lowercase (`attention`, `router`, `ffn gate/up`,
    `residual add`); proper abbreviations keep their casing (MoE, MLA,
    RMSNorm, SwiGLU, RoPE, FFN). "MoE block" in UI labels; "sparse" only as an
    informal prose synonym.

## Styling classes (the tokens of the language)

| class | meaning |
|---|---|
| `.box` | matmul/attention op (bold `.name` + grey `.dims`) |
| `.op` | non-matmul op (grey pill, `.oplabel`) |
| `.comm` | communication (violet) |
| `.res` | residual add (dashed) |
| `.micro` | elided kernel (detail-only, italic) |
| `.grp` | grouping enclosure (thin, `grplabel` inside top) |
| `.tsave/.tredo/.tidle` | tensor chip states (saved amber / replayed italic / idle grey) |
| `.wire` | all dataflow, `#898781`, 1.2px, right angles |

## Toward a DSL

The useful abstraction is NOT a general graph-layout engine. The candidate
primitives, if we ever extract them: `column(spine)`, `seq(op…)`,
`fork(dot)`, `bypass(rail, rejoinAt)`, `alignWith(otherRow)`, `outputChip`,
`auxOut`, `reserveWorstCase`. Until then, the linter is the contract.
