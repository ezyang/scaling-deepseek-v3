# scaling-dsv3 — DeepSeek-V3: from roofline to reality

An interactive essay series on training DeepSeek-V3, live at
[deepseek-v3.ezyang.com](https://deepseek-v3.ezyang.com): the architecture
as an infra-oriented block diagram (01), memory on Hopper as a formula sheet
every reader can edit (02), and the speed-of-light step time worked from the
roofline (03). Unlisted drafts under `studies/` are later posts in progress.

Plain HTML/JS ES modules, zero dependencies, no build step.

## Run

```sh
python3 -m http.server 8000        # any static server; ES modules need http
open http://localhost:8000
node scripts/battery.mjs           # the whole test battery (~12 s)
```

## Layout

| path | what |
|---|---|
| `src/model.js`, `src/params.js`, `src/recipes.js` | The architecture, the checkpoint-exact parameter counts, and the per-matmul precision recipes. |
| `src/blockgraph.js`, `src/memory.js`, `src/localmodel.js`, `src/cells.js` | The transformer block as an op DAG (what each save/recompute marking stashes), the per-rank schedule geometry, and the formula-cell graph the sheets and fit charts evaluate. |
| `src/viewer.js`, `src/anatomy.js`, `src/sheet.js` | The widget family: block diagram + fit chart, pipeline fold and schedule strips, the editable formula sheet. |
| `src/ladder.js`, `src/mmfig.js`, `src/epsim.js`, `src/mesh.js`, `src/fsdpsched.js` | Post 03's figures. |
| `src/kerntrace.js`, `src/kernels.js`, `src/fsdp.js`, `src/dsv41.js` | Widgets of the unlisted drafts. |
| `studies/` | The posts; `series.js` is the manifest, `study.css` the shared style. |
| `scripts/`, `tests/` | The battery: sanity invariants, the diagram linter, numeric and pixel goldens, headless interaction scenarios. |

Conventions, the publishing gates and the widget API live in `CLAUDE.md`,
`docs/widgets.md` and `docs/diagram-grammar.md`.
