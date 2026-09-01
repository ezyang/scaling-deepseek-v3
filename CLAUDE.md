# Claude instructions for scaling-dsv3

Interactive essay series "DeepSeek-V3: from roofline to reality" — plain
HTML/JS ES modules, zero dependencies, no build step. Prefer less code over
generality.

## Publishing & git

- NEVER `git push` without an explicit request. Commit locally and report.
- index.html and studies/01-deepseek-diagram.html are LIVE
  (deepseek-v3.ezyang.com). No unreviewed AI-generated prose in them, surgical
  edits only, and regression-check their widgets before committing renderer
  changes (shared src/ is easy to regress).
- AI-drafted prose is allowed in unpublished posts but must carry an explicit
  disclosure footnote and expects a human edit pass. Site text is assumed
  human-written unless disclosed otherwise.
- Prose is author-owned: coordinate before editing it; text below a
  "SLOP BELOW" marker is placeholder — don't edit or polish it. The author
  edits files mid-session: check `git status` before committing so unrelated
  WIP doesn't get swept into a commit.
- Delete dead sections outright; version control is the archive.
- Shipping a new post: uncomment its entry in BOTH studies/series.js
  (`SERIES` — drives the prev/next cards and the "post N of M" strip) and
  index.html's posts list. The two must stay in step; tests/seriesnav*.js
  pin the links.

## Testing (run after any renderer change)

- `node scripts/sanity.mjs` — model invariants; parameter totals are
  exact-asserted against the published checkpoint.
- `node scripts/diagramlint.mjs` — geometric grammar (arrowheads/text/wires)
  plus placeholder-height drift across a widget matrix.
- `node scripts/battery.mjs [filter…]` — the whole battery (sanity +
  diagramlint + every scenario in tests/) in parallel, ~3 s. Run this one.
  Browser drivers prefer Playwright's chrome-headless-shell (~10× faster
  startup, never touches the real Chrome profile); `CHROME=<path>` overrides.
- `node scripts/interact.mjs <page> <scenario.js>` — sequenced
  click/hover/assert scenarios in headless Chrome. Prefer interaction tests
  over static screenshots; invest in test affordances freely. Durable
  scenarios live in tests/ with a `// @page <path>` header (optional
  `// @args --width N`); one-off debug probes stay in /tmp.
- Verify pages in headless Chrome (`node --check` misses browser-only
  errors). Every widget instance on a real page carries a measured
  `min-height` placeholder; the linter enforces it to ±3px (any gap is a
  visible text reflow when the widget mounts) at BOTH the supported minimum
  viewport (1366px — every container is width-capped, so layout is identical
  at any width above it) and a wide one, and reports the value to paste.
  Below 1366 knob rows may wrap; ≤860 is the stacked mobile layout.

## Modeling conventions

- Exactness: formulas count every term (RMSNorms, router bias, …) even when
  it's a rounding error. When reproducing a published figure, follow the
  model author's own convention, established from primary sources.
- No fudge factors: rooflines are understood to be optimistic; never add a
  flat overhead pad to make totals match reality.
- Simplifications are disclosed prominently, ideally with a toggle between
  the simplified and real numbers.
- Binary units: GiB/MiB/TiB whenever a power of two is meant (an H100's
  "80 GB" is 79.65 GiB — prefer GiB).
- Training only; inference is out of scope.

## Widget & diagram conventions

- docs/widgets.md is the attribute API reference; docs/diagram-grammar.md is
  the visual grammar (the linter enforces its geometric parts). Keep both
  current when the renderer changes.
- Toggles never cause an abrupt layout jump: either reserve worst-case space
  (including for inline text buttons in prose) or animate the reflow (the
  strips="absolute" tween). How far animated reflow scales as more diagrams
  arrive is an open question.
- Kind/detail flips preserve element anchors; byte-strip squares use ONE
  global unit everywhere so views never rescale against each other, and no
  byte is ever double-counted anywhere in a figure.
- Animations are quick (~200 ms), deterministic frame-stepped (no
  rAF-timestamp math), and every co-located element participates — fades and
  dependent overlays tween together, nothing pops.
- Widget state persists in the URL hash (shareable, refresh-stable); presets
  recognize themselves when controls return to a preset's exact state.
- Micro-typography: parameter counts are grey, never italic, consistently
  styled; dtype colors are the warm-magenta precision family (bf16 ink · e4m3 pink · e5m6 purple · fp32 brick), kept clear of the byte-component blue/orange/green; no
  space-wasting negative labels ("not needed").
