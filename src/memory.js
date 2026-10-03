// Per-layer stash analysis from the block op-graph (src/blockgraph.js):
// memory = the outputs of save-marked ops that backward actually needs.
// Pure module (no DOM) so Node can use it.

import { archOf } from './model.js';
import { blockGraph, analyze, resolveMarks } from './blockgraph.js';
import { resolveMatmuls } from './recipes.js';


// Stashed activation bytes per token for one layer, broken down by component.
// Derived from the block op-graph (src/blockgraph.js): memory = the outputs of
// save-marked ops that backward actually needs; `recompute` names a marking
// preset (none / dsv3 / selective / full) and `saved` holds per-op overrides.
export function actBreakdownPerToken(kind, a, recompute, mm, saved, seqLen = 4096, transposedStash = false) {
  if (kind === 'embed') return { residual: 2 * a.hidden };
  if (kind === 'head') return { logits: 6 * a.vocab }; // bf16 logits + fp32 softmax
  const marks = resolveMarks({ recompute, saved });
  return analyze(blockGraph(kind, a, mm, seqLen), marks, transposedStash).buckets;
}

// Full graph analysis for one layer kind (memory + replay overhead), for the
// schematic and the ZeRO-3 timeline.
export function layerAnalysis(kind, cfg) {
  const a = archOf(cfg);
  return analyze(blockGraph(kind, a, resolveMatmuls(cfg), cfg.seqLen), resolveMarks(cfg), cfg.transposedStash ?? false);
}
