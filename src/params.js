// Named parameter quantities, derived once from the architecture and shared
// by every surface that displays a count (the tally, the plan strip, the
// kindtabs) — so they can't drift apart.
//
// Audited against every main-model non-`weight_scale_inv` tensor in the
// published deepseek-ai/DeepSeek-V3 safetensors headers (embedding, layers
// 0–60, final norm, and lm head). In particular, the router's learned
// e_score_correction_bias is a real 256-element parameter tensor. The
// auxiliary MTP module is deliberately outside this inventory.
import { DSV3 as A } from './model.js';

const qkvDown = A.hidden * (A.qRank + A.kvRank + A.qkRope);
const qUp = A.qRank * A.heads * (A.qkNope + A.qkRope);
const kvUp = A.kvRank * A.heads * (A.qkNope + A.vHead);
const oProj = A.heads * A.vHead * A.hidden;
const mlaNorms = A.hidden + A.qRank + A.kvRank;      // norm1 + the two latent norms
const expert = 3 * A.hidden * A.moeInter;            // routed and shared are the same shape
const routerWeight = A.hidden * A.routedExperts;
const routerBias = A.routedExperts;
const router = routerWeight + routerBias;
const denseFfn = 3 * A.hidden * A.denseInter;

export const PARAMS = {
  embed: A.hidden * A.vocab,                          // untied: the lm head is another copy
  qkvDown, qUp, kvUp, oProj,
  attnQkv: qkvDown + qUp + kvUp,
  attnOut: oProj,
  mlaNorms,
  mla: qkvDown + qUp + kvUp + oProj + mlaNorms,
  router, routerWeight, routerBias, expert, denseFfn,
  normsBlk: mlaNorms + A.hidden,                      // + norm2
  denseFfnBlk: denseFfn + A.hidden,                   // FFN half incl. norm2
  moeFfnBlk: (A.routedExperts + A.sharedExperts) * expert + router + A.hidden,
  finalNorm: A.hidden,
};
PARAMS.denseBlock = PARAMS.mla + PARAMS.denseFfnBlk;
PARAMS.moeBlock = PARAMS.mla + PARAMS.moeFfnBlk;
PARAMS.total = 2 * PARAMS.embed + PARAMS.finalNorm
  + A.denseLayers * PARAMS.denseBlock + (A.layers - A.denseLayers) * PARAMS.moeBlock;
// ACTIVE per token, following DeepSeek's own corrected accounting
// (README_WEIGHTS.md: "Activation parameters: 36.6B (including 0.9B for the
// output Head)"): the INPUT EMBEDDING IS NOT COUNTED (a lookup, no FLOPs);
// the untied output head is counted in full; only the MoE FFN shrinks
// (top-k routed + the shared expert fire). Conventions differ across the
// literature (Qwen reports activated and activated-non-embedding separately);
// we reproduce the model author's.
PARAMS.activeMoeFfnBlk = (A.topk + A.sharedExperts) * expert + router + A.hidden;
PARAMS.activeMoeBlock = PARAMS.mla + PARAMS.activeMoeFfnBlk;
PARAMS.activeEmbed = 0;
PARAMS.activeTotal = PARAMS.embed + PARAMS.finalNorm
  + A.denseLayers * PARAMS.denseBlock + (A.layers - A.denseLayers) * PARAMS.activeMoeBlock;
