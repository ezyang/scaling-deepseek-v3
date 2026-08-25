// Named parameter quantities, derived once from the architecture and shared
// by every surface that displays a count (the tally, the plan strip, the
// kindtabs) — so they can't drift apart. EXACT: RMSNorm weights included.
import { DSV3 as A } from './model.js';

const qkvDown = A.hidden * (A.qRank + A.kvRank + A.qkRope);
const qUp = A.qRank * A.heads * (A.qkNope + A.qkRope);
const kvUp = A.kvRank * A.heads * (A.qkNope + A.vHead);
const oProj = A.heads * A.vHead * A.hidden;
const mlaNorms = A.hidden + A.qRank + A.kvRank;      // norm1 + the two latent norms
const expert = 3 * A.hidden * A.moeInter;            // routed and shared are the same shape
const router = A.hidden * A.routedExperts;
const denseFfn = 3 * A.hidden * A.denseInter;

export const PARAMS = {
  embed: A.hidden * A.vocab,                          // untied: the lm head is another copy
  qkvDown, qUp, kvUp, oProj,
  attnQkv: qkvDown + qUp + kvUp,
  attnOut: oProj,
  mlaNorms,
  mla: qkvDown + qUp + kvUp + oProj + mlaNorms,
  router, expert, denseFfn,
  normsBlk: mlaNorms + A.hidden,                      // + norm2
  denseFfnBlk: denseFfn + A.hidden,                   // FFN half incl. norm2
  moeFfnBlk: (A.routedExperts + A.sharedExperts) * expert + router + A.hidden,
  finalNorm: A.hidden,
};
PARAMS.denseBlock = PARAMS.mla + PARAMS.denseFfnBlk;
PARAMS.moeBlock = PARAMS.mla + PARAMS.moeFfnBlk;
PARAMS.total = 2 * PARAMS.embed + PARAMS.finalNorm
  + A.denseLayers * PARAMS.denseBlock + (A.layers - A.denseLayers) * PARAMS.moeBlock;
