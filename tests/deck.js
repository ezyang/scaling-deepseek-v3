// @page studies/02-hopper-memory.html
// the beat deck: explicit advance, animated transitions both ways, baseline
// = the last REAL step, hypothetical steps dashed + auto-reverted
const deck = () => document.getElementById('fitdeck');
const L = () => deck().querySelector('dsv3-layer');
const cap = () => deck().querySelector('.deck-cap').textContent;
const bar = () => L().querySelector('.lv-bar').textContent;
const next = () => { deck().querySelector('.deck-next').click(); };

T.check('step 1 renders the debt (+DP folded in)', cap().includes('the debt') && cap().includes('replica group')
  && bar().includes('total8.65 TiB'), '');
T.check('back/first disabled at step 1', deck().querySelector('.deck-prev').disabled
  && deck().querySelector('.deck-first').disabled, '');
T.check('config readout panel shows and is inert', L().querySelector('.stp[data-knob="pp"] select.v').value === '1'
  && L().querySelector('.stp[data-knob="pp"] select.v').disabled
  && L().querySelector('.stp[data-knob="zero"]'), '');
const deckBarY = L().querySelector('.lv-bar').getBoundingClientRect().top;
next(); await T.tick(1000);
T.check('step 2: ZeRO-1 ▼×2048, all bars still shown', L().zero === 1 && bar().includes('▼×2048')
  && bar().includes('weights1.22 TiB') && bar().includes('· dispatched tokens'), bar().slice(0, 120));
next(); await T.tick(1000);
T.check('step 3: EP64', L().zero === 1 && L().ep === 64, `zero ${L().zero}`);
next(); await T.tick(1000);   // PP16
next(); await T.tick(1000);   // ×1mb dream — the remaining hypothetical
T.check('step 5: hypothetical ×1mb — dashed card, tag in the nav row', L().sched === 'one'
  && getComputedStyle(L().querySelector('.lv')).borderTopStyle === 'dashed'
  && deck().querySelector('.deck-hyp').textContent.includes('idle')
  && !L().querySelector('.lv-hyptag'), '');
T.check('chart y is slide-invariant', Math.abs(L().querySelector('.lv-bar').getBoundingClientRect().top
  - deckBarY) < 1, '');
next(); await T.tick(1000);   // DPV
T.check('step 6: DPV — the hypothetical reverted, solid card', L().sched === '1f1b' && L().vpp === 2
  && getComputedStyle(L().querySelector('.lv')).borderTopStyle === 'solid', L().sched);
T.check('step 6 baseline is PP16/1F1B (skips the hypothetical)',
  L()._pinCfg.state.sched === '1f1b' && L()._pinCfg.state.vpp === 1, '');
const { auditFitCharts } = await import('/src/audit.js');
T.check('audit clean at step 6', auditFitCharts(deck()).findings.length === 0,
  auditFitCharts(deck()).findings[0]);
deck().dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true })); await T.tick(1000);
T.check('ArrowLeft goes back (animated)', L().sched === 'one', '');
// jump to the end
deck().querySelector('.deck-last').click(); await T.tick(1200);
T.check('›| jumps to the destination', cap().includes('destination')
  && deck().querySelector('.deck-next').disabled && deck().querySelector('.deck-last').disabled, '');
T.check('audit clean at the end', auditFitCharts(deck()).findings.length === 0, '');
// the sub-row SET is fixed across slides: stashes killed by AC/fp8 keep
// their labeled zero rows (their bars animated down; rows never vanish)
T.check('dead buckets remain as zero rows at the end', bar().includes('norm outs0')
  && bar().includes('swiglu out0'), '');
T.check('sub-row count is slide-invariant', L().querySelectorAll('.lv-bar text[data-role^="val:part:"]').length === 18, 
  L().querySelectorAll('.lv-bar text[data-role^="val:part:"]').length);
// and |‹ back to the start
deck().querySelector('.deck-first').click(); await T.tick(1200);
T.check('|‹ jumps home, no baseline', cap().includes('the debt') && !bar().includes('▼'), '');
T.done();
