// @page studies/02-hopper-memory.html
// the beat deck: explicit advance, animated transitions, baseline = the last
// REAL step, hypothetical steps dashed + auto-reverted by full configs
const deck = () => document.getElementById('fitdeck');
const L = () => deck().querySelector('dsv3-layer');
const cap = () => deck().querySelector('.deck-cap').textContent;
const bar = () => L().querySelector('.lv-bar').textContent;
const next = () => { deck().querySelector('.deck-nav button:last-of-type').click(); };

T.check('step 1 renders the debt', cap().includes('the debt') && bar().includes('total8.65 TiB'), '');
T.check('back disabled at step 1', deck().querySelector('.deck-nav button').disabled, '');
next(); await T.tick(1000);
T.check('step 2: pure DP — nothing moves: no ghosts, no badges (delta-only)',
  L().querySelectorAll('.lv-bar rect[stroke-dasharray]').length === 0
  && !bar().includes('▼') && !bar().includes('▲') && !bar().includes('saved:'), '');
// the per-slide config READOUT: knob groups render, values in ink, disabled
T.check('config readout panel shows and is inert', L().querySelector('.stp[data-knob="pp"] select.v').value === '1'
  && L().querySelector('.stp[data-knob="pp"] select.v').disabled
  && L().querySelector('.stp[data-knob="zero"]'), '');
const deckBarY = L().querySelector('.lv-bar').getBoundingClientRect().top;
next(); await T.tick(1000);
T.check('step 3: ZeRO-1 ▼×2048, all bars still shown', L().zero === 1 && bar().includes('▼×2048')
  && bar().includes('weights1.22 TiB') && bar().includes('· dispatched tokens'), bar().slice(0, 120));
next(); await T.tick(1000);
T.check('step 4: hypothetical ZeRO-2 — dashed card, tag in the nav row', L().zero === 2
  && getComputedStyle(L().querySelector('.lv')).borderTopStyle === 'dashed'
  && deck().querySelector('.deck-hyp').textContent.includes('unsharded')
  && !L().querySelector('.lv-hyptag'), '');
// the chart's y position NEVER changes slide to slide
T.check('chart y is slide-invariant', Math.abs(L().querySelector('.lv-bar').getBoundingClientRect().top
  - deckBarY) < 1, '');
next(); await T.tick(1000);
T.check('step 5: EP64 — the hypothetical reverted, solid card', L().zero === 1 && L().ep === 64
  && getComputedStyle(L().querySelector('.lv')).borderTopStyle === 'solid', `zero ${L().zero}`);
T.check('step 5 baseline is ZeRO-1 (skips the hypothetical)',
  L()._pinCfg.state.zero === 1 && L()._pinCfg.state.ep === 1, '');
// visual audit holds mid-deck (badges, ghosts, decomposition)
const { auditFitCharts } = await import('/src/audit.js');
T.check('audit clean at step 5', auditFitCharts(deck()).findings.length === 0,
  auditFitCharts(deck()).findings[0]);
// arrow keys navigate; backward snaps
deck().dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true })); await T.tick(200);
T.check('ArrowLeft goes back (animated)', L().zero === 2, '');
deck().dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true })); await T.tick(1000);
// walk to the end: the fit
for (let i = 0; i < 6; i++) { next(); await T.tick(1000); }
T.check('final step: fits with headroom', cap().includes('destination')
  && L().querySelectorAll('.lv-bar text').length > 20, '');
T.check('next disabled at the end', deck().querySelector('.deck-nav button:last-of-type').disabled, '');
T.check('audit clean at the end', auditFitCharts(deck()).findings.length === 0, '');
T.done();
