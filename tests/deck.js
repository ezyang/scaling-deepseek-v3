// @page studies/02-hopper-memory.html
// the beat deck: explicit advance, animated transitions both ways, baseline
// = the last real step, LIVE knobs whose fiddling is a marked detour that
// stepping rewinds first
const deck = () => document.getElementById('fitdeck');
const L = () => deck().querySelector('dsv3-layer');
const cap = () => deck().querySelector('.deck-cap').textContent;
const bar = () => L().querySelector('.lv-bar').textContent;
const next = () => { deck().querySelector('.deck-next').click(); };

// caption prose is under active rewrite — pin structure, not phrasing
T.check('step 1 renders (whole-model debt on the bar)', cap().includes('Step 1')
  && bar().includes('total8.65 TiB'), '');
T.check('back/first disabled at step 1', deck().querySelector('.deck-prev').disabled
  && deck().querySelector('.deck-first').disabled, '');
T.check('knob panel shows the config and is LIVE', L().querySelector('.stp[data-knob="pp"] select.v').value === '1'
  && !L().querySelector('.stp[data-knob="pp"] select.v').disabled
  && L().querySelector('.stp[data-knob="zero"]'), '');
const deckBarY = L().querySelector('.lv-bar').getBoundingClientRect().top;
next(); await T.tick(1000);
T.check('step 2: ZeRO-1 ▼×2048, all bars still shown', L().zero === 1 && bar().includes('▼×2048')
  && bar().includes('weights1.22 TiB') && bar().includes('· dispatched tokens'), bar().slice(0, 120));
next(); await T.tick(1000);
T.check('step 3: EP64', L().zero === 1 && L().ep === 64, `zero ${L().zero}`);
next(); await T.tick(1000);   // PP16 + DualPipeV in one beat
T.check('step 4: PP8 arrives WITH DualPipeV (16 virtual stages)', L().pp === 8 && L().vpp === 2 && L().fold === 'reflect'
  && cap().includes('DualPipeV'), `pp ${L().pp} vpp ${L().vpp}`);
T.check('step 4 baseline is EP64/PP1', L()._pinCfg.state.ep === 64 && L()._pinCfg.state.pp === 1, '');
T.check('chart y is slide-invariant', Math.abs(L().querySelector('.lv-bar').getBoundingClientRect().top
  - deckBarY) < 1, '');
const { auditFitCharts } = await import('/src/audit.js');
T.check('audit clean at step 4', auditFitCharts(deck()).findings.length === 0,
  auditFitCharts(deck()).findings[0]);
// ---- the DETOUR: fiddle a knob mid-slide — tag appears, caption detaches,
// and the next step REWINDS to the slide's config before its own delta
L().querySelector('.stp[data-knob="ep"]').querySelectorAll('button')[0].click(); await T.tick(400);
T.check('fiddling marks a detour (tag + reset button)', L().ep === 32
  && deck().querySelector('.deck-mod').textContent.includes('detour')
  && deck().querySelector('.deck-rst').style.display === ''
  && deck().querySelector('.deck-cap').style.opacity === '0.55', '');
// the reset button pours back to the slide without advancing
deck().querySelector('.deck-rst').click(); await T.tick(500);
T.check('↩ reset returns to the slide, notice clears', L().ep === 64
  && deck().querySelector('.deck-step').textContent.includes('4 /')
  && deck().querySelector('.deck-mod').textContent === ''
  && deck().querySelector('.deck-rst').style.display === 'none', '');
L().querySelector('.stp[data-knob="ep"]').querySelectorAll('button')[0].click(); await T.tick(400);   // detour again
next(); await T.tick(150);    // rewind phase: back on the slide's config, not yet advanced
T.check('stepping rewinds the detour first', L().ep === 64 && L().pp === 8
  && deck().querySelector('.deck-step').textContent.includes('4 /'), deck().querySelector('.deck-step').textContent);
await T.tick(1000);           // then the real step lands
T.check('…then advances with a clean caption', deck().querySelector('.deck-step').textContent.includes('5 /')
  && L().getAttribute('recompute') === 'dsv3' && deck().querySelector('.deck-mod').textContent === ''
  && deck().querySelector('.deck-cap').style.opacity === '', '');
deck().dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true })); await T.tick(1000);
T.check('ArrowLeft goes back (animated)', L().getAttribute('recompute') === 'none'
  && cap().includes('DualPipeV'), '');
// jump to the end
deck().querySelector('.deck-last').click(); await T.tick(1200);
T.check('end » jumps to the last step', cap().includes('Step 6')
  && deck().querySelector('.deck-next').disabled && deck().querySelector('.deck-last').disabled, '');
T.check('audit clean at the end', auditFitCharts(deck()).findings.length === 0, '');
// zeroed buckets HOLD their rows (0 B) — the chart never resizes as we step
T.check('sub-row count still 19 at the last step (zeroed rows persist)',
  L().querySelectorAll('.lv-bar text[data-role^="val:part:"]').length === 19,
  L().querySelectorAll('.lv-bar text[data-role^="val:part:"]').length);
// the sub-row SET is fixed across slides: stashes killed by AC/fp8 keep
// their labeled zero rows (their bars animated down; rows never vanish)
T.check('dead buckets remain as zero rows at the end', bar().includes('norm outs0')
  && bar().includes('swiglu out0'), '');
T.check('sub-row count is slide-invariant', L().querySelectorAll('.lv-bar text[data-role^="val:part:"]').length === 19,
  L().querySelectorAll('.lv-bar text[data-role^="val:part:"]').length);
// and « start back to the beginning
deck().querySelector('.deck-first').click(); await T.tick(1200);
T.check('« start jumps home, no baseline', cap().includes('Step 1') && !bar().includes('▼'), '');
T.done();
