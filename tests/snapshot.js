// @page studies/02-hopper-memory.html
// snapshot story beats: from = saved baseline (ghosts), to = live bars with
// badges, solo picks the row, zero interactivity, sandbox link loads the
// scenario into the full widget
const snap = () => document.querySelector('dsv3-layer[snapshot]');
const bar = () => snap().querySelector('.lv-bar');
const txt = () => bar().textContent;

T.check('snapshot renders the fit chart only', !!bar().querySelector('svg')
  && !snap().querySelector('.lv-scroll'), '');
T.check('no knobs anywhere', !snap().querySelector('.stp') && !snap().querySelector('.savebox'), '');
T.check('chart is pointer-inert', getComputedStyle(bar()).pointerEvents === 'none', '');
T.check('baseline label names the from config', txt().includes('saved: EP64·PP16') && txt().includes('ZeRO-off'), '');
// zero-1 shards optimizer over DP=128: a bold ▼×128 badge on the optim row
T.check('optimizer shrink badge ▼×128', txt().includes('▼×128'), txt().slice(0, 160));
T.check('ghost bars drawn', bar().querySelectorAll('rect[stroke-dasharray]').length > 0, '');
// solo="optim": weights/grads rows dimmed name-only (no value), total stacked
T.check('solo: weights row has no value', txt().includes('weightsgradients (fp32)optimizer'), '');
T.check('total row present', txt().includes('total'), '');
// snapshots keep no URL state
T.check('no snapshot URL state', !location.hash.includes('l:layer'), location.hash);
// the sandbox link loads the scenario into the full widget
const link = [...snap().querySelectorAll('a')].find(a => a.textContent.includes('full widget'));
T.check('sandbox link present', !!link, '');
const full = document.getElementById('local-diagram');
link.click(); await T.tick(600);
T.check('full widget took the scenario', full.zero === 1 && full.pp === 16 && full.ep === 64, `${full.zero}/${full.pp}/${full.ep}`);
T.check('full widget carries the baseline save (blended optim badge)', full._pinCfg?.state?.zero === 0
  && (full.querySelector('.lv-bar')?.textContent ?? '').includes('▼×4.6'), full._pinCfg?.state?.zero);
T.done();
