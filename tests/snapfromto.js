// @page studies/scratch-bars.html
// from/to snapshot machinery, hosted off-essay so 02's editorial choices
// can't orphan the coverage: from = saved baseline (ghosts), to = live bars
// with badges, solo picks the row, sandbox link loads the scenario
const snap = () => document.getElementById('snap-zero');
const bar = () => snap().querySelector('.lv-bar');
const txt = () => bar().textContent;
T.check('snapshot renders the fit chart only', !!bar().querySelector('svg')
  && !snap().querySelector('.lv-scroll'), '');
T.check('baseline label names the from config', txt().includes('saved: EP64·PP8') && txt().includes('ZeRO-off'), '');
// zero-1: dense optim /DP=256, expert optim /EDP=4 — blended ▼×14.6 on
// the V's peak stage (expert-heavy, so the expert share dominates)
T.check('optimizer shrink badge ▼×14.6', txt().includes('▼×14.6'), txt().slice(0, 160));
T.check('ghost bars drawn', bar().querySelectorAll('rect[stroke-dasharray]').length > 0, '');
// solo="optim": the off components' rows are GONE; total keeps the full mass
T.check('solo: no weights/grads/acts rows at all', !txt().includes('weights')
  && !txt().includes('gradients') && txt().includes('optimizer states'), '');
T.check('total row present', txt().includes('total'), '');
// the sandbox link loads the scenario into the target widget
const link = [...snap().querySelectorAll('a')].find((a) => a.textContent.includes('full widget'));
T.check('sandbox link present', !!link, '');
const full = document.getElementById('bars-all');
link.click(); await T.tick(600);
T.check('target took the scenario', full.zero === 1 && full.pp === 8 && full.ep === 64, `${full.zero}/${full.pp}/${full.ep}`);
T.check('target carries the baseline save (blended optim badge)', full._pinCfg?.state?.zero === 0
  && (full.querySelector('.lv-bar')?.textContent ?? '').includes('▼×14.6'), full._pinCfg?.state?.zero);
T.done();
