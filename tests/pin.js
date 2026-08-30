// @page studies/02-hopper-memory.html
// save cluster: save locks a baseline (deltas vs it); reset returns TO the
// save; reset all is factory. Save box lives top right.
const layer = () => document.getElementById('local-diagram');
const barTxt = () => layer().querySelector('.lv-bar')?.textContent ?? '';
const stps = () => ['gpus', 'pp', 'sched', 'ep', 'zero']
  .map(k => layer().parentElement.querySelector(`.stp[data-knob="${k}"]`));
const stepBtn = (i, dir) => stps()[i].querySelectorAll('button')[dir < 0 ? 0 : 1];
const box = () => layer().querySelector('.savebox');
const btn = (t2) => [...box().querySelectorAll('button')].find(b => b.textContent === t2);
T.check('save box top right', !!box() && !!btn('save') && !!btn('reset all'), '');
T.check('reset present but disabled before saving', btn('reset')?.disabled === true, '');
btn('save').click(); await T.tick();
T.check('saved label shows config', barTxt().includes('saved: EP64·PP8'), barTxt().slice(-90));
T.check('reset enabled after saving', btn('reset')?.disabled === false, '');
// change EP; deltas appear; reset returns to the save
stepBtn(3, -1).click(); await T.tick(700);
T.check('delta badge vs save (▲×1.4 blended weights)', barTxt().includes('▲×1.4'), barTxt().slice(60, 220));
const gu = layer().querySelector('g[data-op="ffn_gate_up"] text.dims:not([text-anchor])')?.textContent ?? '';
T.check('diagram numbers wear delta badges', gu.includes('×1.8'), gu);
btn('reset').click(); await T.tick(700);
T.check('reset returns to the saved EP', stps()[3].querySelector('select.v').value === '64', '');
T.check('badges clear at the save point', !barTxt().includes('▲'), '');
// re-save locks in a new baseline
stepBtn(3, -1).click(); await T.tick(700);
btn('save').click(); await T.tick();
T.check('re-save updates the label', barTxt().includes('saved: EP32'), barTxt().slice(-90));
// reset all: factory, save cleared
btn('reset all').click(); await T.tick(400);
T.check('factory: PP1/EP1/ZeRO-off, no save', stps()[3].querySelector('select.v').value === '1' && stps()[1].querySelector('select.v').value === '1' && !barTxt().includes('saved:'), '');
T.done();
