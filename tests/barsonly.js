// @page studies/scratch-bars.html
// barsonly + knobs: fit chart without the block diagram, disclosed knob subsets
const L = (id) => document.getElementById(id);
const stp = (id, k) => L(id).querySelector(`.stp[data-knob="${k}"]`);

for (const id of ['bars-all', 'bars-pp', 'bars-shard', 'bars-prec']) {
  T.check(`${id}: fit chart renders`, !!L(id).querySelector('.lv-bar svg'), '');
  T.check(`${id}: no block diagram`, !L(id).querySelector('.lv-scroll'), '');
}
T.check('bars-all: all knob groups', !!stp('bars-all', 'gpus') && !!stp('bars-all', 'pp')
  && !!stp('bars-all', 'ep') && !!stp('bars-all', 'zero'), '');
T.check('bars-pp: pipeline only', !!stp('bars-pp', 'pp') && !!stp('bars-pp', 'vpp')
  && !stp('bars-pp', 'ep') && !stp('bars-pp', 'zero') && !stp('bars-pp', 'gpus'), '');
T.check('bars-shard: mesh+zero only', !!stp('bars-shard', 'ep') && !!stp('bars-shard', 'zero')
  && !stp('bars-shard', 'pp'), '');
T.check('bars-prec: no steppers, has precision select', !stp('bars-prec', 'pp')
  && [...L('bars-prec').querySelectorAll('select')].some(s => [...s.options].some(o => o.value === 'dsv3-fp8')), '');
// knobs still drive the chart, and instances are independent
const barTxt = (id) => L(id).querySelector('.lv-bar')?.textContent ?? '';
const before = barTxt('bars-pp'), beforeAll = barTxt('bars-all');
stp('bars-pp', 'pp').querySelectorAll('button')[1].click(); await T.tick(600);
T.check('PP step moves bars-pp\'s chart', barTxt('bars-pp') !== before, '');
T.check('bars-all unaffected (independent state)', barTxt('bars-all') === beforeAll, '');
// barsonly keeps FULL interactivity: the drag ruler measures factors
{
  const scrub = L('bars-pp').querySelector('.scrub');
  const r = scrub.getBoundingClientRect(), y = r.top + r.height / 2;
  scrub.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, clientX: r.left + r.width * 0.2, clientY: y }));
  document.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: r.left + r.width * 0.45, clientY: y }));
  document.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, clientX: r.left + r.width * 0.45, clientY: y }));
  await T.tick(80);
  const rul = L('bars-pp').querySelector('.lv-ruler');
  T.check('barsonly: drag ruler works', getComputedStyle(rul).display === 'block' && /×[\d.]+/.test(rul.textContent), rul.textContent);
  // and solo still mutates
  L('bars-pp').querySelector('.lv-bar [data-prop]').dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
  await T.tick(300);
  T.check('barsonly: gutter solo mutates', !L('bars-pp').showGrads || !L('bars-pp').showActs, '');
}
T.done();
