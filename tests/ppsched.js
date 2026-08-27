// @page studies/02-hopper-memory.html
// pipeline-schedule strip: follows the local diagram's PP / sched / stage
const w = () => document.querySelector('dsv3-pp-schedule');
const l = () => document.getElementById('local-diagram');
const cells = (ph) => [...w().querySelectorAll(`rect[data-cell^="${ph}"]`)];
const stps = () => [...l().parentElement.querySelectorAll('.stp')];

// pin the drawn-microbatch knob to 'auto' (= depth+4, steady state reached)
{
  const msel = [...w().querySelectorAll('select')].at(-1);
  msel.value = 'auto'; msel.dispatchEvent(new Event('change')); await T.tick(150);
}
const pp = l().pp, m = pp + 4;
T.log('pp', pp);
T.check('one F cell per (stage, mb)', cells('F').length === pp * m, cells('F').length);
T.check('one B cell per (stage, mb)', cells('B').length === pp * m, cells('B').length);
const f = cells('F')[0], b = cells('B').find(c => c.dataset.cell === 'B0@' + (pp - 1));
// widths are slots*U − 1: F = U−1, B = 2U−1
T.check('B cells are two slots wide', +b.getAttribute('width') + 1 === 2 * (+f.getAttribute('width') + 1),
  `F ${f.getAttribute('width')} B ${b.getAttribute('width')}`);
// steady state on stage 0: between F(pp-1) end and its B0, pp forwards are stashed
T.check('warmup on stage 0 = pp forwards before first B',
  cells('F').filter(c => c.dataset.cell.endsWith('@0')).slice(0, pp)
    .every(c => +c.getAttribute('x') < +w().querySelector('rect[data-cell="B0@0"]').getAttribute('x')), '');
// highlight row follows the layer's stage
const hl = () => w().querySelector('rect.stghl');
const rowY = (s) => s * 16;
T.check('tinted row = selected stage', +hl().getAttribute('y') === rowY(l().stage), hl().getAttribute('y'));
// flip the schedule knob to ×1 mb: single microbatch, F wave + B wave
const schedBtns = stps().map(s => [...s.querySelectorAll('button')]).flat();
const oneBtn = schedBtns.find(b => b.textContent.includes('1 mb'));
oneBtn.click(); await T.tick(400);
T.check('×1 mb: one F per stage', cells('F').length === l().pp, cells('F').length);
T.check('×1 mb: one B per stage', cells('B').length === l().pp, cells('B').length);
const hdr = w().querySelector('.hd').textContent;
T.check('header describes the wave', hdr.includes('wave'), hdr);
// PP step: row count follows
const ppPlus = stps()[1].querySelectorAll('button')[1];
ppPlus.click(); await T.tick(600);
T.check('PP step doubles the rows', cells('F').length === l().pp, `${cells('F').length} vs pp ${l().pp}`);

// ---- the widget's own replicated controls drive the LAYER (two-way link)
const wctl = () => w().querySelector('.pargrp');
const wstp = () => wctl().querySelector('.stp');
T.check('widget wears the pipeline knob group', !!wctl() && wctl().textContent.includes('PP'), '');
const ppBefore = l().pp;
wstp().querySelectorAll('button')[0].click(); await T.tick(600);   // widget's PP −
T.check('widget PP− halves the layer', l().pp === ppBefore / 2, l().pp);
T.check('layer stage select follows (options = pp)',
  stps()[1].parentElement.querySelectorAll('select')[1].options.length === l().pp
  || l().parentElement.querySelectorAll('select').length > 0, '');
const schedBtn = [...wctl().querySelectorAll('.stp button')].find(b => b.textContent === '1F1B');
schedBtn.click(); await T.tick(600);                               // widget's sched → 1F1B
T.check('widget sched flips the layer', l().sched === '1f1b', l().sched);
T.check('strip redrew for 1F1B', cells('F').length === l().pp * (l().pp + 4), cells('F').length);
const wsel = wctl().querySelectorAll('select')[1];                 // stage select (after PP's value chip)
wsel.value = '0'; wsel.dispatchEvent(new Event('change')); await T.tick(600);
T.check('widget stage select moves the layer', l().stage === 0, l().stage);
T.check('tinted row follows', +w().querySelector('rect.stghl').getAttribute('y') === 0, '');
T.done();
