// @page studies/02-hopper-memory.html
// pipeline-schedule strip: always DualPipeV — follows the local diagram's
// PP / sched / stage; gutter stage picking, lane pinning, panning
const w = () => document.querySelector('dsv3-pp-schedule');
const l = () => document.getElementById('local-diagram');
const cells = (ph) => [...w().querySelectorAll(`rect[data-cell^="${ph}"]`)];
const stps = () => ['gpus', 'pp', 'sched', 'ep', 'zero']
  .map(k => l().parentElement.querySelector(`.stp[data-knob="${k}"]`));

// pin the drawn-microbatch knob to 'auto' (= depth+4, steady state reached)
{
  const msel = w().querySelector('[data-knob="mb"]');
  msel.value = 'auto'; msel.dispatchEvent(new Event('change')); await T.tick(150);
}
const pp = l().pp;
T.log('pp', pp);
T.check('opens on the official DualPipeV program (W cells drawn)', cells('W').length > 0, cells('W').length);
T.check('no schedule-family or VPP knobs on the strip', !w().querySelector('[data-knob="prog"]')
  && !w().querySelector('[data-knob="vpp"]') && !w().querySelector('[data-knob="fold"]'), '');
// highlight row follows the layer's stage
const hl = () => w().querySelector('rect.stghl');
const rowY = (s) => s * 16;
T.check('tinted row = selected stage', +hl().getAttribute('y') === rowY(l().stage), hl().getAttribute('y'));
// flip the schedule knob to ×1 mb: single microbatch, F wave + B wave
const schedBtns = stps().map(s => [...s.querySelectorAll('button')]).flat();
const oneBtn = schedBtns.find(b => b.textContent.includes('1 mb'));
oneBtn.click(); await T.tick(400);
T.check('×1 mb: one F per virtual stage (2·PP chunks)', cells('F').length === 2 * l().pp, cells('F').length);
T.check('×1 mb: one B per virtual stage', cells('B').length === 2 * l().pp, cells('B').length);
const hdr = w().querySelector('.hd').textContent;
T.check('header describes the wave', hdr.includes('wave'), hdr);
// in-flight section: one lane per concurrently-held stash, peak = the law
{
  const back1 = [...w().querySelectorAll('.stp[data-knob="sched"] button')].find(b => b.textContent === 'DualPipeV');
  back1.click(); await T.tick(500);
  const expect = l().pp + 0.5;   // uniform PP+½ under the V
  T.check('peak label matches the law', +w().querySelector('text[data-peak]').dataset.peak === expect, '');
  T.check('stash bars carry F and B cells', w().querySelectorAll('rect[data-stash^="B"]').length
    === w().querySelectorAll('rect[data-stash^="F"]').length, '');
  // clicking a stash lights exactly its two schedule ops (same mb AND v);
  // hover does nothing (too distracting) — click again releases
  const lane7 = [...w().querySelectorAll('g.lane')].find(g => g.dataset.mb === '7');
  const cellsOf = (mb) => [...w().querySelectorAll(`rect[data-cell][data-mb="${mb}"]`)];
  lane7.dispatchEvent(new MouseEvent('mouseover', { bubbles: true })); await T.tick(50);
  T.check('hover alone does nothing', cellsOf(3).every(r => r.style.opacity === ''), '');
  lane7.dispatchEvent(new MouseEvent('click', { bubbles: true })); await T.tick(50);
  const lit = cellsOf(7).filter(r => r.style.opacity === '');
  T.check('click: exactly the stash\'s F and B stay lit', lit.length === 2
    && lit.every(r => r.dataset.v === lane7.dataset.v), lit.length);
  T.check('click: everything else dims', cellsOf(3).every(r => r.style.opacity === '0.22')
    && lane7.classList.contains('pin'), '');
  T.check('hitbox spans the row band', +lane7.querySelector('rect[fill="transparent"]').getAttribute('height') > 10, '');
  lane7.dispatchEvent(new MouseEvent('click', { bubbles: true })); await T.tick(50);
  T.check('second click releases', cellsOf(3).every(r => r.style.opacity === '')
    && !lane7.classList.contains('pin'), '');
}
// ---- the widget's own replicated controls drive the LAYER (two-way link)
const wctl = () => w().querySelector('.pargrp');
const wstp = () => wctl().querySelector('.stp');
T.check('widget wears the pipeline knob group', !!wctl() && wctl().textContent.includes('PP'), '');
wstp().querySelectorAll('button')[0].click(); await T.tick(600);   // widget's PP − (8 → 1: {1,8} only)
T.check('widget PP− drops the layer to PP1', l().pp === 1, l().pp);
T.check('PP1 draws the trivial single-row strip', w().querySelectorAll('rect.stghit').length === 1, '');
wstp().querySelectorAll('button')[1].click(); await T.tick(600);   // back to PP8
T.check('back to PP8: the official program again', l().pp === 8 && cells('W').length > 0, '');
// the sX axis is the stage picker: click a gutter row
T.check('no stage dropdown on the strip (axis picks)', !wctl().querySelector('[data-knob="stage"]'), '');
w().querySelector('rect.stghit[data-stage="0"]').dispatchEvent(new MouseEvent('click', { bubbles: true }));
await T.tick(600);
T.check('clicking the s0 gutter moves the layer', l().stage === 0, l().stage);
T.check('tinted row follows', +w().querySelector('rect.stghl').getAttribute('y') === 0, '');
// after a click the strip holds focus: arrows walk the stages
const scr = w().querySelector('.scroll');
scr.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowDown', bubbles: true })); await T.tick(500);
T.check('ArrowDown steps to s1', l().stage === 1, l().stage);
scr.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowUp', bubbles: true })); await T.tick(500);
scr.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowUp', bubbles: true })); await T.tick(200);
T.check('ArrowUp clamps at s0', l().stage === 0, l().stage);
T.check('overscroll-x contained', getComputedStyle(scr).overscrollBehaviorX === 'none', '');
// Perfetto-style panning: shift+drag scrubs the timeline; a shift-click
// never picks a stage (it means pan)
const stage0 = l().stage;
const mev = (type, x, extra = {}) => scr.dispatchEvent(new MouseEvent(type, { clientX: x, bubbles: true, shiftKey: true, ...extra }));
const gutter = scr.querySelector('rect.stghit[data-stage="1"]');
mev('mousedown', 400);
T.check('shift-drag arms the pan (grabbing cursor)', scr.classList.contains('panning'), '');
document.dispatchEvent(new MouseEvent('mousemove', { clientX: 250, bubbles: true, shiftKey: true }));
await T.tick(30);
T.check('dragging left pans right', scr.scrollLeft > 60, scr.scrollLeft);   // clamped by runway
document.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
gutter.dispatchEvent(new MouseEvent('click', { bubbles: true, shiftKey: true })); await T.tick(200);
T.check('shift-click does not pick a stage', l().stage === stage0 && !scr.classList.contains('panning'), l().stage);
scr.scrollLeft = 0;
T.done();
