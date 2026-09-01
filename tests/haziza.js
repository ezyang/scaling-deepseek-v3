// @page studies/02-hopper-memory.html
const layer = () => document.getElementById('local-diagram');
const sheet = document.querySelector('dsv3-sheet');
await T.tick(400);
const trOf = (id) => [...sheet.querySelectorAll('tr')].find(r => r.querySelector('.nm')?.textContent === id);
const rowOf = (id) => trOf(id)?.textContent ?? '';
const valOf = (id) => trOf(id)?.querySelector('td.vl')?.textContent.trim();
// gate/up split
T.check('gate/up splits routed / shared / dense', rowOf('A9').includes('A9a + A9b + A9c')
  && rowOf('A9a').includes('routed experts’ hidden, pre-SwiGLU')
  && rowOf('A9b').includes('shared expert hidden, pre-SwiGLU'), rowOf('A9').slice(0, 60));
T.check('routed = 8/9 of the moe rate; dense MLP zero on this rank',
  valOf('A9a') === '9,412,286,939.43 B'.replace(/x/,'') || +trOf('A9a').querySelector('td.vl').textContent.replace(/[^\d.]/g,'') > 0, valOf('A9a'));
T.check('one kept? and one precision gate all three', rowOf('A9a').includes('R9 ×') && rowOf('A9b').includes('R9 ×')
  && rowOf('A9c').includes('R9 ×') && !!trOf('R9') && !!trOf('B9'), '');
// columns fixed: widths identical before/after a knob change
const widths = () => [...sheet.querySelectorAll('tr:nth-child(2) td')].map(td => td.getBoundingClientRect().width.toFixed(1)).join(',');
const w0 = widths();
const zseg = layer().parentElement.querySelector('.stp[data-knob="zero"]');
[...zseg.querySelectorAll('button')].find(b => b.textContent === 'off').click(); await T.tick(700);
T.check('columns hold their widths across knob changes', widths() === w0, `${w0} vs ${widths()}`);
[...layer().parentElement.querySelector('.stp[data-knob="zero"]').querySelectorAll('button')].find(b => b.textContent === '1').click(); await T.tick(700);
// the Haziza preset
const hz = () => sheet.querySelector('button.hzb');
T.check('Haziza button present, not lit at defaults', !!hz() && getComputedStyle(hz()).backgroundColor === 'rgb(255, 255, 255)', getComputedStyle(hz()).backgroundColor);
hz().click(); await T.tick(800);
T.check('Haziza lights up after applying', getComputedStyle(hz()).backgroundColor === 'rgb(255, 248, 234)', getComputedStyle(hz()).backgroundColor);
T.check('o_proj stash goes bf16, fp8 params on, recipe reads custom',
  layer().matmuls.o_proj === 'bf16' && layer().fp8Params === true
  && layer().parentElement.querySelector('.stp[data-knob="recipe"] button.on')?.textContent === 'custom', '');
const t1 = trOf('T1')?.querySelector('td.vl')?.textContent;
T.log('Haziza T1', t1);
T.check('Haziza T1 pinned', t1 === '65,102,913,728 B', t1);
// clicking the LIT button toggles back to the config you came from
hz().click(); await T.tick(800);
T.check('click again returns to the og config', getComputedStyle(hz()).backgroundColor !== 'rgb(255, 248, 234)'
  && layer().matmuls.o_proj === 'e5m6' && layer().fp8Params === false
  && trOf('T1')?.querySelector('td.vl')?.textContent === '66,296,545,344 B', trOf('T1')?.querySelector('td.vl')?.textContent);
// and forward again (ping-pong)
hz().click(); await T.tick(800);
T.check('ping-pong: forward again', getComputedStyle(hz()).backgroundColor === 'rgb(255, 248, 234)', '');
T.done();
