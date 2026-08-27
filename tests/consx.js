// @page studies/02-hopper-memory.html
// consolidated ×58: cumulative grows everything (strips, chips, totals)
const layer = () => document.getElementById('all-diagram');
const dia = () => layer().querySelector('.lv-scroll svg');
const dims = () => layer().querySelector('g[data-op="ffn_gate_up"] text.dims:not([text-anchor])')?.textContent.trim();
const box = () => layer().querySelector('g[data-op="ffn_gate_up"] rect.box');
const total = () => null;   // the total line was removed by design
const cumBtn = () => [...layer().parentElement.querySelectorAll('button')].find(b => b.textContent === 'per block' || b.textContent.includes('blocks'));
T.check('cum button present', !!cumBtn(), '');
T.log('per-block dims / box h / svg h', `${dims()} / ${box().getAttribute('height')} / ${dia().getAttribute('height')}`);
const h0 = +dia().getAttribute('height');
cumBtn().click(); await T.tick(800);
T.log('cum dims / box h / svg h', `${dims()} / ${box().getAttribute('height')} / ${dia().getAttribute('height')}`);
T.check('gate/up = 5.55 TiB (14 B x 58)', dims() === '5.55 TiB', dims());
T.check('box grew huge', +box().getAttribute('height') > 2000, box().getAttribute('height'));
T.check('svg much taller', +dia().getAttribute('height') > h0 + 3000, dia().getAttribute('height'));

// x0 chip scaled
const x0 = [...dia().querySelectorAll('text.tsave')].find(t => t.textContent.includes('x0'));
T.log('x0 chip', x0?.textContent);
T.check('x0 chip ×58 (3.2 GiB)', x0?.textContent.includes('3.2 GiB'), x0?.textContent);
// halfbox grew
const qup = layer().querySelector('g[data-op="q_up"] rect.box');
T.log('q_up box h', qup.getAttribute('height'));
T.check('q_up halfbox grew', +qup.getAttribute('height') > 60, qup.getAttribute('height'));
// back
cumBtn().click(); await T.tick(800);
T.check('back to per block', dims() === '98.0 GiB', dims());
T.check('svg back', Math.abs(+dia().getAttribute('height') - h0) < 5, dia().getAttribute('height'));
T.done();
