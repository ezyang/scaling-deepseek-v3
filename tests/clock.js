// @page studies/03-sol.html
// the first picture: one microsecond on each meter, and one elementwise
// kernel priced on the CUDA cores and on HBM; the dtype knob scales the
// bytes and leaves the arithmetic alone
const w = document.querySelector('dsv3-clock');
await T.tick(300);
const ro = () => w.querySelector('.ro').textContent;
const bar = (k) => +w.querySelector(`rect[data-bar="${k}"]`).dataset.true;
const wid = (k) => +w.querySelector(`rect[data-bar="${k}"]`).getAttribute('width');
const sheet = (k) => w.querySelector(`[data-sheet="${k}"]`).textContent;
T.check('sheet: 1 µs = 1.98 GFLOP tensor · 67 MFLOP CUDA · 3.35 MB HBM = 1.68 M bf16 numbers',
  sheet('tensor cores') === '1.98 GFLOP' && /67\.0 MFLOP = 33\.5 M FMA/.test(sheet('CUDA cores')) && sheet('HBM') === '3.35 MB = 1.68 M bf16 numbers', [sheet('tensor cores'), sheet('CUDA cores'), sheet('HBM')].join(' | '));
T.check('bf16 x×2: 29.4 M multiplies = 0.88 µs; read + write 117 MB = 35 µs', Math.abs(bar('compute') - 0.876e-6) < 0.01e-6 && Math.abs(bar('read') + bar('write') - 35.05e-6) < 0.1e-6, [bar('compute'), bar('read'), bar('write')].join(' '));
T.check('readout: ×40 the arithmetic, memory-bound', /×40 the arithmetic/.test(ro()) && /35\.1 µs of HBM time/.test(ro()), ro());
const h0 = w.getBoundingClientRect().height, wr0 = wid('read');
w.querySelector('[data-knob="dtype"] button[data-v="fp8"]').click(); await T.tick(350);
T.check('fp8: byte bars halve, arithmetic bar holds', Math.abs(wid('read') / wr0 - 0.5) < 0.02 && Math.abs(bar('compute') - 0.876e-6) < 0.01e-6 && /×20 the arithmetic/.test(ro()), wid('read') / wr0);
T.check('sheet follows the dtype: 3.35 M fp8 numbers per µs', sheet('HBM') === '3.35 MB = 3.35 M fp8 numbers', sheet('HBM'));
w.querySelector('[data-knob="op"] button[data-v="add"]').click(); await T.tick(350);
T.check('x+y: two reads, one write', Math.abs(bar('read') / bar('write') - 2) < 1e-6 && /x \+ y/.test(ro()) && /adds/.test(ro()), bar('read') / bar('write'));
T.check('knob flips: height reserved (no reflow)', Math.abs(w.getBoundingClientRect().height - h0) < 1, '');
T.check('URL hash carries the state', /c:clock=/.test(decodeURIComponent(location.hash)) && /fp8/.test(location.hash) && /add/.test(location.hash), location.hash);
w.querySelector('[data-knob="dtype"] button[data-v="bf16"]').click(); w.querySelector('[data-knob="op"] button[data-v="mul"]').click(); await T.tick(350);
T.log('widget height', w.getBoundingClientRect().height);
T.done();
