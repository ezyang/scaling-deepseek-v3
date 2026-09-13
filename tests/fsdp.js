// @page studies/fsdp.html
// the FSDP sweep: rows = sharding degree, memory vs sync; the pincer readout
const w = document.querySelector('dsv3-fsdp');
await T.tick(300);
const rows = () => [...w.querySelectorAll('g[data-row]')].filter((g) => +(g.getAttribute('opacity') ?? 1) > 0.5);
const ro = () => w.querySelector('.ro').textContent;
T.check('12 live rows at 2048 GPUs (G = 1 … 2048), 13 reserved', rows().length === 12 && w.querySelectorAll('g[data-row]').length === 13, rows().length);
T.check('first fit band on G = 128', w.querySelector('rect[data-first]')?.dataset.first === '128', w.querySelector('rect[data-first]')?.dataset.first);
T.check('resting readout states the pincer', /ZeRO-3 first fits at G = 128/.test(ro()) && /InfiniBand/.test(ro()) && /×20\.\d speed-of-light|×20\.\d\)/.test(ro()), ro());
// exact terms ride data-true: G = 2048 ZeRO-3 memory = 14 B/param ÷ 2048
const g2048 = w.querySelector('g[data-row="2048"]');
const memVal = +g2048.querySelector('text[data-memval]').dataset.memval;
T.check('G = 2048: 14 B/param ÷ 2048', Math.abs(memVal - 671026419200 * 14 / 2048) < 1, memVal);
const g1 = w.querySelector('g[data-row="1"]');
T.check('G = 1: no reduce-scatter, no all-gather — but the all-reduce across 2048 replicas remains',
  !g1.querySelector('rect[data-time="ag"]') && +g1.querySelector('rect[data-time="grad"]').dataset.true > 100, g1.querySelector('rect[data-time="grad"]')?.dataset.true);
// hover a row: exact readout
g2048.dispatchEvent(new MouseEvent('mouseover', { bubbles: true })); await T.tick(50);
T.check('hover readout names G and the link', /G = 2048/.test(ro()) && /InfiniBand/.test(ro()), ro());
w.querySelector('svg').dispatchEvent(new MouseEvent('mouseleave')); await T.tick(50);
// ZeRO-1: never fits (weights + grads replicated)
const h0 = w.getBoundingClientRect().height;
w.querySelector('[data-knob="zero"] button[data-v="1"]').click(); await T.tick(350);
T.check('ZeRO-1 never fits', /ZeRO-1 never fits/.test(ro()), ro());
T.check('knob flip: height reserved (no reflow)', Math.abs(w.getBoundingClientRect().height - h0) < 1, '');
w.querySelector('[data-knob="zero"] button[data-v="3"]').click(); await T.tick(350);
// the dense counterfactual threads the needle inside one node
w.querySelector('[data-knob="model"] button[data-v="dense"]').click(); await T.tick(350);
T.check('dense 37B: first fits at G = 8, on NVLink, under speed of light',
  /first fits at G = 8 / .test(ro()) && /NVLink/.test(ro()) && /×0\.\d speed-of-light|\(×0\.\d\)/.test(ro()), ro());
T.check('URL hash carries the state', /f%3Asweep=|f:sweep=/.test(decodeURIComponent(location.hash)) && /dense/.test(location.hash), location.hash);
w.querySelector('[data-knob="model"] button[data-v="dsv3"]').click(); await T.tick(350);
// cluster stepper: fewer GPUs → fewer live rows, more tokens per GPU → longer compute
w.querySelector('[data-knob="gpus"] button[data-dir="-1"]').click(); await T.tick(350);
T.check('1024 GPUs: 11 live rows', rows().length === 11, rows().length);
T.check('compute line doubled with tokens per GPU', Math.abs(+w.querySelector('line[data-comp]').dataset.comp - 7.78) < 0.05, w.querySelector('line[data-comp]').dataset.comp);
w.querySelector('[data-knob="gpus"] button[data-dir="1"]').click(); await T.tick(350);
T.log('widget height', w.getBoundingClientRect().height);
T.done();
