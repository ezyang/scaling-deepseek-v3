// @page studies/scratch-globals.html
// optim diagram: margin legend rows SOLO components; numbers track squares
const layer = () => document.getElementById('opt-diagram');
const dims = () => layer().querySelector('g[data-op="ffn_gate_up"] text.dims:not([text-anchor])')?.textContent.trim();
const dia = () => layer().querySelector('.lv-scroll svg');
const nBlue = () => dia().querySelectorAll('rect[fill="#2a78d6"]').length;
const nGreen = () => dia().querySelectorAll('rect[fill="#1baf7a"]').length;
const firstGreenY = () => Math.min(...[...dia().querySelectorAll('rect[fill="#1baf7a"]')].map(r => +r.getAttribute('y')));
const rows = () => [...layer().closest('.anat-grid').querySelectorAll('.anp-leg .row')];
T.check('two legend rows', rows().length === 2, rows().length);
T.check('both: 70.0 GiB', dims() === '70.0 GiB', dims());
const gy0 = firstGreenY();
// solo weights
rows()[0].click(); await T.tick(600);
T.check('solo weights: 14.0 GiB', dims() === '14.0 GiB', dims());
T.check('no green squares', nGreen() === 0, nGreen());
T.check('weights row on, optim row off', !rows()[0].classList.contains('off') && rows()[1].classList.contains('off'), '');
// solo optimizer
rows()[1].click(); await T.tick(600);
T.check('solo optim: 56.0 GiB', dims() === '56.0 GiB', dims());
T.check('blue gone, green slid up', nBlue() === 0 && firstGreenY() <= gy0, `${nBlue()}/${firstGreenY()}`);
const plan = layer().closest('.anat-grid').querySelector('dsv3-anatomy-plan');
T.check('plan optim-only 85.7 GiB each', plan.querySelector('g[data-op="block-moe"] text.dims')?.textContent.startsWith('85.7 GiB'), '');
// solo again -> all back
rows()[1].click(); await T.tick(600);
T.check('restored both: 70.0 GiB', dims() === '70.0 GiB', dims());
// lm head dims nit: no "7168 →" in the param-lens plans
const lmDims = (p) => p.querySelector('g[data-op="lm_head"] text.dims')?.textContent ?? '';
const plans = [...document.querySelectorAll('dsv3-anatomy-plan')].slice(0, 2);
T.check('no 7168→ in param-lens plans', plans.every(p => !lmDims(p).includes('→')), '');
T.done();
