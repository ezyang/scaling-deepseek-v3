// @page studies/02-hopper-memory.html
// fit chart ruler: drag-only spans; any outside click dismisses
const layer = () => document.getElementById('local-diagram');
const bar = () => layer().querySelector('.lv-bar');
const svg = () => bar().querySelector('svg');
const scrub = () => bar().querySelector('.scrub');
const shown = () => bar().querySelector('.lv-ruler').style.display !== 'none';
const lab = () => bar().querySelector('.lv-ruler-lab')?.textContent ?? '';
const r = svg().getBoundingClientRect();
const k = r.width / 800, perDbl = 650 / 16 * k;
const capX = r.left + (110 + (Math.log2(80 * 2 ** 30) - 28) / 16 * 650) * k;
const mk = (type, x, y = r.top + 30) => new MouseEvent(type, { bubbles: true, clientX: x, clientY: y });
// shaded infeasible region exists; no black capacity line
T.check('infeasible region shaded', !!svg().querySelector('rect[opacity="0.07"]'), '');
T.check('no capacity line', ![...svg().querySelectorAll('line')].some(l => l.getAttribute('stroke') === '#0b0b0b' && +l.getAttribute('stroke-width') >= 1.2), '');
// plain click: NOTHING shows
scrub().dispatchEvent(mk('mousedown', capX - 4 * perDbl));
document.dispatchEvent(mk('mouseup', capX - 4 * perDbl));
await T.tick(60);
T.check('plain click shows nothing', !shown(), '');
// drag 4 doublings: span x16
scrub().dispatchEvent(mk('mousedown', capX - 6 * perDbl));
document.dispatchEvent(mk('mousemove', capX - 2 * perDbl));
document.dispatchEvent(mk('mouseup', capX - 2 * perDbl));
await T.tick(60);
T.log('span label', lab());
T.check('drag span reads x16 with endpoints', lab().startsWith('×16') && lab().includes('→'), lab());
// clicking a legend row dismisses AND solos
const row0 = layer().querySelector('.lv-bar g[data-prop]');
row0.dispatchEvent(mk('mousedown', r.left + 20));
await T.tick(700);
T.check('legend click dismisses ruler', !shown(), '');
row0.dispatchEvent(mk('mousedown', r.left + 20)); await T.tick(700);   // un-solo
// re-span, then click elsewhere on the page dismisses
scrub().dispatchEvent(mk('mousedown', capX - 6 * perDbl));
document.dispatchEvent(mk('mousemove', capX - 2 * perDbl));
document.dispatchEvent(mk('mouseup', capX - 2 * perDbl));
await T.tick(60);
T.check('span shown again', shown(), '');
document.body.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, clientX: 5, clientY: 5 }));
await T.tick(60);
T.check('outside click dismisses', !shown(), '');
T.done();
