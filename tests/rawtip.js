// @page studies/02-hopper-memory.html
// raw-bytes hover: rounded labels reveal unrounded B values (final sim)
const layer = () => document.getElementById('local-diagram');
const tsp = layer().querySelector('g[data-op="ffn_gate_up"] text.dims tspan[data-raw]');
T.check('gate/up label carries data-raw', !!tsp, '');
T.log('raw', tsp?.dataset.raw);
const tip = () => layer().querySelector('.lv-tip');
const hover = (el2) => {
  const r = el2.getBoundingClientRect();
  el2.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: r.left + r.width / 2, clientY: r.top + r.height / 2 }));
};
hover(tsp); await T.tick(30);
T.check('hover shows unrounded B', tip().style.display === 'block' && /^[\d,]+(\.\d+)? B$/.test(tip().textContent), tip().textContent);
T.check('raw matches data-raw', Math.abs(parseFloat(tip().textContent.replace(/,/g, '')) - +tsp.dataset.raw) < 1, tip().textContent);
// the displayed rounded label rounds the SAME quantity
T.check('7.0 GiB label rounds the raw value', Math.abs(+tsp.dataset.raw / 2 ** 30 - 7.0) < 0.05, (+tsp.dataset.raw / 2 ** 30).toFixed(3));
// fit chart: total value label above the scrub, hover shows data-true
const val = layer().querySelector('.lv-bar text[data-role="val:total"], .lv-bar text[data-role="val:4"]')
  ?? [...layer().querySelectorAll('.lv-bar text[data-role^="val:"]')].pop();
T.log('val role', val?.dataset.role);
val.scrollIntoView({ block: 'center' }); await T.tick(50);
const vr = val.getBoundingClientRect();
const hitEl = document.elementFromPoint(vr.left + 4, vr.top + vr.height / 2);
T.check('value label reachable by pointer (not under scrub)', hitEl === val || val.contains(hitEl), `${hitEl?.tagName} ${hitEl?.getAttribute?.('data-role')}`);
hover(val); await T.tick(30);
T.check('fit value hover shows raw B', tip().style.display === 'block' && tip().textContent.endsWith(' B'), tip().textContent);
// act chip labels too
const chip = layer().querySelector('.lv-scroll text.tsave tspan[data-raw]');
T.check('act chip label carries data-raw', !!chip, '');
hover(chip); await T.tick(30);
T.check('chip hover shows raw B', tip().textContent.endsWith(' B'), tip().textContent);
// moving off hides it
layer().querySelector('.lv-scroll svg').dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: 5, clientY: 5 }));
await T.tick(30);
T.done();
