// @page studies/01-deepseek-diagram.html
// 01: compact static boxes + FLAT group label after the sizes toggle
const layer = () => document.getElementById('diagram');
const pl = () => document.getElementById('params-diagram');
const boxes = (l) => [...l.querySelectorAll('rect.box')].map(r => +r.getAttribute('height'));
const grpLabel = (l) => [...l.querySelectorAll('text.grplabel')].map(t => t.textContent).find(t => t.includes('experts'));

T.log('diagram box heights', JSON.stringify([...new Set(boxes(layer()))].sort((a,b)=>a-b)));
T.check('no 38px boxes in static', !boxes(layer()).includes(38), boxes(layer()).join(','));
T.check('no 60px half boxes in static', !boxes(layer()).includes(60), '');
T.check('compact 32px boxes present', boxes(layer()).includes(32), '');
T.log('svg height', layer().querySelector('svg')?.getAttribute('height') ?? layer().querySelector('svg')?.viewBox?.baseVal?.height);

// factored default: group label carries ×256
T.log('grp label (factored)', grpLabel(layer()));
T.check('factored label has ×256', grpLabel(layer())?.includes('×256'), grpLabel(layer()));

// click the sizes button (mini head) → multiplied: ×256 leaves the label
const sizesBtn = [...layer().parentElement.querySelectorAll('button')].find(b => b.textContent.includes('×192') || b.textContent === '24576');
T.check('sizes button found', !!sizesBtn, '');
sizesBtn.click(); await T.tick();
T.log('grp label (multiplied)', grpLabel(layer()));
T.check('multiplied label drops ×256', !grpLabel(layer())?.includes('×256'), grpLabel(layer()));
T.check('multiplied label shows 11.3B total', grpLabel(layer())?.includes('11.3B'), grpLabel(layer()));
// box dims now show the folded 7.5B
const gu = layer().querySelector('g[data-op="ffn_gate_up"] text.dims:not([text-anchor])')?.textContent;
T.log('gate/up dims multiplied', gu);
T.check('gate/up shows 7.5B', gu?.includes('7.5B'), gu);
sizesBtn.click(); await T.tick();   // restore

// params-lens instance: same compact boxes
T.check('params diagram also compact', !boxes(pl()).includes(38) && boxes(pl()).includes(32), boxes(pl()).join(','));
T.done();
