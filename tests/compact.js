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

// params-lens instance: count squares, one square = one 7168×2048 expert
// matrix — active (blue) first, inactive (grey) after
const fills = (id) => {
  const g = pl().querySelector(`g[data-op="${id}"]`), b = g.getBBox();
  const r = [...pl().querySelectorAll('rect[width="5"][height="4"]')].filter((q) => {
    const x = +q.getAttribute('x'), y = +q.getAttribute('y');
    return x >= b.x && x <= b.x + b.width && y >= b.y && y <= b.y + b.height;
  });
  return { blue: r.filter((q) => q.getAttribute('fill') === '#2a78d6').length, grey: r.filter((q) => q.getAttribute('fill') === '#c3c2b7').length };
};
T.check('gate/up: 16 active + 496 inactive matrices', JSON.stringify(fills('ffn_gate_up')) === '{"blue":16,"grey":496}', JSON.stringify(fills('ffn_gate_up')));
T.check('down: 8 active + 248 inactive matrices', JSON.stringify(fills('ffn_down')) === '{"blue":8,"grey":248}', JSON.stringify(fills('ffn_down')));
const emb = document.querySelector('dsv3-anatomy:has(#params-diagram) [data-op="embed"]');
const embF = [...(emb?.querySelectorAll('rect[width="4"]') ?? [])].map((q) => q.getAttribute('fill'));
T.check('plan: embedding squares all inactive grey', embF.length > 0 && embF.every((f) => f === '#c3c2b7'), embF.length + ' ' + [...new Set(embF)]);
T.done();
