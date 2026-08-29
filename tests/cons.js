// @page studies/scratch-globals.html
// 02 consolidated diagram: weights + grads + optimizer + activations band
const layer = () => document.getElementById('all-diagram');
T.check('exists', !!layer(), '');
const dia = () => layer().querySelector('.lv-scroll svg');
const n = (c) => dia().querySelectorAll(`rect[fill="${c}"]`).length;
const dims = () => layer().querySelector('g[data-op="ffn_gate_up"] text.dims:not([text-anchor])')?.textContent.trim();
T.log('blue/orange/green/amber', [n('#2a78d6'), n('#eb6834'), n('#1baf7a'), n('#eda100')].join('/'));
T.check('grads squares present', n('#eb6834') > 0, n('#eb6834'));
T.check('amber squares or hollows on wires', n('#eda100') > 0 || dia().querySelectorAll('rect[stroke="#eda100"]').length > 0, '');
const amberChips = [...dia().querySelectorAll('text.tsave')].filter(t => t.textContent.includes('MiB') || t.textContent.includes('GiB'));
T.log('amber chip count', amberChips.length);
T.check('per-tensor amber chips on wires', amberChips.length > 5, amberChips.length);
T.log('sample chip', amberChips[1]?.textContent);
T.log('gate/up dims', dims());
T.check('gate/up = 98.0 GiB (14 B/param)', dims() === '98.0 GiB', dims());
T.check('no saved-for-backward total line (removed)', ![...dia().querySelectorAll('text')].some(t => t.textContent.includes('saved for backward')), '');
const cbs = () => [...layer().closest('.anat-grid').querySelectorAll('.anp-leg .row')];
T.check('four legend rows', cbs().length === 4, cbs().length);
// solo activations: params squares vanish, chips remain, boxes lose numbers
cbs()[3].click(); await T.tick(600);
T.check('solo acts: no box numbers', !dims(), dims());
T.check('solo acts: param squares gone', n('#2a78d6') === 0 && n('#1baf7a') === 0, '');
T.check('solo acts: amber chips remain', n('#eda100') > 0 || dia().querySelectorAll('rect[stroke="#eda100"]').length > 0, '');
cbs()[3].click(); await T.tick(600);
T.check('solo again restores: 98.0 GiB', dims() === '98.0 GiB', dims());
T.done();
