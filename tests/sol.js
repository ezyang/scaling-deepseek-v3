// @page studies/03-sol.html
// the summed step: forward / backward / recompute / total rows by op group,
// every op at the FP8 peak; knobs move the bars, the readout carries the terms
const w = document.querySelector('dsv3-sol');
await T.tick(300);
const ro = () => w.querySelector('.ro').textContent;
const rowval = (id) => +w.querySelector(`g[data-row="${id}"] text[data-rowval]`).dataset.rowval;
T.check('four rows', w.querySelectorAll('g[data-row]').length === 4, w.querySelectorAll('g[data-row]').length);
T.check('total = forward + backward + recompute', Math.abs(rowval('total') - rowval('fwd') - rowval('bwd') - rowval('replay')) < 1e-9, rowval('total'));
T.check('backward = 2× forward (less the forward-only loss)', Math.abs(rowval('bwd') / rowval('fwd') - 2) < 1e-3, rowval('bwd') / rowval('fwd'));
T.check('resting readout: the sum, tokens per GPU, the reported step', /3\.99 s per step/.test(ro()) && /30,720 tokens per GPU/.test(ro()) && /20\.8 s step/.test(ro()) && /×5\.2 the sum/.test(ro()), ro());
T.check('no-recompute tick (3.89 s)', !!w.querySelector('line[data-ghost]') && /nothing recomputed, 3\.89 s/.test(ro()), '');
T.check('tech sheet shows the H800 numbers', w.querySelector('[data-sheet="FP8"]').textContent === '1,979 TFLOP/s' && w.querySelector('[data-sheet="NVLink"]').textContent === '200 GB/s' && w.querySelector('[data-sheet="InfiniBand"]').textContent === '50 GB/s', '');
// hover a segment: the exact terms
const seg = w.querySelector('g[data-row="fwd"] rect[data-seg="routed"]');
seg.dispatchEvent(new MouseEvent('mouseover', { bubbles: true })); await T.tick(50);
T.check('segment readout: FLOP/token × tokens ÷ peak', /routed experts · forward/.test(ro()) && /40\.9 GFLOP\/token × 30,720 tokens ÷ 1,979 TFLOP\/s/.test(ro()), ro());
w.querySelector('svg').dispatchEvent(new MouseEvent('mouseleave')); await T.tick(50);
// no recompute: the row empties, the total drops to the tick
const h0 = w.getBoundingClientRect().height;
w.querySelector('[data-knob="recompute"] button[data-v="none"]').click(); await T.tick(350);
T.check('recompute none: empty replay row, no tick', rowval('replay') === 0 && !w.querySelector('g[data-row="replay"] rect[data-seg]') && !w.querySelector('[data-ghost]'), rowval('replay'));
T.check('knob flip: height reserved (no reflow)', Math.abs(w.getBoundingClientRect().height - h0) < 1, '');
T.check('URL hash carries the state', /s:sum=/.test(decodeURIComponent(location.hash)) && /none/.test(location.hash), location.hash);
w.querySelector('[data-knob="recompute"] button[data-v="dsv3"]').click(); await T.tick(350);
// the cluster knob: half the GPUs, twice the tokens per GPU, everything doubles (the reported line too)
w.querySelector('[data-knob="gpus"] button[data-dir="-1"]').click(); await T.tick(350);
T.check('1024 GPUs: 61,440 tokens/GPU, sum doubles', /61,440 tokens per GPU/.test(ro()) && Math.abs(rowval('total') - 7.98) < 0.02, rowval('total'));
T.check('reported line doubles with it', Math.abs(+w.querySelector('line[data-rep]').dataset.rep - 41.66) < 0.05, w.querySelector('line[data-rep]').dataset.rep);
w.querySelector('[data-knob="gpus"] button[data-dir="1"]').click(); await T.tick(350);
// H100: same FP8 peak, wider NVLink on the sheet, no reported throughput → no line
w.querySelector('[data-knob="hw"] button[data-v="h100"]').click(); await T.tick(350);
T.check('H100: sheet flips NVLink to 450, sum unchanged, no reported line', w.querySelector('[data-sheet="NVLink"]').textContent === '450 GB/s' && Math.abs(rowval('total') - 3.99) < 0.01 && !w.querySelector('line[data-rep]') && /No reported throughput/.test(ro()), '');
w.querySelector('[data-knob="hw"] button[data-v="h800"]').click(); await T.tick(350);
T.log('widget height', w.getBoundingClientRect().height);
T.done();
