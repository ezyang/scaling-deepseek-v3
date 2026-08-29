// @page studies/scratch-globals.html
// 02: optimizer-states diagram (lens=param-bytes optim)
const layer = () => document.getElementById('opt-diagram');
T.check('opt diagram exists', !!layer(), '');
const rects = (sel) => [...layer().querySelectorAll(sel)];
const blue = rects('rect[fill="#2a78d6"]').length;
const green = rects('rect[fill="#1baf7a"]').length;
T.log('blue squares', blue); T.log('green squares', green);
T.check('gate/up blue row present (32 somewhere)', blue >= 32, blue);
T.check('green squares ~4x blue fills', green > blue, `${green} vs ${blue}`);
const guBlue = rects('g[data-op="ffn_gate_up"] ~ * , g[data-op="ffn_gate_up"] rect').length;
// count strips inside the gate/up box region via x-position of its rect
const box = layer().querySelector('g[data-op="ffn_gate_up"] rect.box');
T.check('gate/up box grew for 5 rows', +box.getAttribute('height') >= 60, box.getAttribute('height'));
// no cumulative button on this instance
const btns = [...layer().parentElement.querySelectorAll('button')].map(b => b.textContent);
T.check('no per-block/×N toggle', !btns.some(t => t === 'per block' || t.includes('blocks')), btns.join('|'));
// hollow greens on tiny ops (stroke, no fill)
const hollowGreen = rects('rect[stroke="#1baf7a"]').length;
T.log('hollow green', hollowGreen);
T.check('hollow green traces exist', hollowGreen > 0, hollowGreen);
// legend mentions optimizer
const leg = layer().closest('.anat-grid').querySelector('.anp-leg')?.textContent ?? '';
T.log('legend', leg.slice(0, 60));
T.check('margin legend names optimizer', leg.includes('optimizer'), leg.slice(0, 60));
// plan strips: green on embedding/lm head for THIS instance's plan
const plan = layer().closest('.anat-grid')?.querySelector('dsv3-anatomy-plan') ?? document.querySelectorAll('dsv3-anatomy-plan')[1];
const pg = plan ? [...plan.querySelectorAll('rect[fill="#1baf7a"]')].length : -1;
T.log('plan green squares', pg);
T.check('plan has green squares', pg > 0, pg);
// the parameters diagram above is unaffected (no green there)
const first = document.getElementById('diagram');
T.check('first diagram has no green', [...first.querySelectorAll('rect[fill="#1baf7a"]')].length === 0, '');
T.done();
