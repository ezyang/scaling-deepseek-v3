// @page studies/02-hopper-memory.html
// the free-floating SwiGLU-input stash format (Haziza): the quantize pill's own dtype button
const f8 = document.querySelector('dsv3-anatomy[controls="dtype"]');
const gib = () => +([...f8.querySelectorAll('.lv-scroll text, svg text')].map(t => t.textContent).join(' ').match(/= ([\d.]+) GiB/) ?? [])[1];
const swb = () => f8.querySelector('button[data-dt="swiglu_in"]');
const chipOn = (name) => [...f8.querySelectorAll('.stp[data-knob="recipe"] button')].find(b => b.classList.contains('on'))?.textContent === name;
T.check('quantize pill carries the stash-format button (e4m3 under dsv3-fp8)', swb()?.textContent === 'e4m3', swb()?.textContent);
T.check('quantize pill sits between gate/up and SwiGLU', !!f8.querySelector('g[data-op="quant"]')
  && f8.querySelector('g[data-op="quant"] rect').getBBox().y > f8.querySelector('g[data-op="ffn_gate_up"] rect').getBBox().y
  && f8.querySelector('g[data-op="quant"] rect').getBBox().y < f8.querySelector('g[data-op="swiglu"] rect').getBBox().y, '');
// the GEMM's own bf16 output is never stashed; the quantized copy is
const chips = (cls) => [...f8.querySelectorAll(`.lv-scroll text.${cls}`)].map(t => t.textContent).filter(t => t.includes('gate, up (routed)'));
T.check('gate/up GEMM output reads as an unstashed bf16 wire', chips('tidle').some(t => t.startsWith('·') && t.includes('bf16')), chips('tidle').join('|'));
T.check('the quantized copy is the e4m3 stash', chips('tsave').some(t => /^[↓⇓]/.test(t) && t.includes('e4m3')), chips('tsave').join('|'));
T.check('recipe recognized at load', chipOn('dsv3-fp8'), '');
const g0 = gib();
swb().click(); await T.tick(500);
T.check('flip to bf16: label + custom + stash grows', swb().textContent === 'bf16'
  && chipOn('custom') && gib() > g0, `${swb().textContent} ${gib()} vs ${g0}`);
T.log('delta GiB', (gib() - g0).toFixed(1));
swb().click(); await T.tick(500);
T.check('flip back: dsv3-fp8 relights, bytes restore', chipOn('dsv3-fp8') && gib() === g0, `${gib()}`);

// boolean mark buttons: both states read ↻; class carries the boolean
const ac = document.querySelector('dsv3-anatomy[controls="marks"]');
const mb = (id) => ac.querySelector(`button[data-mark="${id}"]`);
T.check('kept op: ↻ struck (st-keep)', mb('x1')?.textContent === '↻' && mb('x1').classList.contains('st-keep'), mb('x1')?.className);
T.check('replayed op: ↻ live (st-redo)', mb('swiglu')?.textContent === '↻' && mb('swiglu').classList.contains('st-redo'), mb('swiglu')?.className);
T.check('no 💾 anywhere', ![...ac.querySelectorAll('button')].some(b => b.textContent.includes('💾')), '');
T.check('region chips read ↻ none / ↻ all', !!ac.querySelector('button[data-regionact="save"]')
  && ac.querySelector('button[data-regionact="save"]').textContent.includes('↻ none'), '');

// local sim: full tooltips now (⚠/box facts) + the in: button rides along
const layer = document.getElementById('local-diagram');
const host = layer.parentElement;
T.check('local diagram has the in: button too', !!layer.querySelector('button[data-dt="swiglu_in"]'), '');
const tipped = layer.querySelector('.lv-scroll g[data-tip]');
const tip = () => layer.querySelector('.lv-tip');
const r = tipped.getBoundingClientRect();
tipped.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: r.left + 5, clientY: r.top + 5 }));
await T.tick(30);
T.check('local diagram data-tip tooltips work now', tip().style.display === 'block' && tip().textContent.length > 10,
  tip().textContent.slice(0, 60));
T.done();
