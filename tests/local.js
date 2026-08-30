// @page studies/02-hopper-memory.html
// 02 local diagram: steppers, knob tweens, cumulative toggle, ZeRO, stage map
const layer = () => document.getElementById('local-diagram');
T.check('exists', !!layer(), '');
const dia = () => layer().querySelector('.lv-scroll svg');
const dims = () => layer().querySelector('g[data-op="ffn_gate_up"] text.dims:not([text-anchor])')?.textContent.trim();
const stps = () => ['gpus', 'pp', 'sched', 'ep', 'zero']
  .map(k => layer().parentElement.querySelector(`.stp[data-knob="${k}"]`));
const stepBtn = (i, dir) => stps()[i].querySelectorAll('button')[dir < 0 ? 0 : 1];
const stpVal = (i) => stps()[i].querySelector('select.v').value;
const sels = () => [...layer().parentElement.querySelectorAll('select:not(.v)')];
T.check('stepper groups + selects (stage/precision/recompute)', stps().length === 5 && sels().length === 3, `${stps().length}/${sels().length}`);
// AC knob: the dsv3 recompute preset shrinks activations and marks ↻ chips
const rsel = () => sels().find(s2 => [...s2.options].some(o => o.value === 'dsv3' && [...s2.options].some(o2 => o2.value === 'attn-replay')));
const actsVal = () => layer().querySelector('.lv-bar')?.textContent ?? '';   // whole chart text: values live at bar ends
const a0 = actsVal();
rsel().value = 'dsv3'; rsel().dispatchEvent(new Event('change')); await T.tick(300);
T.check('dsv3 preset shrinks activations', actsVal() !== a0, `${a0} -> ${actsVal()}`);
T.check('recomputed chips appear', [...layer().querySelectorAll('.lv-scroll text.tredo')].length > 3, '');
// precision knob: fp8 stashes shrink activations further
const psel = () => sels().find(s2 => [...s2.options].some(o => o.value === 'dsv3-fp8'));
const a1 = actsVal();
psel().value = 'dsv3-fp8'; psel().dispatchEvent(new Event('change')); await T.tick(300);
T.check('fp8 recipe shrinks stashes further', actsVal() !== a1, `${a1} -> ${actsVal()}`);
psel().value = 'bf16'; psel().dispatchEvent(new Event('change')); await T.tick(200);
rsel().value = 'none'; rsel().dispatchEvent(new Event('change')); await T.tick(300);
T.check('restored to save-everything', actsVal() === a0, `${actsVal()} vs ${a0}`);
// cluster-size stepper: 2048 -> 1024 halves DP; expert-DP hits 1 (unsharded expert optim)
stepBtn(0, -1).click(); await T.tick(600);
T.check('1024 GPUs: gate/up 3.1 GiB (expert-DP 1)', dims() === '3.1 GiB', dims());
stepBtn(0, +1).click(); await T.tick(600);
T.check('2048 GPUs again: 2.2 GiB', dims() === '2.2 GiB', dims());
// the value chip is a dropdown too: jump EP 64 -> 4 directly
const epSel = stps()[3].querySelector('select.v');
epSel.value = '4'; epSel.dispatchEvent(new Event('change')); await T.tick(600);
T.check('dropdown jump EP4: 21.9 GiB', dims() === '21.9 GiB', dims());
epSel.value = '64'; epSel.dispatchEvent(new Event('change')); await T.tick(600);
T.check('dropdown back EP64: 2.2 GiB', dims() === '2.2 GiB', dims());
T.check('EP shows 64, PP shows 16', stpVal(3) === '64' && stpVal(1) === '16', `${stpVal(3)}/${stpVal(1)}`);
T.check('default = peak stage under DualPipeV (1: two chunks, 16.5 mb in flight)', sels()[0].selectedOptions[0].textContent.includes('L1–2+L57–58') && sels()[0].selectedOptions[0].textContent.includes('peak'), sels()[0].selectedOptions[0].textContent);
const chart = () => layer().querySelector('.lv-bar')?.textContent ?? '';
T.check('chart row shows the DPV in-flight count (PP+½)', chart().includes('activations ×16.5mb'), chart().slice(0,120));
// schedule knob: ×1 mb shrinks the acts (legend re-labels ×1)
const sseg = (lv) => [...stps()[2].querySelectorAll('button')].find(b => b.textContent === lv);
sseg('×1 mb').click(); await T.tick(600);
T.check('sched ×1: chart row says ×1 mb', chart().includes('activations ×1mb'), '');
sseg('1F1B').click(); await T.tick(600);
T.check('sched back to 1F1B', chart().includes('activations ×16.5mb'), '');
// fit bar: 4 segments + capacity tick + snapped total
const bar = () => [...layer().querySelectorAll('.lv-bar rect[height="8"]')];
T.log('bar rows', bar().length);
T.check('fit bar has 5 log rows (4 comps + total)', bar().length === 5, bar().length);
T.check('capacity tick labeled', (layer().querySelector('.lv-bar')?.textContent ?? '').includes('80 GiB (H100)'), '');
T.check('no saved-for-backward total line', ![...dia().querySelectorAll('text')].some(x => x.textContent.includes('saved for backward')), '');
T.check('cum default: 2.2 GiB', dims() === '2.2 GiB', dims());

// cumulative toggle (tweened)
const cumBtn = () => [...layer().parentElement.querySelectorAll('button')].find(b => b.textContent === 'per block' || b.textContent.includes('blocks'));
T.check('button ×2 blocks (the stage\'s MoE chunks)', cumBtn()?.textContent === '×2 blocks', cumBtn()?.textContent);
cumBtn().click(); await T.tick(60);
const mid = dia().querySelectorAll('rect[fill="#eda100"]').length;
await T.tick(600);
const end = dia().querySelectorAll('rect[fill="#eda100"]').length;
T.log('green mid/end', `${mid}/${end}`);
T.check('per block: 1.1 GiB', dims() === '1.1 GiB', dims());
T.check('tween passed through a mid state (amber act chips)', mid > end && mid < 200, `${mid} vs ${end}`);
cumBtn().click(); await T.tick(600);
T.check('back to 2.2 GiB', dims() === '2.2 GiB', dims());

// EP stepper: 64 -> 32 (tweened)
stepBtn(3, -1).click(); await T.tick(600);
T.check('EP32: 3.5 GiB', dims() === '3.5 GiB', dims());
T.check('(8/rank)', [...dia().querySelectorAll('text.grplabel')].some(t => t.textContent.includes('(8/rank)')), '');
stepBtn(3, +1).click(); await T.tick(600);
T.check('EP64 again: 2.2 GiB', dims() === '2.2 GiB', dims());

// PP stepper: 16 -> 8 (jumps to the new peak; DPV stage 1 hosts 8 MoE blocks)
stepBtn(1, -1).click(); await T.tick(600);
T.log('PP8 dims', dims());
T.check('PP8 peak stage (8 moe): 7.0 GiB', dims() === '7.0 GiB', dims());
T.check('PP8 jumped to peak stage 1', sels()[0].selectedOptions[0].textContent.includes('peak') && sels()[0].selectedOptions[0].textContent.startsWith('1:'), sels()[0].selectedOptions[0].textContent);
stepBtn(1, +1).click(); await T.tick(600);
T.check('PP16 again: 2.2 GiB', dims() === '2.2 GiB', dims());

// ZeRO level segments (re-query each time; render rebuilds)
const zseg = (lv) => [...stps()[4].querySelectorAll('button')].find(b => b.textContent === lv);
T.check('ZeRO segments present', !!zseg('off') && !!zseg('3'), '');
T.check('ZeRO-1 active by default', zseg('1').classList.contains('on'), '');
zseg('off').click(); await T.tick(600);
T.check('ZeRO off: 3.1 GiB', dims() === '3.1 GiB', dims());
zseg('2').click(); await T.tick(600);
T.check('ZeRO-2: 1.8 GiB (grads shard too)', dims() === '1.8 GiB', dims());
zseg('3').click(); await T.tick(600);
T.check('ZeRO-3: 1.5 GiB (weights shard too)', dims() === '1.5 GiB', dims());
zseg('1').click(); await T.tick(600);
T.check('ZeRO-1 again: 2.2 GiB', dims() === '2.2 GiB', dims());

// EP off via stepper (6 halvings), disabled at the end
for (let i = 0; i < 6; i++) { stepBtn(3, -1).click(); await T.tick(250); }
await T.tick(600);
T.check('EP1 value', stpVal(3) === '1', stpVal(3));
T.check('minus disabled at 1', stepBtn(3, -1).disabled, '');
T.check('EP1: 84.9 GiB', dims() === '84.9 GiB', dims());
for (let i = 0; i < 6; i++) { stepBtn(3, +1).click(); await T.tick(250); }
await T.tick(600);
T.check('plus disabled at 64', stepBtn(3, +1).disabled, '');
T.check('restored: 2.2 GiB', dims() === '2.2 GiB', dims());

// plan stage map + legend
const plan = layer().closest('.anat-grid')?.querySelector('dsv3-anatomy-plan');
T.check('plan stage map (stage 1: 2d+2moe; vocab rides stage 0 under reflect)', plan.textContent.includes('MoE block ×2') && plan.textContent.includes('dense block ×2') && !plan.textContent.includes('(last stage only)'), '');
T.check('unit legend on chart (448 MiB / square)', chart().includes('448.0 MiB / square'), chart().slice(0,120));
T.check('no margin legend for local', !layer().closest('.anat-grid').querySelector('.anp-leg'), '');
const rowLabels = () => [...layer().querySelectorAll('.lv-bar g[data-prop]')];
T.check('four clickable chart row labels', rowLabels().length === 4, rowLabels().length);
T.check('row labels are names only (values at bar ends)', rowLabels().every(g => !/GiB|MiB/.test(g.textContent)), '');
T.check('absolute values at bar ends, no capacity factors', chart().includes('127.7 GiB') && !chart().includes('×1/'), chart().slice(-160));
// row-label mousedown SOLOS the component (filter, not toggle)
const md = (el2) => el2.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
const totalEnd = () => {
  const rects = [...layer().querySelectorAll('.lv-bar rect[height="8"]')];
  return Math.max(...rects.map(r2 => +r2.getAttribute('x') + +r2.getAttribute('width')));
};
const endBefore = totalEnd();
md(rowLabels()[0]); await T.tick(600);
T.check('solo weights: weights stays on', rowLabels()[0].getAttribute('opacity') !== '0.35', '');
// total row: unchanged extent, now stacked grey other + blue on top
T.check('total bar does not resize under solo', Math.abs(totalEnd() - endBefore) < 1.5, `${totalEnd()} vs ${endBefore}`);
T.check('grey other base appears', !!layer().querySelector('.lv-bar rect[fill="#c3c2b7"]'), '');
T.check('total label stays full (127.7 GiB)', chart().includes('127.7 GiB'), '');
T.check('hidden rows lose their factors', ![...layer().querySelectorAll('.lv-bar text')].some(x => x.textContent.includes('×1/13')), '');
T.check('solo weights: others dim', rowLabels().slice(1).every(g => g.getAttribute('opacity') === '0.35'), '');
T.check('solo weights: box shows weights only', dims() === '448.0 MiB', dims());
md(rowLabels()[0]); await T.tick(600);
T.check('solo again restores all', rowLabels().every(g => g.getAttribute('opacity') !== '0.35'), '');
T.check('stacked other gone when all visible', !layer().querySelector('.lv-bar rect[fill="#c3c2b7"]'), '');
T.check('restored: 2.2 GiB', dims() === '2.2 GiB', dims());
T.check('two head rows for local', layer().querySelectorAll('.lv-head').length === 2, layer().querySelectorAll('.lv-head').length);
T.check('bar sits between the rows', !!layer().querySelector('.lv-head + .lv-bar + .lv-head'), '');
T.done();
