// @page studies/02-hopper-memory.html
// 02 local diagram: steppers, knob tweens, pinned cumulative, ZeRO, stage map
const layer = () => document.getElementById('local-diagram');
T.check('exists', !!layer(), '');
const dia = () => layer().querySelector('.lv-scroll svg');
const dims = () => layer().querySelector('g[data-op="ffn_gate_up"] text.dims:not([text-anchor])')?.textContent.trim();
const stps = () => ['gpus', 'pp', 'sched', 'ep', 'zero']
  .map(k => layer().parentElement.querySelector(`.stp[data-knob="${k}"]`));
const stepBtn = (i, dir) => stps()[i].querySelectorAll('button')[dir < 0 ? 0 : 1];
const stpVal = (i) => stps()[i].querySelector('select.v').value;
const sels = () => [...layer().parentElement.querySelectorAll('select:not(.v)')];
T.check('stepper groups + selects (precision/recompute — the rank picker is a segment now)',
  stps().length === 5 && sels().length === 2
  && layer().parentElement.querySelector('.stp[data-knob="rank"]'), `${stps().length}/${sels().length}`);
// defaults = the OPTIMIZED config (the deck's step 6): dsv3 recompute +
// dsv3-fp8 recipe — the ↻ chips are already on screen at load
const rsel = () => sels().find(s2 => [...s2.options].some(o => o.value === 'dsv3' && [...s2.options].some(o2 => o2.value === 'attn-replay')));
const psel = () => sels().find(s2 => [...s2.options].some(o => o.value === 'dsv3-fp8'));
const actsVal = () => layer().querySelector('.lv-bar')?.textContent ?? '';   // whole chart text: values live at bar ends
T.check('defaults are optimized (dsv3 recompute + dsv3-fp8 recipe)',
  rsel().value === 'dsv3' && psel().value === 'dsv3-fp8', `${rsel().value}/${psel().value}`);
T.check('recomputed chips present at load', [...layer().querySelectorAll('.lv-scroll text.tredo')].length > 3, '');
const a0 = actsVal();
rsel().value = 'none'; rsel().dispatchEvent(new Event('change')); await T.tick(300);
T.check('save-everything grows activations', actsVal() !== a0, `${a0} -> ${actsVal()}`);
// precision knob: bf16 stashes grow further still
const a1 = actsVal();
psel().value = 'bf16'; psel().dispatchEvent(new Event('change')); await T.tick(300);
T.check('bf16 recipe grows stashes further', actsVal() !== a1, `${a1} -> ${actsVal()}`);
psel().value = 'dsv3-fp8'; psel().dispatchEvent(new Event('change')); await T.tick(200);
rsel().value = 'dsv3'; rsel().dispatchEvent(new Event('change')); await T.tick(300);
T.check('restored to the optimized default', actsVal() === a0, `${actsVal()} vs ${a0}`);
// cluster-size stepper: 2048 -> 1024 halves DP; expert-DP hits 1 (unsharded expert optim)
stepBtn(0, -1).click(); await T.tick(600);
T.check('1024 GPUs: gate/up 8.8 GiB (expert-DP 2)', dims() === '8.8 GiB', dims());
stepBtn(0, +1).click(); await T.tick(600);
T.check('2048 GPUs again: 7.0 GiB', dims() === '7.0 GiB', dims());
// the value chip is a dropdown too: jump EP 64 -> 4 directly
const epSel = stps()[3].querySelector('select.v');
epSel.value = '4'; epSel.dispatchEvent(new Event('change')); await T.tick(600);
T.check('dropdown jump EP4: 85.8 GiB', dims() === '85.8 GiB', dims());
epSel.value = '64'; epSel.dispatchEvent(new Event('change')); await T.tick(600);
T.check('dropdown back EP64: 7.0 GiB', dims() === '7.0 GiB', dims());
T.check('EP shows 64, PP shows 8', stpVal(3) === '64' && stpVal(1) === '8', `${stpVal(3)}/${stpVal(1)}`);
const rankSeg = () => layer().parentElement.querySelector('.stp[data-knob="rank"]');
T.check('default = the interior/peak rank (two-way segment, r1–7 on)',
  [...rankSeg().querySelectorAll('button')].find((b) => b.classList.contains('on')).textContent.includes('r1–7 · peak')
  && layer().stage !== 0, rankSeg().textContent);
const chart = () => layer().querySelector('.lv-bar')?.textContent ?? '';
T.check('chart row shows the DPV in-flight count (PP+½)', chart().includes('activations ×8.5mb'), chart().slice(0,120));
// schedule knob: ×1 mb shrinks the acts (legend re-labels ×1)
const sseg = (lv) => [...stps()[2].querySelectorAll('button')].find(b => b.textContent === lv);
sseg('×1 mb').click(); await T.tick(600);
T.check('sched ×1: chart row says ×1 mb', chart().includes('activations ×1mb'), '');
sseg('DualPipeV').click(); await T.tick(600);
T.check('sched back to DualPipeV', chart().includes('activations ×8.5mb'), '');
// fit bar: 4 segments + capacity tick + snapped total
const bar = () => [...layer().querySelectorAll('.lv-bar rect[height="8"]')];
T.log('bar rows', bar().length);
T.check('fit bar has 5 log rows (4 comps + total)', bar().length === 5, bar().length);
T.check('capacity tick labeled', (layer().querySelector('.lv-bar')?.textContent ?? '').includes('80 GiB (H100)'), '');
T.check('no saved-for-backward total line', ![...dia().querySelectorAll('text')].some(x => x.textContent.includes('saved for backward')), '');
T.check('cum default: 7.0 GiB', dims() === '7.0 GiB', dims());

// cumulative is PINNED for local (no per-block toggle: the fit bar totals
// the rank, so a per-block diagram would disagree with the chart)
const cumBtn = () => [...layer().parentElement.querySelectorAll('button')].find(b => b.textContent === 'per block' || b.textContent.includes('blocks'));
T.check('no ×N blocks / per-block button on the local head', !cumBtn(), cumBtn()?.textContent);

// EP stepper: 64 -> 32 (tweened)
stepBtn(3, -1).click(); await T.tick(600);
T.check('EP32: 12.3 GiB', dims() === '12.3 GiB', dims());
T.check('(8/rank)', [...dia().querySelectorAll('text.grplabel')].some(t => t.textContent.includes('(8/rank)')), '');
stepBtn(3, +1).click(); await T.tick(600);
T.check('EP64 again: 7.0 GiB', dims() === '7.0 GiB', dims());

// PP stepper: {1, 8} only — down to the whole-model view and back
stepBtn(1, -1).click(); await T.tick(600);
T.log('PP1 dims', dims());
T.check('PP1: the whole model (EP64): 39.6 GiB', dims() === '39.6 GiB', dims());
T.check('PP1: no rank picker (nothing to choose)', !rankSeg(), '');
T.check('PP − disabled at 1', stepBtn(1, -1).disabled, '');
stepBtn(1, +1).click(); await T.tick(600);
T.check('PP8 again: 7.0 GiB, + disabled', dims() === '7.0 GiB' && stepBtn(1, +1).disabled, dims());

// ZeRO level segments (re-query each time; render rebuilds)
const zseg = (lv) => [...stps()[4].querySelectorAll('button')].find(b => b.textContent === lv);
T.check('ZeRO segments present', !!zseg('off') && !!zseg('3'), '');
T.check('ZeRO-1 active by default', zseg('1').classList.contains('on'), '');
zseg('off').click(); await T.tick(600);
T.check('ZeRO off: 12.3 GiB', dims() === '12.3 GiB', dims());
zseg('2').click(); await T.tick(600);
T.check('ZeRO-2: 4.4 GiB (grads shard too)', dims() === '4.4 GiB', dims());
zseg('3').click(); await T.tick(600);
T.check('ZeRO-3: 3.1 GiB (weights shard too)', dims() === '3.1 GiB', dims());
zseg('1').click(); await T.tick(600);
T.check('ZeRO-1 again: 7.0 GiB', dims() === '7.0 GiB', dims());

// EP off via stepper (6 halvings), disabled at the end
for (let i = 0; i < 6; i++) { stepBtn(3, -1).click(); await T.tick(250); }
await T.tick(600);
T.check('EP1 value', stpVal(3) === '1', stpVal(3));
T.check('minus disabled at 1', stepBtn(3, -1).disabled, '');
T.check('EP1: 337.8 GiB', dims() === '337.8 GiB', dims());
for (let i = 0; i < 6; i++) { stepBtn(3, +1).click(); await T.tick(250); }
await T.tick(600);
T.check('plus disabled at 64', stepBtn(3, +1).disabled, '');
T.check('restored: 7.0 GiB', dims() === '7.0 GiB', dims());

// plan stage map + legend
const plan = layer().closest('.anat-grid')?.querySelector('dsv3-anatomy-plan');
T.check('plan stage map (stage 1: 8 moe; vocab rides stage 0 under reflect)', plan.textContent.includes('MoE block ×8') && !plan.textContent.includes('(last stage only)'), '');
T.check('unit legend on chart (448 MiB / square)', chart().includes('448.0 MiB / square'), chart().slice(0,120));
T.check('no margin legend for local', !layer().closest('.anat-grid').querySelector('.anp-leg'), '');
const rowLabels = () => [...layer().querySelectorAll('.lv-bar g[data-prop]')];
T.check('four clickable chart row labels', rowLabels().length === 4, rowLabels().length);
T.check('row labels are names only (values at bar ends)', rowLabels().every(g => !/GiB|MiB/.test(g.textContent)), '');
T.check('absolute values at bar ends, no capacity factors', chart().includes('61.7 GiB') && !chart().includes('×1/'), chart().slice(-160));
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
T.check('total label stays full (61.7 GiB)', chart().includes('61.7 GiB'), '');
T.check('hidden rows lose their factors', ![...layer().querySelectorAll('.lv-bar text')].some(x => x.textContent.includes('×1/13')), '');
T.check('solo weights: others dim', rowLabels().slice(1).every(g => g.getAttribute('opacity') === '0.35'), '');
T.check('solo weights: box shows weights only', dims() === '1.8 GiB', dims());
md(rowLabels()[0]); await T.tick(600);
T.check('solo again restores all', rowLabels().every(g => g.getAttribute('opacity') !== '0.35'), '');
T.check('stacked other gone when all visible', !layer().querySelector('.lv-bar rect[fill="#c3c2b7"]'), '');
T.check('restored: 7.0 GiB', dims() === '7.0 GiB', dims());
T.check('two head rows for local', layer().querySelectorAll('.lv-head').length === 2, layer().querySelectorAll('.lv-head').length);
T.check('bar sits between the rows', !!layer().querySelector('.lv-head + .lv-bar + .lv-head'), '');
// fp8-resident PARAMETERS (e4m3+ᵀ, 1×128 scales): weights ×2.0625/2 on the
// non-vocab share — exact, and gated on an fp8 recipe
{
  const host = layer().parentElement;
  const p8 = () => host.querySelector('input[data-knob="fp8params"]');
  T.check('e4m3+ᵀ params checkbox ENABLED under the default fp8 recipe', p8() && !p8().disabled, '');
  const w = () => +[...layer().querySelectorAll('.lv-bar text[data-role="val:0"]')][0].dataset.true;
  const w0 = w();
  p8().click(); await T.tick(600);
  T.check('fp8-resident params: weights ×2.0625/2 exactly (no vocab on this rank)',
    Math.abs(w() / w0 - 2.0625 / 2) < 1e-9, (w() / w0).toFixed(6));
  p8().click(); await T.tick(600);
  T.check('and back', w() === w0, w());
  psel().value = 'bf16'; psel().dispatchEvent(new Event('change')); await T.tick(600);
  T.check('checkbox disabled under bf16', p8().disabled, '');
  psel().value = 'dsv3-fp8'; psel().dispatchEvent(new Event('change')); await T.tick(300);
}
T.done();
