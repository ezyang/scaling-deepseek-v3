// @page studies/02-hopper-memory.html
// Scenario: the per-block / ×N-blocks (cumulative) toggle on 02's bytes diagram
const layer = () => document.getElementById('diagram');
const cumBtn = () => [...layer().parentElement.querySelectorAll('button')]
  .find(b => b.textContent === 'per block' || b.textContent.includes('blocks'));
const boxDims = (op) => layer().querySelector(`g[data-op="${op}"] text.dims:not([text-anchor])`)?.textContent.trim();
const planDims = () => document.querySelector('dsv3-anatomy-plan g[data-op="block-moe"] text.dims')?.textContent.trim();

T.log('initial button', cumBtn()?.textContent);
T.check('starts per block', cumBtn()?.textContent === 'per block');
T.log('gate/up dims (per block)', boxDims('ffn_gate_up'));

// toggle ON
T.click_el = cumBtn(); T.click_el.click(); await T.tick(600);  // let the fade tween finish
T.log('button after on', cumBtn()?.textContent);
T.check('button reads ×58 blocks (moe)', cumBtn()?.textContent === '×58 blocks', cumBtn()?.textContent);
T.log('gate/up dims (cumulative)', boxDims('ffn_gate_up'));
T.check('gate/up cumulative bytes', boxDims('ffn_gate_up')?.includes('812.0 GiB'), boxDims('ffn_gate_up'));
T.check('tabs hidden in cumulative', !layer().querySelector('[data-kind]'), '');

// flip kind via the plan while cumulative
T.click('dsv3-anatomy-plan g[data-kind="dense"]'); await T.tick();
T.log('kind after plan click', layer().kind);
T.check('kind flipped to dense', layer().kind === 'dense');
T.log('button after kind flip', cumBtn()?.textContent);
T.check('button reads ×3 blocks (dense)', cumBtn()?.textContent === '×3 blocks', cumBtn()?.textContent);
T.log('dense gate/up dims', boxDims('ffn_gate_up'));
T.check('dense ffn cumulative = 1.5 GiB (264M×3×2B)', boxDims('ffn_gate_up')?.includes('1.5 GiB'), boxDims('ffn_gate_up'));

// toggle OFF while dense
cumBtn().click(); await T.tick();
T.log('button after off', cumBtn()?.textContent);
T.check('back to per block', cumBtn()?.textContent === 'per block', cumBtn()?.textContent);
T.check('tabs visible again', !!layer().querySelector('[data-kind]'));
T.log('dense gate/up dims (per block)', boxDims('ffn_gate_up'));

// flip back to moe via tab
T.click('#diagram [data-kind="moe"]'); await T.tick();
T.check('kind back to moe', layer().kind === 'moe');
T.log('plan moe dims', planDims());
T.done();
