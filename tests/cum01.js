// @page studies/01-deepseek-diagram.html
// 01: top diagram (no lens) and params diagram (counts lens) cumulative toggles
const L = (id) => document.getElementById(id);
const btnOf = (id) => [...L(id).parentElement.querySelectorAll('button')]
  .find(b => b.textContent === 'per block' || b.textContent.includes('blocks'));
const dims = (id, op) => L(id).querySelector(`g[data-op="${op}"] text.dims:not([text-anchor])`)?.textContent.trim();

// top diagram
T.log('top btn', btnOf('diagram')?.textContent);
btnOf('diagram')?.click(); await T.tick();
T.log('top btn after', btnOf('diagram')?.textContent);
T.log('top gate/up', dims('diagram', 'ffn_gate_up'));
T.check('top: cumulative collapses counts', dims('diagram', 'ffn_gate_up')?.includes('435.9B'), dims('diagram', 'ffn_gate_up'));
btnOf('diagram')?.click(); await T.tick();
T.check('top: toggles back', btnOf('diagram')?.textContent === 'per block', btnOf('diagram')?.textContent);
T.log('top gate/up back', dims('diagram', 'ffn_gate_up'));

// params diagram
T.log('params btn', btnOf('params-diagram')?.textContent);
btnOf('params-diagram')?.click(); await T.tick();
T.log('params btn after', btnOf('params-diagram')?.textContent);
T.log('params gate/up', dims('params-diagram', 'ffn_gate_up'));
T.check('params: cumulative counts', dims('params-diagram', 'ffn_gate_up')?.includes('435.9B'), dims('params-diagram', 'ffn_gate_up'));
// flip to dense while cumulative, via the second plan
const plans = document.querySelectorAll('dsv3-anatomy-plan');
plans[1].querySelector('g[data-kind="dense"]').dispatchEvent(new MouseEvent('click', { bubbles: true })); await T.tick();
T.check('params: kind flipped', L('params-diagram').kind === 'dense');
T.log('params btn dense', btnOf('params-diagram')?.textContent);
T.check('params: ×3 blocks on dense', btnOf('params-diagram')?.textContent === '×3 blocks', btnOf('params-diagram')?.textContent);
T.log('params dense ffn', dims('params-diagram', 'ffn_gate_up'));
T.done();
