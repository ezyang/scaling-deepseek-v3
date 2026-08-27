// @page studies/02-hopper-memory.html
// 02: param-bytes has no sizes toggle, values always multiplied; plan drops ×58 in cumulative
const layer = () => document.getElementById('diagram');
const btns = () => [...layer().parentElement.querySelectorAll('button')].map(b => b.textContent);
T.log('mini-head buttons', JSON.stringify(btns()));
T.check('no sizes toggle in param-bytes', !btns().some(t => t.includes('×192') || t === '24576'), btns().join('|'));
T.check('cumulative button present', btns().some(t => t === 'per block' || t.includes('blocks')), '');

// multiplied bytes on the routed gate/up box (7.5B params × 2 B = 14.0 GiB)
const gu = layer().querySelector('g[data-op="ffn_gate_up"] text.dims:not([text-anchor])')?.textContent;
T.log('gate/up dims', gu);
T.check('gate/up bytes multiplied (14.0 GiB, no ×256)', gu?.includes('14.0 GiB') && !gu?.includes('×256'), gu);
const grp = [...layer().querySelectorAll('text.grplabel')].map(t => t.textContent).find(t => t.includes('experts'));
T.log('grp label', grp);
T.check('grp label multiplied too', !grp?.includes('×256'), grp);

// plan block labels: ×58 present per-block, gone in cumulative
const planMoe = () => document.querySelector('dsv3-anatomy-plan g[data-op="block-moe"] text.name')?.textContent;
const planDense = () => document.querySelector('dsv3-anatomy-plan g[data-op="block-dense"] text.name')?.textContent;
T.log('plan moe (per block)', planMoe());
T.check('plan shows ×58 per-block', planMoe()?.includes('×58'), planMoe());
const cumBtn = [...layer().parentElement.querySelectorAll('button')].find(b => b.textContent === 'per block' || b.textContent.includes('blocks'));
cumBtn.click(); await T.tick(600);
T.log('plan moe (cumulative)', planMoe());
T.log('plan dense (cumulative)', planDense());
T.check('plan drops ×58 in cumulative', !planMoe()?.includes('×58'), planMoe());
T.check('plan drops ×3 in cumulative', !planDense()?.includes('×3'), planDense());
const cumBtn2 = [...layer().parentElement.querySelectorAll('button')].find(b => b.textContent.includes('blocks') || b.textContent === 'per block');
T.check('button still says ×58 blocks', cumBtn2.textContent === '×58 blocks', cumBtn2.textContent);
cumBtn.click(); await T.tick(600);
T.check('×58 returns per-block', planMoe()?.includes('×58'), planMoe());
T.done();
