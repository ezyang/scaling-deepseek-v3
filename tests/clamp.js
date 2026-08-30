// @page studies/02-hopper-memory.html
const layer = () => document.getElementById('local-diagram');
const stps = () => ['gpus', 'pp', 'sched', 'ep', 'zero']
  .map(k => layer().parentElement.querySelector(`.stp[data-knob="${k}"]`));
const stepBtn = (i, dir) => stps()[i].querySelectorAll('button')[dir < 0 ? 0 : 1];
const epVal = () => stps()[3].querySelector('select.v').value;
// shrink the CLUSTER to 256 GPUs: DP = 256/8 = 32, so EP must clamp from
// 64 to 32 and its + must disable (PP itself only offers {1, 8} now)
const gsel = () => stps()[0].querySelector('select.v');
gsel().value = '256'; gsel().dispatchEvent(new Event('change')); await T.tick(600);
T.log('EP after 256 GPUs', epVal());
T.check('EP clamped to 32 at 256 GPUs·PP8', epVal() === '32', epVal());
T.check('EP + disabled (EP = DP)', stepBtn(3, +1).disabled, '');
T.check('EP dropdown has no 64', ![...stps()[3].querySelectorAll('option')].some(o => o.value === '64'), '');
gsel().value = '2048'; gsel().dispatchEvent(new Event('change')); await T.tick(600);
T.check('EP 64 available again at 2048 GPUs', [...stps()[3].querySelectorAll('option')].some(o => o.value === '64'), '');
T.check('PP offers exactly {1, 8}', [...stps()[1].querySelectorAll('option')].map(o => o.value).join(',') === '1,8', '');
T.done();
