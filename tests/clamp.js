// @page studies/02-hopper-memory.html
const layer = () => document.getElementById('local-diagram');
const stps = () => ['gpus', 'pp', 'sched', 'ep', 'zero']
  .map(k => layer().parentElement.querySelector(`.stp[data-knob="${k}"]`));
const stepBtn = (i, dir) => stps()[i].querySelectorAll('button')[dir < 0 ? 0 : 1];
const epVal = () => stps()[3].querySelector('select.v').value;
// PP -> 64: DP = 32, so EP must clamp from 64 to 32 and its + must disable
stepBtn(1, +1).click(); await T.tick(600);   // 16 -> 32
stepBtn(1, +1).click(); await T.tick(600);   // 32 -> 64
T.log('EP after PP64', epVal());
T.check('EP clamped to 32 at PP64', epVal() === '32', epVal());
T.check('EP + disabled (EP = DP)', stepBtn(3, +1).disabled, '');
T.check('EP dropdown has no 64', ![...stps()[3].querySelectorAll('option')].some(o => o.value === '64'), '');
stepBtn(1, -1).click(); await T.tick(600); stepBtn(1, -1).click(); await T.tick(600);
T.check('EP 64 available again at PP16', [...stps()[3].querySelectorAll('option')].some(o => o.value === '64'), '');
T.done();
