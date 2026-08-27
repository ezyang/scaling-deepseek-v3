// @page studies/02-hopper-memory.html
// solo breakdown sub-bars + alert factor badges
const layer = () => document.getElementById('local-diagram');
const chart = () => layer().querySelector('.lv-bar');
const md = (el2) => el2.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
const rowLabels = () => [...layer().querySelectorAll('.lv-bar g[data-prop]')];
// solo weights -> sub-bars ease into the freed rows
md(rowLabels()[0]); await T.tick(700);
const txt = () => chart().textContent;
T.check('sub-bars: experts / non-expert / vocab', txt().includes('· experts') && txt().includes('· non-expert'), txt().slice(-200));
T.check('no vocab sub-bar on a mid stage', !txt().includes('· vocab'), '');
// pin, then EP 64->32: ONLY the experts sub-bar gets a red x2 badge
const saveBtn = [...layer().querySelector('.savebox').querySelectorAll('button')].find(b => b.textContent === 'save');
saveBtn.click(); await T.tick();
const stps = () => ['gpus', 'pp', 'sched', 'ep', 'zero']
  .map(k => layer().parentElement.querySelector(`.stp[data-knob="${k}"]`));
stps()[3].querySelectorAll('button')[0].click(); await T.tick(700);   // EP -
const badges = [...chart().querySelectorAll('tspan[fill="#d03b3b"]')].map(t => t.textContent.trim());
T.log('red badges', JSON.stringify(badges));
T.check('experts sub-bar shows red ▲×2', badges.some(b => b === '▲×2'), badges.join('|'));
T.check('in-diagram numbers wear badges too', [...layer().querySelectorAll('.lv-scroll tspan[fill="#d03b3b"]')].length > 0, '');
// EP back up: badges fade to ▼/none
stps()[3].querySelectorAll('button')[1].click(); await T.tick(700);
T.check('badges clear when back at baseline', ![...chart().querySelectorAll('tspan')].some(t => t.textContent.includes('▲')), '');
// unsolo
md(rowLabels()[0]); await T.tick(700);
T.check('sub-bars gone when unsoloed', !txt().includes('· experts'), '');
[...layer().querySelector('.savebox').querySelectorAll('button')].find(b => b.textContent === 'reset all').click(); await T.tick(400);
T.done();
