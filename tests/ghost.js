// @page studies/02-hopper-memory.html
const layer = () => document.getElementById('local-diagram');
const md = (el2) => el2.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
const rowLabels = () => [...layer().querySelectorAll('.lv-bar g[data-prop]')];
const ghosts = () => [...layer().querySelectorAll('.lv-bar rect[stroke-dasharray="2 2"]')];
const stp = (k) => layer().parentElement.querySelector(`.stp[data-knob="${k}"]`);
[...layer().querySelector('.savebox').querySelectorAll('button')].find(b => b.textContent === 'save').click(); await T.tick();
// ghosts are DELTA-ONLY: at the save point nothing has changed, so none draw
T.check('no ghosts at the save point (delta-only)', ghosts().length === 0, ghosts().length);
stp('ep').querySelectorAll('button')[0].click(); await T.tick(700);   // EP 64 -> 32
T.log('ghosts after EP change', ghosts().length);
T.check('ghosts only where EP moved (weights, grads, acts?, total — optim invariant)',
  ghosts().length > 0 && !ghosts().some(g => g.getAttribute('stroke') === '#1baf7a'), ghosts().length);
md(rowLabels()[0]); await T.tick(700);   // solo weights
const gcolors = ghosts().map(g => g.getAttribute('stroke'));
T.log('ghost colors under solo', gcolors.join('|'));
T.check('soloed: only weights-family + total ghosts', gcolors.every(c => c === '#2a78d6' || c === '#898781')
  && gcolors.filter(c => c === '#2a78d6').length >= 2, gcolors.join('|'));
md(rowLabels()[0]); await T.tick(700);
[...layer().querySelector('.savebox').querySelectorAll('button')].find(b => b.textContent === 'reset all').click(); await T.tick(400);
T.done();
