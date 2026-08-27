// @page studies/02-hopper-memory.html
const layer = () => document.getElementById('local-diagram');
const md = (el2) => el2.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
const rowLabels = () => [...layer().querySelectorAll('.lv-bar g[data-prop]')];
const ghosts = () => [...layer().querySelectorAll('.lv-bar rect[stroke-dasharray="2 2"]')];
[...layer().querySelector('.savebox').querySelectorAll('button')].find(b => b.textContent === 'save').click(); await T.tick();
T.log('ghosts all-on', ghosts().length);
T.check('ghosts for all rows + total', ghosts().length === 5, ghosts().length);
md(rowLabels()[0]); await T.tick(700);   // solo weights
const gcolors = ghosts().map(g => g.getAttribute('stroke'));
T.log('ghost colors under solo', gcolors.join('|'));
T.check('weights + 2 sub-row + total ghosts (others hidden)', ghosts().length === 4 && gcolors.filter(c => c === '#2a78d6').length === 3 && gcolors.includes('#898781'), gcolors.join('|'));
md(rowLabels()[0]); await T.tick(700);
[...layer().querySelector('.savebox').querySelectorAll('button')].find(b => b.textContent === 'reset all').click(); await T.tick(400);
T.done();
