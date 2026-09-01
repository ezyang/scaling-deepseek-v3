// @page studies/02-hopper-memory.html
const layer = document.getElementById('local-diagram');
const dia = () => layer.querySelector('.lv-scroll svg');
const texts = () => [...dia().querySelectorAll('text')].map(t => t.textContent);
// dtype tags on saved chips (e4m3 pink / e5m6 purple / fp32 brick)
const tagged = [...dia().querySelectorAll('text.tsave tspan[fill="#d6408b"], text.tsave tspan[fill="#7b2fa8"]')];
T.check('saved chips wear dtype tags', tagged.length >= 3, tagged.map(t => t.textContent).join(','));
T.check('E5M6 tag on the attn-out stash', tagged.some(t => t.textContent.includes('e5m6')), '');
// redo chips: dtype + hollow counterfactual grids
const redo = [...dia().querySelectorAll('text.tredo')].filter(t => /norm|swiglu|latent|q ·|rope/.test(t.textContent));
T.check('redo chips carry dtype tags too', redo.some(t => t.querySelector('tspan[fill]')), redo[0]?.textContent);
const hollowAmber = dia().querySelectorAll('rect[stroke="#eda100"][width="4.2"]').length;
T.check('hollow ↻ counterfactual grids present', hollowAmber > 20, hollowAmber);
// bf16 phantom tails on fp8 stashes
const phantom = dia().querySelectorAll('rect[stroke="#d19023"]').length;
T.check('bf16 phantom tails on sub-bf16 stashes', phantom > 5, phantom);
// fp32 aux artifacts at rank scale
T.check('lse/rstd aux chips shown', texts().some(t => /← lse ·/.test(t)) && texts().some(t => /rstd/.test(t)),
  texts().filter(t => /lse|rstd/.test(t)).slice(0, 3).join(' | '));
// idle chips name the wire precision
T.check('idle chips named with dtype', [...dia().querySelectorAll('text.tidle')].some(t => t.textContent.startsWith('·')), '');
// chip tooltips (needTip) live
const chip = [...dia().querySelectorAll('text.tsave tspan[data-tip]')][0];
T.check('chips carry state tooltips (on the NAME tspan only — no raw-B conflict)',
  !!chip && chip.dataset.tip.includes('kept alive by') && !chip.closest('text').hasAttribute('data-tip'),
  chip?.dataset.tip.slice(0, 40));
T.done();
