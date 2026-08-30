// @page studies/02-hopper-memory.html
// audit overlay: the visual language verified in text — toggle, hover
// highlighting, and re-derivation after knob changes (never mid-tween red)
const layer = () => document.getElementById('local-diagram');
const audBtn = () => layer().querySelector('button.audit');
const panel = () => layer().querySelector('.lv-audit');
const lines = () => [...(panel()?.querySelectorAll('.aud-ln') ?? [])];

T.check('audit button in the misc row', !!audBtn(), '');
audBtn().click(); await T.tick(400);
T.check('panel opens with verified implications', lines().length > 10
  && panel().textContent.includes('implications verified'), lines().length);
T.check('no lies at rest', lines().every((l) => !l.classList.contains('bad')), '');
T.check('value + edge + ruler patterns narrated', panel().textContent.includes('rounded')
  && panel().textContent.includes('log₂ axis') && panel().textContent.includes('doublings'), '');
// hover a line → its pattern stays lit, the rest dims
const ln = lines().find((l) => l.dataset.sel?.startsWith('[data-role="val:'));
ln.dispatchEvent(new MouseEvent('mouseover', { bubbles: true })); await T.tick(50);
const dimmed = [...layer().querySelectorAll('.lv-bar svg [data-role]')].filter((n) => n.style.opacity === '0.2');
T.check('hover dims everything but the verified pattern', dimmed.length > 5, dimmed.length);
ln.dispatchEvent(new MouseEvent('mouseout', { bubbles: true })); await T.tick(50);
T.check('mouseout restores', [...layer().querySelectorAll('.lv-bar svg [data-role]')]
  .every((n) => n.style.opacity === ''), '');
// save + knob change: the report re-derives at rest (badges/ghosts join it)
[...layer().querySelector('.savebox').querySelectorAll('button')].find((b) => b.textContent === 'save').click();
await T.tick(200);
layer().parentElement.querySelector('.stp[data-knob="ep"]').querySelectorAll('button')[0].click();
await T.tick(700);   // settle
T.check('report re-derives after the change (ghost lines appear, still no lies)',
  panel().textContent.includes('baseline') && lines().every((l) => !l.classList.contains('bad')),
  lines().filter((l) => l.classList.contains('bad')).map((l) => l.textContent)[0]);
[...layer().querySelector('.savebox').querySelectorAll('button')].find((b) => b.textContent === 'reset all').click();
await T.tick(700);
audBtn().click(); await T.tick(400);
T.check('toggle closes the panel', !panel(), '');
T.done();
