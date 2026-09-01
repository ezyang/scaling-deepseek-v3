// @page studies/02-hopper-memory.html
const layer = () => document.getElementById('local-diagram');
const sheet = document.querySelector('dsv3-sheet');
await T.tick(400);
const trOf = (id) => [...sheet.querySelectorAll('tr')].find(r => r.querySelector('.nm')?.textContent === id);
// bounds: EP at 64 → + dimmed; PP at 8 → + dimmed; ZeRO at 1 → both live
T.check('EP + is dimmed at its 64 bound', trOf('P3').querySelector('.sb.up').classList.contains('dis')
  && !trOf('P3').querySelector('.sb.dn').classList.contains('dis'), '');
T.check('PP + dimmed at 8', trOf('P2').querySelector('.sb.up').classList.contains('dis'), '');
// pinned toggles render as plain cells (x0 is always saved — no mark button)
T.check('x0 kept? is pinned: no toggle face', !trOf('R2a').querySelector('td.vl.tg')
  && trOf('R2a').querySelector('td.vl').title.includes('pinned'), '');
T.check('editable cells wear the tint (edv/tg classes present)',
  !!trOf('P1').querySelector('td.vl.edv') && !!trOf('F1').querySelector('td.vl.tg'), '');
// dimmed direction doesn't fire
const md = (el2) => el2.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
const v0 = trOf('P3').querySelector('td.vl').textContent.trim();
md(trOf('P3').querySelector('.sb.up')); await T.tick(600);
T.check('dimmed + is inert', trOf('P3').querySelector('td.vl').textContent.trim() === v0, '');
// after stepping down, + re-enables
md(trOf('P3').querySelector('.sb.dn')); await T.tick(700);
T.check('after stepping down, + re-enables', !trOf('P3').querySelector('.sb.up').classList.contains('dis'), '');
md(trOf('P3').querySelector('.sb.up')); await T.tick(700);
// formula is the rightmost column
const cellsHdr = [...sheet.querySelectorAll('th')].map(t => t.textContent);
T.check('formula lives on the rightmost column', cellsHdr[cellsHdr.length - 1] === 'formula', cellsHdr.join('|'));
T.done();
