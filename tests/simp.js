// @page studies/02-hopper-memory.html
const sheet = document.querySelector('dsv3-sheet');
await T.tick(400);
const rowOf = (id) => [...sheet.querySelectorAll('tr')].find(r => r.querySelector('.nm')?.textContent === id)?.textContent ?? '';
const t1 = () => rowOf('T1');
T.check('exact: lse row present', rowOf('A6b')?.includes('lse (fp32)'), rowOf('A6b'));
const exactT1 = t1();
sheet.querySelector('.simp input').click(); await T.tick(150);
T.check('simplify: aux rows gone', !rowOf('A6b')?.includes('lse') && !rowOf('A3b')?.includes('rstd'), rowOf('A3b'));
T.check('simplify: parents drop the aux ids (A6 = A6a alone)', rowOf('A6')?.includes('= A6a') && !rowOf('A6')?.includes('A6b'), rowOf('A6'));
T.check('simplify: Q3 loses the final norm', !rowOf('Q3')?.includes('7168') || !rowOf('Q3')?.includes('H1 ×'), rowOf('Q3'));
T.check('simplify: T1 drifts below the exact total', t1() !== exactT1, `${t1()} vs ${exactT1}`);
T.check('drift note shown', sheet.textContent.includes('drift slightly'), '');
sheet.querySelector('.simp input').click(); await T.tick(150);
T.check('back to exact', t1() === exactT1 && rowOf('A6b')?.includes('lse'), '');
T.done();
