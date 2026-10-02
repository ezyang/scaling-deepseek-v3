// @page studies/03-roofline.html
// 03's comm table: SOL cells link to the comm sheet, ratios follow β_IB (a
// sheetedit knob) from the exact trace times; the per-kernel sheet is folded
// and a click on a kernel-table SOL cell opens it at the row
await T.tick(100);
const comm = document.querySelector('.tally.comm');
const row = (tr) => [...tr.cells].slice(1).map((td) => td.textContent.replace(/\s+/g, ' ').trim()).join('|');
const total = () => row(comm.querySelector('tr.hl'));
T.check('published totals', total() === '6,230 µs|15,450|2.48|6,230 µs|13,435|2.16', total());
T.check('every SOL cell names a sheet row', [...comm.querySelectorAll('[data-cellref]')].every((q) =>
  [...document.querySelectorAll('.cellsheet td.nm')].some((td) => td.innerHTML === q.dataset.cellref)), '');
const ib = [...document.querySelectorAll('.cellsheet tr')].find((tr) => tr.cells[0]?.innerHTML === 'β<sub>IB</sub>');
ib.querySelector('td.vl').click();
await T.tick(20);
const inp = ib.querySelector('input');
inp.value = '100,000,000,000';
inp.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
await T.tick(50);
T.check('β_IB doubled: SOL halves, ratios double, amber', total() === '3,110 µs|15,450|4.96|3,110 µs|13,435|4.31' && comm.querySelectorAll('td.off').length === 6, total());
ib.querySelector('td.vl').click();
await T.tick(20);
ib.querySelector('input').value = '50,000,000,000';
ib.querySelector('input').dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
await T.tick(50);
T.check('β_IB back to 50 GB/s: published again', total() === '6,230 µs|15,450|2.48|6,230 µs|13,435|2.16' && !comm.querySelector('.off'), total());
const fold = document.querySelector('details.fold'), cfold = document.querySelectorAll('details.fold')[1];
T.check('both SOL sheets folded by default', fold && !fold.open && cfold && !cfold.open, '');
document.querySelector('.tally.kern [data-cellref]').click();
await T.tick(50);
T.check('a kernel SOL click opens the fold at its row', fold.open && fold.querySelector('tr.sel')?.cells[0].innerHTML === 't<sup>dn</sup><sub>q</sub>', '');
document.querySelector('.tally.comm [data-cellref]').click();
await T.tick(50);
T.check('a comm SOL click opens its fold at its row', cfold.open && cfold.querySelector('tr.sel')?.cells[0].innerHTML === 't<sub>disp</sub>', '');
T.done();
