// @page studies/02-hopper-memory.html
// the cell graph: formula tooltips, pinned drill-down stack, formula sheet
const layer = () => document.getElementById('local-diagram');
const tip = () => layer().querySelector('.lv-tip');
const val = (id) => layer().querySelector(`.lv-bar text[data-role="val:${id}"]`);
const mm = (el2, type = 'mousemove') => {
  const r = el2.getBoundingClientRect();
  el2.dispatchEvent(new MouseEvent(type, { bubbles: true, clientX: r.left + 4, clientY: r.top + 4 }));
};
val('total').scrollIntoView({ block: 'center' }); await T.tick(50);
// hover: one entry, formula + value
mm(val('3')); await T.tick(30);
T.check('acts hover shows the A1 formula', tip().textContent.includes('A1 · saved activations')
  && tip().textContent.includes('= (L1 × D1 + L2 × D2) × 4096 × P6 + D3 × 4096'), tip().textContent.slice(0, 90));
// pin and drill: A1 → P6 (in flight) → P2 (PP)
mm(val('3'), 'click'); await T.tick(30);
T.check('click pins (refs go live)', tip().classList.contains('pinned')
  && tip().querySelector('.cellref[data-cell="P6"]'), '');
tip().querySelector('.cellref[data-cell="P6"]').dispatchEvent(new MouseEvent('click', { bubbles: true }));
await T.tick(30);
T.check('drilling P6 pushes its entry below (stack of 2)',
  tip().querySelectorAll('.lv-cellent').length === 2
  && tip().textContent.includes('microbatches in flight') && tip().textContent.includes('= P2 + 0.5 = 8.5'),
  tip().textContent.slice(-120));
const p2ref = [...tip().querySelectorAll('.lv-cellent[data-k="1"] .cellref')].find(s => s.dataset.cell === 'P2');
p2ref.dispatchEvent(new MouseEvent('click', { bubbles: true })); await T.tick(30);
T.check('drilling P2 grows the stack to 3 (a leaf: no formula)',
  tip().querySelectorAll('.lv-cellent').length === 3 && tip().textContent.includes('pipeline stages (PP)'), '');
// one path only: clicking a ref in entry 0 truncates below it
const w1ref = tip().querySelector('.lv-cellent[data-k="0"] .cellref[data-cell="D1"]')
  ?? tip().querySelector('.lv-cellent[data-k="0"] .cellref');
w1ref.dispatchEvent(new MouseEvent('click', { bubbles: true })); await T.tick(30);
T.check('clicking a ref in entry 0 truncates the path (stack of 2 again)',
  tip().querySelectorAll('.lv-cellent').length === 2, tip().querySelectorAll('.lv-cellent').length);
// clicking outside closes
layer().querySelector('.lv-scroll svg').dispatchEvent(new MouseEvent('click', { bubbles: true }));
await T.tick(30);
T.check('outside click unpins', tip().style.display === 'none', '');
// the sheet: bound, live, and in agreement with the chart
const sheet = document.querySelector('dsv3-sheet');
await T.tick(200);
const rows = () => [...sheet.querySelectorAll('tr')].slice(1);
T.check('sheet renders all cells', rows().length >= 20, rows().length);
const t1row = rows().find(r => r.querySelector('.nm')?.textContent === 'T1');
T.check('sheet T1 equals the chart total', t1row.textContent.includes('61.7 GiB')
  && +val('total').dataset.true === 66296545344, t1row?.textContent);
// live: flip ZeRO off, the sheet's O1 formula loses the sharding
const zseg = layer().parentElement.querySelector('.stp[data-knob="zero"]');
[...zseg.querySelectorAll('button')].find(b => b.textContent === 'off').click(); await T.tick(700);
const o1 = () => rows().find(r => r.querySelector('.nm')?.textContent === 'O1')?.textContent;
T.check('sheet is live: ZeRO off unshards the O1 formula', o1().includes('8 × (Q1 + Q2 + Q3)'), o1());
[...layer().parentElement.querySelector('.stp[data-knob="zero"]').querySelectorAll('button')]
  .find(b => b.textContent === '1').click(); await T.tick(700);
T.check('and back', o1().includes('Q1 / P5'), o1());
T.done();
