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
// hover: one entry — the acts total is the SUM OF ITS ACCORDION
mm(val('3')); await T.tick(30);
T.check('acts hover shows A1 = the bucket sum', tip().textContent.includes('A1 · saved activations')
  && tip().textContent.includes('= A2 + A3 + A4'), tip().textContent.slice(0, 90));
// pin and drill: A1 → A8 (dispatched) → P6 (in flight) → truncate
mm(val('3'), 'click'); await T.tick(30);
T.check('click pins (refs go live)', tip().classList.contains('pinned')
  && tip().querySelector('.cellref[data-cell="A8"]'), '');
tip().querySelector('.cellref[data-cell="A8"]').dispatchEvent(new MouseEvent('click', { bubbles: true }));
await T.tick(30);
T.check('drilling A8 pushes its entry below (rates × 4096 × P6)',
  tip().querySelectorAll('.lv-cellent').length === 2
  && tip().textContent.includes('stash · dispatched tokens')
  && /× 4096 × P6/.test(tip().textContent), tip().textContent.slice(-120));
const p6ref = [...tip().querySelectorAll('.lv-cellent[data-k="1"] .cellref')].find(s => s.dataset.cell === 'P6');
p6ref.dispatchEvent(new MouseEvent('click', { bubbles: true })); await T.tick(30);
T.check('drilling P6 grows the stack to 3 (= P2 + 0.5 = 8.5)',
  tip().querySelectorAll('.lv-cellent').length === 3
  && tip().textContent.includes('microbatches in flight') && tip().textContent.includes('= P2 + 0.5 = 8.5'), '');
// one path only: clicking a ref in entry 0 truncates below it
const a2ref = tip().querySelector('.lv-cellent[data-k="0"] .cellref[data-cell="A2"]');
a2ref.dispatchEvent(new MouseEvent('click', { bubbles: true })); await T.tick(30);
T.check('clicking a ref in entry 0 truncates the path (stack of 2 again)',
  tip().querySelectorAll('.lv-cellent').length === 2, tip().querySelectorAll('.lv-cellent').length);
// clicking outside closes
layer().querySelector('.lv-scroll svg').dispatchEvent(new MouseEvent('click', { bubbles: true }));
await T.tick(30);
T.check('outside click unpins', tip().style.display === 'none', '');
// accordion sub-rows are cells too: solo the acts row, hover a bucket value
const md = (el2) => el2.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
md([...layer().querySelectorAll('.lv-bar g[data-prop]')][3]); await T.tick(700);
const pv = layer().querySelector('.lv-bar text[data-role="val:part:3:6"]');   // dispatched tokens → A8
T.check('acts sub-row carries its cell', pv?.dataset.cell === 'A8', pv?.dataset.cell);
mm(pv); await T.tick(30);
T.check('bucket hover: 0/1 recompute choice × save-everything rates', tip().textContent.includes('stash · dispatched tokens')
  && /= R8 × \(L1 × [\d.]+ \+ L2 × [\d.]+\) × 4096 × P6 = /.test(tip().textContent), tip().textContent.slice(0, 110));
md([...layer().querySelectorAll('.lv-bar g[data-prop]')][3]); await T.tick(700);   // un-solo
// the parents are accordion SUMS
mm(val('0')); await T.tick(30);
T.check('weights total = the sum of its accordion rows', tip().textContent.includes('= W2 + W3 + W4 = '), tip().textContent.slice(0, 90));

// the sheet: bound, live, and in agreement with the chart
const sheet = document.querySelector('dsv3-sheet');
await T.tick(200);
const rows = () => [...sheet.querySelectorAll('tr')].slice(1);
T.check('sheet renders all cells (sub-rows included)', rows().length >= 40, rows().length);
const t1row = rows().find(r => r.querySelector('.nm')?.textContent === 'T1');
T.check('sheet T1 equals the chart total', t1row.textContent.includes('61.7 GiB')
  && +val('total').dataset.true === 66296545344, t1row?.textContent);
// input rows for the formula switches: ZeRO level, fp8 params, per-bucket ↻
const rowOf = (id) => rows().find(r => r.querySelector('.nm')?.textContent === id)?.textContent;
T.check('Z1 / F1 / R rows present', rowOf('Z1')?.includes('ZeRO level') && rowOf('F1')?.includes('0/1')
  && rowOf('R8')?.includes('kept for backward'), '');
T.check('recompute choices read as 0/1 (dsv3: dispatched kept, norm outs replayed)',
  rowOf('R8')?.includes('1') && rowOf('R3')?.trim().endsWith('0'), `${rowOf('R3')}`);
// live: flip ZeRO off, the sub-cell formulas lose the sharding (O1 stays the accordion sum)
const zseg = layer().parentElement.querySelector('.stp[data-knob="zero"]');
[...zseg.querySelectorAll('button')].find(b => b.textContent === 'off').click(); await T.tick(700);
const o2 = () => rowOf('O2');
T.check('sheet is live: ZeRO off unshards the O2 formula', o2().includes('8 × Q1') && !o2().includes('/ P5'), o2());
[...layer().parentElement.querySelector('.stp[data-knob="zero"]').querySelectorAll('button')]
  .find(b => b.textContent === '1').click(); await T.tick(700);
T.check('and back', o2().includes('8 × Q1 / P5'), o2());
// the tooltip's bold coordinate jumps to (and highlights) the sheet row
mm(val('total'), 'click'); await T.tick(30);
tip().querySelector('b[data-jump="T1"]').dispatchEvent(new MouseEvent('click', { bubbles: true }));
await T.tick(50);
T.check('jump highlights the sheet row', sheet.querySelector('tr[data-cell="T1"]')?.classList.contains('hl'), '');
layer().querySelector('.lv-scroll svg').dispatchEvent(new MouseEvent('click', { bubbles: true }));
T.done();
