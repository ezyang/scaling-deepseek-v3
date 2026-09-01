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
  && tip().textContent.includes('dispatched tokens')
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
T.check('bucket hover: 0/1 recompute choice × dims × the B• precision input',
  tip().textContent.includes('dispatched tokens')
  && tip().textContent.includes('= R8 × (L1 × (8×7168 × B8)) × 4096 × P6 = '), tip().textContent.slice(0, 130));
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
const valOf = (id) => rows().find(r => r.querySelector('.nm')?.textContent === id)?.querySelector('td.vl')?.textContent.trim();
T.check('input rows present: Z1 / S / F1 / R / B / E-H', rowOf('Z1')?.includes('ZeRO level') && rowOf('F1')?.includes('0/1')
  && rowOf('R8')?.includes('kept?') && rowOf('S5')?.includes('shard group · optimizer')
  && rowOf('B8')?.includes('precision (B/elem)') && rowOf('H1')?.includes('lm head'), '');
T.check('recompute choices read as 0/1 (dsv3: dispatched kept, norm1 replayed)',
  valOf('R8') === '1' && valOf('R3a') === '0', `${valOf('R3a')}`);
// breakout buckets: per-TENSOR rows, each a whole 0/1 — the motivating case
// is attn-replay keeping norm2 (the anchor) while norm1 replays
T.check('norms broken out individually (+ their rstds as own rows)', rowOf('A3')?.includes('A3a + A3b + A3c + A3d')
  && rowOf('A3a')?.includes('norm1 out') && rowOf('A3b')?.includes('rstd (fp32)')
  && rowOf('A3c')?.includes('norm2 out'), rowOf('A3'));
T.check('an aux is gated by ITS TENSOR’s kept? (rstd reads R3c; no R3d row)',
  rowOf('A3d')?.includes('R3c ×') && !rows().some(r => r.querySelector('.nm')?.textContent === 'R3d'), rowOf('A3d'));
T.check('alternate names: dispatched = routed experts’ input, norm2 out = shared expert input',
  rowOf('A8')?.includes('(routed experts’ input)') && rowOf('A3c')?.includes('(shared expert input)'), '');
T.check('residual broken out (x0 pinned / x1)', rowOf('A2')?.includes('A2a + A2b')
  && rowOf('A2a')?.includes('x0'), rowOf('A2'));
{
  const rseg = () => layer().parentElement.querySelector('.stp[data-knob="recompute"]');
  const rp = async (k) => { [...rseg().querySelectorAll('button')].find(b => b.textContent === k).click(); await T.tick(700); };
  const fxA = () => [...sheet.querySelectorAll('tr')].slice(1)
    .map(r => `${r.querySelector('.nm')?.textContent}:${r.querySelector('.fx')?.textContent}`).join('|');
  const f0 = fxA();
  await rp('attn-replay');
  T.check('attn-replay splits the norms (norm1 replayed, norm2 kept) with formulas UNCHANGED',
    fxA() === f0 && valOf('R3a') === '0' && valOf('R3c') === '1', '');
  await rp('full');
  T.check('full: formulas unchanged too (x0 stays kept, x1 flips)',
    fxA() === f0 && valOf('R2a') === '1' && valOf('R2b') === '0', '');
  await rp('dsv3');
}
T.check('low precision is legible: attn-out references B6a (e5m6 1.5), the lse split into its own row',
  rowOf('A6a')?.includes('128×128 × B6a') && rowOf('B6a')?.includes('1.5 B/elem')
  && rowOf('A6b')?.includes('lse (fp32)'), rowOf('A6a')?.slice(0, 120));
// the STABILITY AUDIT: toggling model inputs must never change a formula —
// only input VALUES move (Z1→S•, F1, recipe/ᵀ/E5M6→B•). Capture every
// formula, toggle the byte-side knobs, and diff.
const fxAll = () => [...sheet.querySelectorAll('tr')].slice(1)
  .map(r => `${r.querySelector('.nm')?.textContent}:${r.querySelector('.fx')?.textContent}`).join('|');
const fx0 = fxAll();
const zseg = () => layer().parentElement.querySelector('.stp[data-knob="zero"]');
const zpick = async (k) => { [...zseg().querySelectorAll('button')].find(b => b.textContent === k).click(); await T.tick(700); };
await zpick('off');
T.check('ZeRO off: formulas unchanged, S5 flips 4 → 1', fxAll() === fx0 && valOf('S5') === '1', valOf('S5'));
await zpick('1');
const p8 = layer().parentElement.querySelector('input[data-knob="fp8params"]');
p8.click(); await T.tick(700);
T.check('fp8 params on: formulas unchanged, F1 = 1', fxAll() === fx0 && valOf('F1') === '1', valOf('F1'));
p8.click(); await T.tick(400);
const tcb = layer().parentElement.querySelector('input[data-knob="transposed"]');
tcb.click(); await T.tick(700);
T.check('e4m3ᵀ on: formulas unchanged, B8 doubles (dual folded into the input)',
  fxAll() === fx0 && rowOf('B8')?.includes('2.0625 B/elem'), rowOf('B8'));
tcb.click(); await T.tick(400);
const rpick = async (k) => {
  [...layer().parentElement.querySelector('.stp[data-knob="recipe"]').querySelectorAll('button')]
    .find(b => b.textContent === k).click();
  await T.tick(700);
};
await rpick('bf16');
T.check('recipe bf16: formulas unchanged, B8 = 2 B/elem', fxAll() === fx0 && rowOf('B8')?.includes('2 B/elem'), rowOf('B8'));
await rpick('dsv3-fp8');
// the tooltip's bold coordinate jumps to (and highlights) the sheet row
mm(val('total'), 'click'); await T.tick(30);
tip().querySelector('b[data-jump="T1"]').dispatchEvent(new MouseEvent('click', { bubbles: true }));
await T.tick(50);
T.check('jump highlights the sheet row', sheet.querySelector('tr[data-cell="T1"]')?.classList.contains('hl'), '');
layer().querySelector('.lv-scroll svg').dispatchEvent(new MouseEvent('click', { bubbles: true }));
T.done();
