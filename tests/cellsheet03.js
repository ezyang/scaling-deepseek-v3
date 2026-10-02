// @page studies/03-roofline.html
// 03's static cell sheets behave like 02's <dsv3-sheet>: hovering a formula
// name shows its cell card; clicking it jumps to (and selects) the row that
// defines it, across sheets (the closing tally's cell column too); clicking
// a row selects it, again clears; a click on a name never pins the card
await T.tick(100);
const at = (el, type, o = {}) => { const q = el.getBoundingClientRect(); el.dispatchEvent(new MouseEvent(type, { bubbles: true, clientX: q.left + 2, clientY: q.top + 2, ...o })); };
const sheets = [...document.querySelectorAll('.cellsheet')];
const refs = [...document.querySelectorAll('.cellsheet td.fx b')];
T.check('every formula name resolves to a defining row', refs.every((b) => b.classList.contains('ref')), refs.filter((b) => !b.classList.contains('ref')).map((b) => b.innerHTML).join(' '));
const tal = [...document.querySelectorAll('.tally td.fx b')];
T.check('the closing tally cites a sheet cell on every row', tal.length === 7 && tal.every((b) => b.classList.contains('ref')), tal.filter((b) => !b.classList.contains('ref')).map((b) => b.innerHTML).join(' ') || tal.length);
// T_fp8 = (C − C_bf16) / (GPUs · π^sol_fp8), in the precision sheet: hover GPUs (defined in the 6ND sheet; the Smol sheet precedes it)
const si = sheets.findIndex((sh) => [...sh.querySelectorAll('tr')].some((tr) => tr.cells[0]?.innerHTML === 'N<sub>bf16</sub>'));
const t2 = [...sheets[si].querySelectorAll('tr')].find((tr) => tr.cells[0]?.innerHTML === 'T<sub>fp8</sub>');
const g = [...t2.querySelectorAll('td.fx b')].find((b) => b.textContent === 'GPUs');
at(g, 'mousemove');
const tips = document.querySelectorAll('body > .cs-tip');   // one per sheet, in order
const tip = tips[si];
T.check('hover: the card names the cell, its quantity and value', tip.style.display === 'block' && tip.textContent === 'GPUs · GPUs in the cluster= 2,048', tip.textContent);
const cg = [...sheets[2].querySelectorAll('td.fx b')].find((b) => b.innerHTML === 'D');
at(cg, 'mousemove');
const tip0 = tips[2];
T.check('hover: a derived cell shows formula = exact ≈ SI', tip0.textContent === 'D · tokens per step= B · S = 62,914,560 ≈ 62.9 M', tip0.textContent);
sheets[2].dispatchEvent(new MouseEvent('mouseleave'));
T.check('mouseleave hides it', tip0.style.display === 'none', '');
// click a name → the defining row (another sheet) is selected, no pin
getSelection().removeAllRanges();
at(g, 'click', { detail: 1 });
const gRow = [...sheets[2].querySelectorAll('tr')].find((tr) => tr.cells[0]?.textContent === 'GPUs');
T.check('click a name: its defining row is selected', gRow.classList.contains('sel') && document.querySelectorAll('.cellsheet tr.sel').length === 1, '');
T.check('click a name: the card does not pin', !tip.classList.contains('pinned'), tip.className);
const r = gRow.getBoundingClientRect();
T.check('click a name: the row is scrolled into view', r.top >= 0 && r.bottom <= innerHeight, `${r.top}`);
// row clicks: select, then clear (250 ms fuse)
at(t2.cells[1], 'click', { detail: 1 });
await T.tick(300);
T.check('row click selects that row alone', t2.classList.contains('sel') && !gRow.classList.contains('sel'), '');
at(t2.cells[1], 'click', { detail: 1 });
await T.tick(300);
T.check('clicking the selected row clears it', !document.querySelector('.cellsheet tr.sel'), '');
T.done();
