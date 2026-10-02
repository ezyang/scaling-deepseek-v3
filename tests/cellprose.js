// @page studies/03-roofline.html
// prose numbers quoting a cell (<span data-cellref="row name">): the published
// text matches the row's SI column, sheetedit keeps it in step with knob
// edits (amber while off), and a click jumps to (selects) the row
await T.tick(100);
const qs = [...document.querySelectorAll('main [data-cellref]')];
const rowOf = (q) => [...document.querySelectorAll('.cellsheet tr')].find((tr) => tr.cells[0]?.innerHTML === q.dataset.cellref);
const si = (q) => rowOf(q)?.cells[4].textContent.replace(/\s+/g, ' ').trim();
const txt = (q) => q.textContent.replace(/\s+/g, ' ').trim();
T.check('every quoted cell exists and reads as its SI column', qs.length > 0 && qs.every((q) => rowOf(q) && txt(q) === si(q)), qs.map((q) => `${txt(q)} | ${si(q)}`).join('; '));
const dt = qs.find((q) => q.dataset.cellref === 'ΔT<sub>bf16</sub>');
T.check('BF16 adjustment quoted: 121 ms', txt(dt) === '121 ms', txt(dt));
const type = (name, s) => {
  const tr = [...document.querySelectorAll('.cellsheet tr')].find((r) => r.cells[0]?.textContent === name);
  tr.cells[2].click();
  const inp = tr.querySelector('input.cs-in'); inp.value = s;
  inp.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
};
type('GPUs', '4096');
T.check('GPUs 4096: the quote halves with its row, amber', txt(dt) === '60.3 ms' && txt(dt) === si(dt) && dt.classList.contains('off'), txt(dt));
type('GPUs', '2048');
T.check('back to published: verbatim, not amber', txt(dt) === '121 ms' && !dt.classList.contains('off'), txt(dt));
dt.dispatchEvent(new MouseEvent('mousemove', { clientX: 10, clientY: 10 }));
const tip = [...document.querySelectorAll('body > .cs-tip')].find((t) => t.style.display === 'block');
T.check('hover shows the cell card', /^ΔTbf16 · step time added by BF16 pricing/.test(tip?.textContent ?? '') && /\(vs every GEMM at FP8\) ≈ 121 ms$/.test(tip.textContent), tip?.textContent);
dt.click();
T.check('click selects the row', rowOf(dt).classList.contains('sel'), '');
T.done();
