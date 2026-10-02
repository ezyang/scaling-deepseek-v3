// @page studies/03-roofline.html
// 03's cell sheets are live (studies/sheetedit.js): every row's formula
// column compiles and reproduces its published value exactly; the real-knob
// rows are click-to-type; dependents recompute everywhere they're repeated
// and turn amber; bad values are refused with a reason; edits live in the
// hash and the reset button's hover list names them and undoes them one by one.
await T.tick(300);
const { published, fmtExact } = await import('/studies/sheetedit.js');
const trs = [...document.querySelectorAll('.cellsheet tr')].filter((tr) => tr.querySelector('td.nm'));
const rowsOf = (txt) => trs.filter((tr) => tr.cells[0].textContent === txt);
const val = (txt) => rowsOf(txt)[0].cells[2].textContent;
const pubHTML = trs.map((tr) => tr.innerHTML);
const bad = trs.filter((tr) => fmtExact(published.get(tr.cells[0].innerHTML)) !== tr.cells[2].textContent);
T.check('every formula reproduces its published exact value', bad.length === 0, bad.map((tr) => tr.cells[0].textContent).join(' '));
const edv = [...document.querySelectorAll('.cellsheet td.vl.edv')].map((td) => td.parentElement.cells[0].textContent);
T.check('editable: exactly the real knobs', edv.join(' ') === 'πsolbf16 πsolfp8 βIB B S GPUs EP NVL', edv.join(' '));
T.check('untouched: no hash, nothing amber', location.hash === '' && !document.querySelector('.cellsheet tr.off'), location.hash);

const type = (txt, s, key = 'Enter') => {
  rowsOf(txt)[0].cells[2].click();
  const inp = rowsOf(txt)[0].querySelector('input.cs-in');
  inp.value = s;
  inp.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true }));
  return inp;
};
type('GPUs', '4096');
const si = (txt) => rowsOf(txt)[0].cells[4].textContent;
T.check('GPUs 4096: T_c halves (SI column re-prefixed)', si('Tc') === '2.32 s', si('Tc'));
T.check('… a rounded value keeps no exact entry', val('Tc') === '', val('Tc'));
T.check('… repeated rows follow (T+c in three sheets)', rowsOf('T+c').every((tr) => tr.cells[4].textContent === si('T+c') && tr.classList.contains('off')), rowsOf('T+c').length);
T.check('… independent rows stay published (N, B)', !rowsOf('N')[0].classList.contains('off') && val('B') === '15,360', '');
T.check('… hash carries only the departure', decodeURIComponent(location.hash) === '#c:cells={"GPUs":4096}', decodeURIComponent(location.hash));
T.check('… and lights the reset dot', T.el('.resetb').classList.contains('mod'), '');

type('πsolfp8', '2913.2T');
T.check('SI suffix: 2913.2T = 2,913,200,000,000,000', val('πsolfp8') === '2,913,200,000,000,000', val('πsolfp8'));
T.check('… π^sol_fp8 drives T_c (and η)', si('Tc') === '1.16 s' && si('ηfp8') === '147 %', si('Tc') + ' ' + si('ηfp8'));
const inp = type('EP', '16');
T.check('EP 16 refused: input stays open, flagged', inp.isConnected && inp.classList.contains('bad'), '');
T.check('… with the reason', T.el('.cs-err').style.display === 'block' && /4 nodes/.test(T.el('.cs-err').textContent), T.el('.cs-err').textContent);
inp.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
T.check('Esc: value back, error gone', val('EP') === '64' && T.el('.cs-err').style.display === 'none', val('EP'));
type('NVL', '3');
T.check('NVL 3 refused (must divide EP)', /divide EP/.test(T.el('.cs-err').textContent), T.el('.cs-err').textContent);
document.activeElement.blur();
T.check('blur on a bad value drops it', val('NVL') === '8' && T.el('.cs-err').style.display === 'none', val('NVL'));

// the reset button's hover list
const lines = [...document.querySelectorAll('.resetc .ln')];
T.check('reset list: one line per edit', lines.map((l) => l.querySelector('.go').textContent).join(' | ') === 'GPUs 2,048 → 4,096 | πsolfp8 1,456,600,000,000,000 → 2,913,200,000,000,000', lines.map((l) => l.textContent).join(' | '));
T.check('hidden until the button is hovered', getComputedStyle(T.el('.resetp')).display === 'none', '');
lines[1].querySelector('.un').click();
T.check('undo one: π back, GPUs kept', val('πsolfp8') === '1,456,600,000,000,000' && val('GPUs') === '4,096' && document.querySelectorAll('.resetc .ln').length === 1, '');
type('GPUs', '2,048');
T.check('typing the published value back: hash clean, every row restored verbatim',
  location.hash === '' && trs.every((tr, i) => tr.innerHTML === pubHTML[i]) && !T.el('.resetb').classList.contains('mod'), location.hash + ' ' + trs.filter((tr, i) => tr.innerHTML !== pubHTML[i]).map((tr) => tr.innerHTML).join(' // '));
// humanized input: an SI prefix plus the row's own unit, loosely spelled
for (const s of ['800 TFLOP/s', '800TFLOPs', '0.8 pflops', '800e12 FLOP/sec']) {
  type('πsolbf16', s);
  T.check(`"${s}" reads as 800 TFLOP/s`, val('πsolbf16') === '800,000,000,000,000' && rowsOf('πsolbf16')[0].cells[4].textContent === '800 TFLOP/s', val('πsolbf16'));
}
type('βIB', '25 GBps');
T.check('"25 GBps" reads as 25 GB/s', val('βIB') === '25,000,000,000', val('βIB'));
for (const [row, s, why] of [['βIB', '400 Gb/s', /B\/s/], ['πsolfp8', '50 GB/s', /FLOP\/s/], ['EP', '32 GPUs', /no unit/]]) {
  const el = type(row, s);
  T.check(`"${s}" refused for ${row}`, el.isConnected && why.test(T.el('.cs-err').textContent), T.el('.cs-err').textContent);
  el.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
}
type('πsolbf16', '757.7 TFLOP/s'); type('βIB', '50 GB/s');
T.check('published values typed back with units: hash clean', location.hash === '', location.hash);
T.done();
