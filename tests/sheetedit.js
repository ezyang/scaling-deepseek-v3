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
T.check('editable: exactly the real knobs', edv.join(' ') === 'B S GPUs πbf16 EP NVL βIB', edv.join(' '));
T.check('untouched: no hash, nothing amber', location.hash === '' && !document.querySelector('.cellsheet tr.off'), location.hash);

const type = (txt, s, key = 'Enter') => {
  rowsOf(txt)[0].cells[2].click();
  const inp = rowsOf(txt)[0].querySelector('input.cs-in');
  inp.value = s;
  inp.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true }));
  return inp;
};
type('GPUs', '4096');
T.check('GPUs 4096: T_c halves', val('Tc') === fmtExact([published.get('T<sub>c</sub>')[0], published.get('T<sub>c</sub>')[1] * 2n]), val('Tc'));
T.check('… repeated rows follow (T+c in three sheets)', rowsOf('T+c').every((tr) => tr.cells[2].textContent === val('T+c') && tr.classList.contains('off')), rowsOf('T+c').length);
T.check('… SI column re-prefixed', rowsOf('Tc')[0].cells[4].textContent === '3.41 s', rowsOf('Tc')[0].cells[4].textContent);
T.check('… independent rows stay published (N, B)', !rowsOf('N')[0].classList.contains('off') && val('B') === '15,360', '');
T.check('… hash carries only the departure', decodeURIComponent(location.hash) === '#c:cells={"GPUs":4096}', decodeURIComponent(location.hash));
T.check('… and lights the reset dot', T.el('.resetb').classList.contains('mod'), '');

type('πbf16', '1979T');
T.check('SI suffix: 1979T = 1,979,000,000,000,000', val('πbf16') === '1,979,000,000,000,000', val('πbf16'));
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
T.check('reset list: one line per edit', lines.map((l) => l.querySelector('.go').textContent).join(' | ') === 'GPUs 2,048 → 4,096 | πbf16 989,000,000,000,000 → 1,979,000,000,000,000', lines.map((l) => l.textContent).join(' | '));
T.check('hidden until the button is hovered', getComputedStyle(T.el('.resetp')).display === 'none', '');
lines[1].querySelector('.un').click();
T.check('undo one: π back, GPUs kept', val('πbf16') === '989,000,000,000,000' && val('GPUs') === '4,096' && document.querySelectorAll('.resetc .ln').length === 1, '');
type('GPUs', '2,048');
T.check('typing the published value back: hash clean, every row restored verbatim',
  location.hash === '' && trs.every((tr, i) => tr.innerHTML === pubHTML[i]) && !T.el('.resetb').classList.contains('mod'), location.hash + ' ' + trs.filter((tr, i) => tr.innerHTML !== pubHTML[i]).map((tr) => tr.innerHTML).join(' // '));
T.done();
