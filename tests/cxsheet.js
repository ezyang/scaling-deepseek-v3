// @page studies/context-extension.html
// The context-extension roofline (sheets only) rides studies/sheetedit.js like 03:
// every formula reproduces its published value, the HTML's numbers are what the
// formulas render (the generator in refs wrote them), and the phase table is read
// off the sheet and follows the knobs.
await T.tick(300);
const { published, fmtExact } = await import('/studies/sheetedit.js');
const trs = [...document.querySelectorAll('.cellsheet tr')].filter((tr) => tr.querySelector('td.nm'));
const rowsOf = (txt) => trs.filter((tr) => tr.cells[0].textContent === txt);
const val = (txt) => rowsOf(txt)[0].cells[2].textContent;
const bad = trs.filter((tr) => fmtExact(published.get(tr.cells[0].innerHTML)) !== tr.cells[2].textContent);
T.check('every formula reproduces its published exact value', bad.length === 0, bad.map((tr) => tr.cells[0].textContent).join(' '));
const raw = new DOMParser().parseFromString(await (await fetch(location.pathname)).text(), 'text/html');
const txt = (el) => el.textContent.replace(/\s+/g, ' ').trim();
const rawTrs = [...raw.querySelectorAll('.cellsheet tr')].filter((tr) => tr.querySelector('td.nm'));
const stale = rawTrs.flatMap((tr, i) => [2, 3, 4].filter((c) => txt(tr.cells[c]) !== txt(trs[i].cells[c]))
  .map((c) => `${txt(tr.cells[0])} col ${c}: ${txt(tr.cells[c])} → ${txt(trs[i].cells[c])}`));
const refs = [...document.querySelectorAll('[data-cellref]')], rawRefs = [...raw.querySelectorAll('[data-cellref]')];
rawRefs.forEach((q, i) => { if (txt(q) !== txt(refs[i])) stale.push(`prose ${q.dataset.cellref}: ${txt(q)} → ${txt(refs[i])}`); });
T.check('the HTML\'s numbers (every column, every quoting prose span) are what the formulas render',
  rawTrs.length === trs.length && rawRefs.length === refs.length && stale.length === 0, stale.join(' | '));
const names = [...document.querySelectorAll('.cellsheet td.fx b')];
T.check('every formula name resolves to a defining row', names.every((b) => b.classList.contains('ref')), names.filter((b) => !b.classList.contains('ref')).map((b) => b.innerHTML).join(' '));
const edv = [...document.querySelectorAll('.cellsheet td.vl.edv')].map((td) => td.parentElement.cells[0].textContent);
T.check('editable: exactly this page\'s knobs', edv.join(' ') === 'πsolbf16 πsolfp8 βIB βsolNV', edv.join(' '));
T.check('untouched: no hash, nothing amber', location.hash === '' && !document.querySelector('.cellsheet tr.off'), location.hash);
// the anchors: DeepSeek's recipe keeps D fixed; part 3's numbers carry over
T.check('all three phases train 62,914,560 tokens per step', ['D', 'D1', 'D2'].every((k) => val(k) === '62,914,560'));
T.check('CP 8 and 32 keep 60 microbatches per replica', val('CP1') === '8' && val('CP2') === '32' && ['m0', 'm1', 'm2'].every((k) => val(k) === '60'));
T.check('pretraining attention time = part 3\'s T_attn (1.49 s)', /^1\.49/.test(rowsOf('T0attn')[0].cells[4].textContent), rowsOf('T0attn')[0].cells[4].textContent);
const tab = () => [...document.querySelectorAll('.phases tbody tr')].map((tr) => [...tr.cells].map((c) => c.textContent));
const t0 = tab();
T.check('phase table: three rows, S and CP from the sheet', t0.length === 3 && t0.map((r) => r[1] + '/' + r[3]).join(' ') === '4,096/1 32,768/8 131,072/32', JSON.stringify(t0));
// a knob edit flows into the table
const cell = (nm) => rowsOf(nm)[0].cells[2];
const type = (nm, s) => {
  cell(nm).click();
  const inp = document.querySelector('.cellsheet .cs-in');
  inp.value = s; inp.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
};
type('πsolbf16', '989,000,000,000,000');
await T.tick();
T.check('a faster BF16 rate shrinks phase 2\'s attention time in the table', parseFloat(tab()[2][5]) < parseFloat(t0[2][5]), `${t0[2][5]} → ${tab()[2][5]}`);
type('πsolbf16', '757,700,000,000,000');
await T.tick();
T.check('back to untouched', location.hash === '' && JSON.stringify(tab()) === JSON.stringify(t0), location.hash);
T.done();
