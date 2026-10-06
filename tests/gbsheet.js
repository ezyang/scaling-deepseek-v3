// @page studies/blackwell-roofline.html
// The Blackwell roofline page (sheets first) rides studies/sheetedit.js like 03:
// every formula reproduces its published value, the HTML's numbers are what the
// formulas render (the generator in refs wrote them), the page's knob rules apply,
// and the configuration table is built from the sheet's formulas and follows the knobs.
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
T.check('editable: exactly this page\'s knobs', edv.join(' ') === 'πsolbf16 πsolfp8 βNV βIB NVL B S GPUs a8 PP VP EP', edv.join(' '));
T.check('untouched: no hash, nothing amber', location.hash === '' && !document.querySelector('.cellsheet tr.off'), location.hash);
// the anchors: part 3's checkpoint tally, NVIDIA's published rows
T.check('activated params = 36,625,618,432 (part 3)', val('N') === '36,625,618,432', val('N'));
T.check('BF16 GEMMs are the 58 routers only', val('Nbf16') === '106,430,464', val('Nbf16'));
T.check('reported 6,288 tok/s/GPU; MLPerf median step 39.355 s', val('ρNV') === '6,288' && val('TML') === '39.355', `${val('ρNV')} ${val('TML')}`);
T.check('interleaved bubble at PP 2, VP 8, 32 microbatches = 1/256', val('bILV') === '0.00390625', val('bILV'));
const cfg = () => [...document.querySelectorAll('.configs tbody tr')].map((tr) => [...tr.cells].map((c) => c.textContent));
const c0 = cfg();
T.check('configuration table: the benchmark and MLPerf rows, SOL over measured ≈ 2.2× each',
  c0.length === 2 && c0[0][7] === '6,288' && c0[1][7] === '6,245' && c0.every((r) => /^2\.2\d×$/.test(r[8])), JSON.stringify(c0));
// knobs: edit a hardware rate and the table follows; a rule rejects an EP outside the domain
const cell = (nm) => rowsOf(nm)[0].cells[2];
const type = (nm, s) => {
  cell(nm).click();
  const inp = document.querySelector('.cellsheet .cs-in');
  inp.value = s; inp.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
};
type('a8', '0');
T.check('a₈ = 0 (BF16 attention): compute time grows, the table re-builds', parseFloat(cfg()[0][3]) > parseFloat(c0[0][3]), `${c0[0][3]} → ${cfg()[0][3]}`);
type('a8', '1');
type('EP', '128');
const err = document.querySelector('.cs-err')?.textContent ?? '';
document.querySelector('.cellsheet .cs-in')?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
await T.tick();
T.check('EP 128 is rejected (EP must divide NVL) and the cell keeps 32', err.includes('NVL') && val('EP') === '32', `${err} · ${val('EP')}`);
T.check('back to untouched', location.hash === '' && JSON.stringify(cfg()) === JSON.stringify(c0), location.hash);
T.done();
