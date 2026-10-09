// @page studies/h100-16b.html
// The DeepSeek-V3 16B / H100 page (sheets only) rides studies/sheetedit.js like 03:
// every formula reproduces its published value, the HTML's numbers are what the
// formulas render (the generator in refs wrote them), the page's knob rules apply,
// and the cluster-size table is built from the sheet's formulas and follows the knobs.
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
T.check('editable: exactly this page\'s knobs', edv.join(' ') === 'πsolbf16 βNV βIB B S GPUs fdoc ra2a', edv.join(' '));
T.check('untouched: no hash, nothing amber', location.hash === '' && !document.querySelector('.cellsheet tr.off'), location.hash);
// the anchors: the job log's parameter counts, TorchTitan's FLOPs per token, the measured rows
T.check('params = 15,706,484,224 total, 2,451,435,008 activated (job log)', val('N') === '15,706,484,224' && val('Nact') === '2,451,435,008', `${val('N')} ${val('Nact')}`);
T.check('TorchTitan FLOPs/token = 18,105,996,288 (its tflops ÷ tps)', val('φtt') === '18,105,996,288', val('φtt'));
T.check('measured 4,194.6 tok/s/GPU, 24.84 GiB peak reserved', val('ρ') === '4,194.6' && val('Mmeas') === '24.84', `${val('ρ')} ${val('Mmeas')}`);
const cfg = () => [...document.querySelectorAll('.configs tbody tr')].map((tr) => [...tr.cells].map((c) => c.textContent));
const c0 = cfg();
T.check('cluster table: 256 and 32 GPUs, measured 4,195 and 7,155, SOL over measured ≈ 9× and ≈ 5×',
  c0.length === 2 && c0[0][6] === '4,195' && c0[1][6] === '7,155' && /^9\.\d×$/.test(c0[0][8]) && /^5\.\d×$/.test(c0[1][8]), JSON.stringify(c0));
T.check('memory tally within 15 % of the measured peak at both sizes',
  c0.every((r) => Math.abs(parseFloat(r[1]) / parseFloat(r[2]) - 1) < 0.15), JSON.stringify(c0.map((r) => [r[1], r[2]])));
// knobs: saving the all-to-alls cuts NVLink time; a rule rejects a partial node
const cell = (nm) => rowsOf(nm)[0].cells[2];
const type = (nm, s) => {
  cell(nm).click();
  const inp = document.querySelector('.cellsheet .cs-in');
  inp.value = s; inp.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
};
const fNV = () => rowsOf('fNV')[0].cells[4].textContent;
const nv0 = fNV();
type('ra2a', '0');
T.check('r_a2a = 0 (all-to-alls saved): NVLink share drops', parseFloat(fNV()) < parseFloat(nv0), `${nv0} → ${fNV()}`);
type('ra2a', '1');
type('GPUs', '100');
const err = document.querySelector('.cs-err')?.textContent ?? '';
document.querySelector('.cellsheet .cs-in')?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
await T.tick();
T.check('GPUs 100 is rejected (whole nodes) and the cell keeps 256', err.includes('nodes') && val('GPUs') === '256', `${err} · ${val('GPUs')}`);
T.check('back to untouched', location.hash === '' && JSON.stringify(cfg()) === JSON.stringify(c0), location.hash);
T.done();
