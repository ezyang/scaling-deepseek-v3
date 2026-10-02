// @page studies/kimi-k3-roofline.html
// The Kimi K3 roofline page (sheets only) rides studies/sheetedit.js like 03:
// every formula reproduces its published value, the HTML's numbers are what
// the formulas render, the parameter tally hits the advertised anchors to the
// digit, the page's own knob rules apply, and both summary tables are built
// from the sheet's formulas and follow the knobs.
await T.tick(300);
const { published, fmtExact } = await import('/studies/sheetedit.js');
const trs = [...document.querySelectorAll('.cellsheet tr')].filter((tr) => tr.querySelector('td.nm'));
const rowsOf = (txt) => trs.filter((tr) => tr.cells[0].textContent === txt);
const val = (txt) => rowsOf(txt)[0].cells[2].textContent;
const si = (txt) => rowsOf(txt)[0].cells[4].textContent;
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
// the parameter tally: the released checkpoint's text model, to the digit
T.check('total params = 2,779,484,476,000', val('Ntot') === '2,779,484,476,000', val('Ntot'));
T.check('activated params = 104,189,612,640', val('N') === '104,189,612,640', val('N'));
const edv = [...document.querySelectorAll('.cellsheet td.vl.edv')].map((td) => td.parentElement.cells[0].textContent);
T.check('editable: exactly this page\'s knobs', edv.join(' ') === 'η βNV βIB NVL B S GPUs PP EP rres gB oB', edv.join(' '));
T.check('untouched: no hash, nothing amber', location.hash === '' && !document.querySelector('.cellsheet tr.off'), location.hash);
// the default: 288 GPUs, PP 8 × EP 36, one replica — both policies fit, no ZeRO-1 traffic
T.check('default layout: DP 1, ZeRO-1 bytes 0', val('DP') === '1' && val('Vsync') === '0', `${val('DP')} ${val('Vsync')}`);
const pct = (txt) => parseFloat(si(txt));
T.check('both policies fit (≤ 100 %)', pct('fAmem') <= 100 && pct('fBmem') <= 100, `${si('fAmem')} ${si('fBmem')}`);
T.check('compute bound: NVLink and scale-out shares under 100 %', pct('fNV') < 100 && pct('fIB') < 100, `${si('fNV')} ${si('fIB')}`);

const type = (txt, s, key = 'Enter') => {
  rowsOf(txt)[0].cells[2].click();
  const inp = rowsOf(txt)[0].querySelector('input.cs-in');
  inp.value = s;
  inp.dispatchEvent(new KeyboardEvent('keydown', { key, bubbles: true }));
  return inp;
};
// the page's own rules
let inp = type('EP', '16');
T.check('EP 16 refused: must divide NVL', inp.isConnected && /divide NVL/.test(T.el('.cs-err').textContent), T.el('.cs-err').textContent);
inp.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
inp = type('GPUs', '300');
T.check('GPUs 300 refused: PP · EP must divide it', /PP · EP must divide/.test(T.el('.cs-err').textContent), T.el('.cs-err').textContent);
inp.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
// doubling the cluster: a second replica appears, ZeRO-1 traffic with it, policy A now fits
type('GPUs', '576');
T.check('GPUs 576: DP 2, ZeRO-1 bytes > 0, A fits', val('DP') === '2' && val('Vsync') !== '0' && pct('fAmem') <= 100, `${val('DP')} ${val('Vsync')} ${si('fAmem')}`);
T.check('… tokens/s per GPU (A) above the 288-GPU policy-B figure', pct('ρA') > 0 && rowsOf('ρA')[0].classList.contains('off'), si('ρA'));
T.check('… hash carries the departure', decodeURIComponent(location.hash) === '#c:cells={"GPUs":576}', decodeURIComponent(location.hash));
type('GPUs', '288');
T.check('typed back: hash clean', location.hash === '', location.hash);

// the frontier table: one row per cluster size, best fitting layout per policy, from the sheet's formulas
const fr = [...document.querySelectorAll('.frontier tbody tr')];
T.check('frontier table has rows', fr.length >= 8, fr.length);
const row288 = fr.find((tr) => tr.cells[0].textContent === '288');
T.check('288 GPUs: policy A PP 8 × EP 36 × DP 1 (the default), policy B PP 4 × EP 72 × DP 1', row288 && /PP 8 × EP 36 × DP 1/.test(row288.cells[2].textContent) && /PP 4 × EP 72 × DP 1/.test(row288.cells[5].textContent), row288?.textContent);
const row216 = fr.find((tr) => tr.cells[0].textContent === '216');
T.check('216 GPUs: policy A does not fit, policy B does', /does not fit/.test(row216.cells[2].textContent) && /PP \d+ × EP \d+/.test(row216.cells[3].textContent), row216.textContent);
const rhoB = (tr) => parseFloat(tr.cells[tr.cells.length - 1].textContent.replace(/,/g, ''));
const fitsB = fr.filter((tr) => !/does not fit/.test(tr.cells[tr.cells.length - 1].textContent));
T.check('tokens/s per GPU falls with cluster size once it fits (policy B)', fitsB.every((tr, i) => !i || rhoB(tr) <= rhoB(fitsB[i - 1])), fitsB.map(rhoB).join(' '));
const mins = [...document.querySelectorAll('.mins tbody tr')];
T.check('min-GPUs table: four gradient × optimizer combinations, each with a policy-B fit', mins.length === 4 && mins.every((tr) => /PP \d+ × EP \d+/.test(tr.cells[4].textContent)), mins.map((tr) => tr.textContent).join(' | '));
T.check('… FP32/FP32 row names the default: 288 · PP 8 × EP 36 for policy A, 216 for policy B', mins[3].cells[2].textContent === '288 · PP 8 × EP 36 × DP 1' && /^216 ·/.test(mins[3].cells[4].textContent), `${mins[3].cells[2].textContent} / ${mins[3].cells[4].textContent}`);
T.check('… BF16 states bring policy A down to 216', /^216 ·/.test(mins[2].cells[2].textContent), mins[2].cells[2].textContent);
// the frontier follows a knob: with BF16 optimizer states policy A fits at 216 GPUs
type('oB', '6');
const r216 = [...document.querySelectorAll('.frontier tbody tr')].find((tr) => tr.cells[0].textContent === '216');
T.check('o_B 6: the frontier re-builds, policy A now fits at 216', /PP \d+ × EP \d+/.test(r216.cells[2].textContent), r216.textContent);
type('oB', '8');
T.done();
