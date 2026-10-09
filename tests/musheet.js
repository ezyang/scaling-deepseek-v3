// @page studies/muon.html
// The Muon appendix (sheets only) rides studies/sheetedit.js like 03: every formula
// reproduces its published value, the HTML's numbers are what the formulas render (the
// generator in refs wrote them), part 3's numbers carry over, and a knob edit flows to
// the step-time shares.
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
const edv = [...document.querySelectorAll('.cellsheet td.vl.edv')].map((td) => td.parentElement.cells[0].textContent);
T.check('editable: exactly this page\'s knobs', edv.join(' ') === 'nNS πsolbf16 βIB βHBM πsolfp8', edv.join(' '));
T.check('untouched: no hash, nothing amber', location.hash === '' && !document.querySelector('.cellsheet tr.off'), location.hash);
T.check('AI-drafted prose is marked', document.querySelectorAll('main .ai').length > 10 && [...document.querySelectorAll('main p:not(.note)')].every((p) => p.closest('.ai') || p.querySelector('.sheet-dl')));
// the anchors: the matrix tally closes on part 3's checkpoint, part 3's times carry over
T.check('routed experts = part 3\'s N_exp; Muon gets 669,172,039,680 params', val('Nexp') === '653,908,770,816' && val('NMu') === '669,172,039,680');
T.check('part 3\'s T+c 6.39 s, T_step 6.64 s, T_IB 5.85 s, T^PP_sync 146 ms', /^6\.39/.test(si('T+c')) && /^6\.64/.test(si('Tstep')) && /^5\.85/.test(si('TIB')) && /^146/.test(si('TPPsync')), [si('T+c'), si('Tstep'), si('TIB'), si('TPPsync')].join(' '));
T.check('Newton–Schulz: 32.3 PFLOP, 0.234 % of 6ND; routed expert 0.397 %', /^32\.3 PFLOP/.test(si('CNS')) && /^0\.234/.test(si('φ')) && /^0\.397/.test(si('φe')), [si('CNS'), si('φ'), si('φe')].join(' '));
T.check('each matrix once: 29.9 ms (the dense FFN matrix bounds it); replicated 357 ms', /^29\.9/.test(si('TownNS')) && /^357/.test(si('TrepNS')), [si('TownNS'), si('TrepNS')].join(' '));
T.check('step shares: AdamW 0.0324 %, Muon once 0.482 %, replicated 5.41 %', /^0\.0324/.test(si('δAd')) && /^0\.482/.test(si('δown')) && /^5\.41/.test(si('δrep')), [si('δAd'), si('δown'), si('δrep')].join(' '));
T.check('the extra traffic hides: both IB totals stay under T+c', [si('TrepIB'), si('TownIB')].every((s) => parseFloat(s) < 6.39));
// a knob edit: ten Newton–Schulz iterations double the FLOPs and flow to the shares
const cell = (nm) => rowsOf(nm)[0].cells[2];
const type = (nm, s) => {
  cell(nm).click();
  const inp = document.querySelector('.cellsheet .cs-in');
  inp.value = s; inp.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
};
type('nNS', '10');
await T.tick();
T.check('n_NS = 10: 64.6 PFLOP, the replicated share grows past 10 %', /^64\.6 PFLOP/.test(si('CNS')) && parseFloat(si('δrep')) > 10 && location.hash.includes('nNS'), `${si('CNS')} · ${si('δrep')} · ${location.hash}`);
type('nNS', '5');
await T.tick();
T.check('back to untouched', location.hash === '' && /^5\.41/.test(si('δrep')), location.hash);
T.done();
