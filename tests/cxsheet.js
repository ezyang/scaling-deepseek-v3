// @page studies/context-extension.html
// The context-extension roofline (sheets only) rides studies/sheetedit.js like 03:
// every formula reproduces its published value, the HTML's numbers are what the
// formulas render (the generator in refs wrote them), the knob rules apply, and the CP
// sweep runs through the sheet's own formulas and pins the page's verdicts.
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
T.check('editable: exactly this page\'s knobs', edv.join(' ') === 'πsolbf16 πsolfp8 βIB frc CP1 CP2 βsolNV', edv.join(' '));
T.check('untouched: no hash, nothing amber', location.hash === '' && !document.querySelector('.cellsheet tr.off'), location.hash);
// the anchors: DeepSeek's recipe keeps D fixed; part 3's numbers carry over
T.check('all three phases train 62,914,560 tokens per step', ['D', 'D1', 'D2'].every((k) => val(k) === '62,914,560'));
T.check('pretraining attention time = part 3\'s T_attn (1.49 s)', /^1\.49/.test(rowsOf('T0attn')[0].cells[4].textContent), rowsOf('T0attn')[0].cells[4].textContent);
T.check('the trace\'s FlashAttention kernels: 1,237 µs + 3,778.75 µs', val('tFfa') === '0.001237' && val('tBfa') === '0.00377875');
T.check('default CP 8 / 32 keep 60 microbatches', val('CP1') === '8' && val('CP2') === '32' && val('m1') === '60' && val('m2') === '60');
// finding 2: the K/V floors over InfiniBand at CP 32 vs the latent's (no compute charged)
T.check('finding 2: K/V floors 88 % (causal) and 106 % (every slice) of the reported pair; latent 3.1 s', val('CPmin2') === '32' && /^88\.1/.test(rowsOf('rIBkvc')[0].cells[4].textContent) && /^106/.test(rowsOf('rIBkv')[0].cells[4].textContent) && /^3\.11/.test(rowsOf('TIBlat')[0].cells[4].textContent), [rowsOf('rIBkvc')[0].cells[4].textContent, rowsOf('rIBkv')[0].cells[4].textContent].join(' '));
// the sweep and its verdicts (same scheme and recompute in both phases)
const sweep = () => [...document.querySelectorAll('.sweep tbody tr')].map((tr) => [...tr.cells].map((c) => c.textContent));
const fits = () => Object.fromEntries([...document.querySelectorAll('.fits tbody tr')].map((tr) => [tr.cells[0].textContent + ' · ' + tr.cells[1].textContent, tr.cells[4].textContent]));
const s0 = sweep(), f0 = fits();
T.check('sweep: CP 1…256, twelve columns per row', s0.length === 9 && s0.every((r) => r.length === 13) && s0[0][0] === '1' && s0[8][0] === '256', JSON.stringify(s0[0]));
T.check('sweep: CP 1 fits nowhere under either policy at 128K, CP 32 fits under DeepSeek\'s', s0[0][9].startsWith('— / —') && !s0[5][9].startsWith('—'), `${s0[0][9]} | ${s0[5][9]}`);
T.check('Ulysses fits the 119K under neither policy', f0['head-parallel (Ulysses) · DeepSeek’s'] === 'none' && f0['head-parallel (Ulysses) · full'] === 'none', JSON.stringify(f0));
T.check('K/V passing, every slice: none with DeepSeek\'s recompute, CP 16/16 with full', f0['K/V passing, every slice · DeepSeek’s'] === 'none' && f0['K/V passing, every slice · full'].startsWith('16/16 '), JSON.stringify(f0));
T.check('K/V passing, causal sends: outside ±5 % under both policies', f0['K/V passing, causal sends · DeepSeek’s'] === 'none' && f0['K/V passing, causal sends · full'] === 'none', JSON.stringify(f0));
T.check('latent all-gather fits a band (8/128 under DeepSeek\'s, 32/32 under full)', f0['latent all-gather · DeepSeek’s'].includes('8/128') && f0['latent all-gather · full'].includes('32/32'), JSON.stringify(f0));
// knobs: a rule rejects a CP that gives DualPipe an odd microbatch count; a rate edit re-runs the sweep
const cell = (nm) => rowsOf(nm)[0].cells[2];
const type = (nm, s) => {
  cell(nm).click();
  const inp = document.querySelector('.cellsheet .cs-in');
  inp.value = s; inp.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
};
type('CP2', '8');
const err = document.querySelector('.cs-err')?.textContent ?? '';
document.querySelector('.cellsheet .cs-in')?.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
await T.tick();
T.check('CP₂ = 8 is rejected (15 microbatches) and the cell keeps 32', err.includes('even') && val('CP2') === '32', `${err} · ${val('CP2')}`);
type('frc', '1');
await T.tick();
T.check('f_rc = 1: s_max grows past 65K; the sweep ignores the knob (it runs both policies)', parseFloat(rowsOf('smax')[0].cells[4].textContent) > 60 && JSON.stringify(sweep()) === JSON.stringify(s0), rowsOf('smax')[0].cells[4].textContent);
type('frc', '0');
type('πsolbf16', '989,000,000,000,000');
await T.tick();
T.check('a faster BF16 rate re-runs the sweep', JSON.stringify(sweep()) !== JSON.stringify(s0));
type('πsolbf16', '757,700,000,000,000');
await T.tick();
T.check('back to untouched', location.hash === '' && JSON.stringify(sweep()) === JSON.stringify(s0) && JSON.stringify(fits()) === JSON.stringify(f0), location.hash);
T.done();
