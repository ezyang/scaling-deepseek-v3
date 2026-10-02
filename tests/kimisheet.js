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
T.check('editable: exactly this page\'s knobs', edv.join(' ') === 'πsolbf16 πsolfp8 βNV βIB NVL B S GPUs PP EP Vpp ovl rres μ gB oB βhost', edv.join(' '));
T.check('untouched: no hash, nothing amber', location.hash === '' && !document.querySelector('.cellsheet tr.off'), location.hash);
// the default: 432 GPUs, PP 6 × EP 72, one replica, 10 % reserve — 1F1B fits both on the line; ZB-V-Min gives A room; no ZeRO-1 traffic
T.check('default layout: DP 1, ZeRO-1 bytes 0', val('DP') === '1' && val('Vsync') === '0', `${val('DP')} ${val('Vsync')}`);
const pct = (txt) => parseFloat(si(txt));
T.check('1F1B: both fit the usable memory (A within 1 %)', pct('fAmem') <= 100 && pct('fAmem') > 95 && pct('fBmem') <= 100, `${si('fAmem')} ${si('fBmem')}`);
const bytes = (txt) => { const t = si(txt); return parseFloat(t) * (t.endsWith('GB') ? 1e9 : 1); };
T.check('ZB-V-Min: 3 microbatches in flight, not 3.5; policy A under usable memory', val('nVminflt') === '3' && bytes('MAVmin') <= bytes('Muse'), `${val('nVminflt')} ${si('MAVmin')} ${si('Muse')}`);
T.check('usable memory = device memory less the reserve', val('μ') === '0.1' && bytes('Muse') < bytes('MGPU'), `${si('Muse')} ${si('MGPU')}`);
T.check('self-overlap default: EP traffic under the expert GEMMs, nothing exposed', val('ovl') === '1' && pct('fselfEP') < 100 && val('TxEP') === '0', `${si('fselfEP')} ${val('TxEP')}`);
T.check('offload: policy B\'s stash fits the host link inside compute, A\'s does not; the optimizer prefetch spans several microbatches', pct('φB') > 100 && pct('φA') < 100 && +val('mpf') > 1, `${si('φA')} ${si('φB')} ${val('mpf')}`);
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
// doubling the cluster: a second replica appears, ZeRO-1 traffic with it, memory eases
type('GPUs', '864');
T.check('GPUs 864: DP 2, ZeRO-1 bytes > 0, A well under', val('DP') === '2' && val('Vsync') !== '0' && pct('fAmem') < 90, `${val('DP')} ${val('Vsync')} ${si('fAmem')}`);
T.check('… ρ^A re-rendered amber', pct('ρA') > 0 && rowsOf('ρA')[0].classList.contains('off'), si('ρA'));
T.check('… hash carries the departure', decodeURIComponent(location.hash) === '#c:cells={"GPUs":864}', decodeURIComponent(location.hash));
type('GPUs', '432');
// the overlap knob: starve NVLink so EP traffic exceeds the expert GEMMs; self-overlap exposes the excess, fwd-bwd overlap hides it
const rhoA0 = pct('ρA');
type('βNV', '300 GB/s');
T.check('β_NV 300 GB/s: EP traffic over 100 % of the expert GEMMs, exposed time > 0, ρ^A falls', pct('fselfEP') > 100 && val('TxEP') !== '0' && pct('ρA') < rhoA0, `${si('fselfEP')} ${si('TxEP')} ${si('ρA')}`);
type('ovl', '0');
T.check('ovl 0: nothing exposed (max(0, ·) via the sheet language), ρ^A back', val('TxEP') === '0' && pct('ρA') === rhoA0, `${val('TxEP')} ${si('ρA')}`);
type('ovl', '1'); type('βNV', '900 GB/s');
T.check('restored: hash clean', location.hash === '', location.hash);
T.check('typed back: hash clean', location.hash === '', location.hash);

// the frontier table: one row per cluster size, best fitting layout per policy, from the sheet's formulas
const fr = [...document.querySelectorAll('.frontier tbody tr')];
T.check('frontier table has rows', fr.length >= 8, fr.length);
const row432 = fr.find((tr) => tr.cells[0].textContent === '432');
T.check('432 GPUs: policy A PP 6 × EP 72 × DP 1 · V-Min (the default), policy B V-ZB; EP pegged to the domain', row432 && row432.cells[2].textContent === 'PP 6 × EP 72 × DP 1 · V-Min' && /EP 72 × DP 1 · V-ZB/.test(row432.cells[5].textContent), row432?.textContent);
T.check('every frontier layout uses EP 72', [...document.querySelectorAll('.frontier td.lay')].every((td) => /EP 72/.test(td.textContent)), '');
const row288 = fr.find((tr) => tr.cells[0].textContent === '288');
T.check('288 GPUs: neither fits without offload at the 10 % reserve', /does not fit/.test(row288.cells[2].textContent) && /does not fit/.test(row288.cells[3].textContent), row288.textContent);
const rhoB = (tr) => parseFloat(tr.cells[tr.cells.length - 1].textContent.replace(/,/g, ''));
const fitsB = fr.filter((tr) => !/does not fit/.test(tr.cells[tr.cells.length - 1].textContent));
const zb = fitsB.findIndex((tr) => /V-ZB/.test(tr.cells[tr.cells.length - 3].textContent));
T.check('policy B: once V-ZB fits the rate is flat at the ceiling and above every smaller cluster\'s', zb >= 0 && fitsB.slice(zb).every((tr) => rhoB(tr) === rhoB(fitsB[zb])) && fitsB.slice(0, zb).every((tr) => rhoB(tr) < rhoB(fitsB[zb])), fitsB.map(rhoB).join(' '));
const mins = [...document.querySelectorAll('.mins tbody tr')];
T.check('min-GPUs table: four gradient × optimizer combinations × offload no/yes, each with a fit', mins.length === 8 && mins.every((tr) => /PP \d+ × EP \d+/.test(tr.cells[3].textContent) && /PP \d+ × EP \d+/.test(tr.cells[5].textContent)), mins.map((tr) => tr.textContent).join(' | '));
T.check('… FP32/FP32, no offload: 432 · PP 6 × EP 72 · V-Min for policy A (the default), 432 for B', mins[6].cells[3].textContent === '432 · PP 6 × EP 72 × DP 1 · V-Min' && /^432 ·/.test(mins[6].cells[5].textContent), `${mins[6].cells[3].textContent} / ${mins[6].cells[5].textContent}`);
T.check('… with offload: 288 for both, flagged ‡ (prefetch spans microbatches)', /^288 ·/.test(mins[7].cells[3].textContent) && /‡/.test(mins[7].cells[4].textContent) && /^288 ·/.test(mins[7].cells[5].textContent), mins[7].textContent);
T.check('… BF16 grads + states, no offload: 288', /^288 ·/.test(mins[0].cells[3].textContent), mins[0].cells[3].textContent);
const mgs = [...document.querySelectorAll('.margins tbody tr')];
T.check('margin table: 0 % to 30 %, the minimum grows with the reserve', mgs.length === 7 && /^432 ·/.test(mgs[0].cells[1].textContent) && /^216 ·/.test(mgs[0].cells[4].textContent) && /^864 ·/.test(mgs[6].cells[1].textContent), mgs.map((tr) => tr.textContent).join(' | '));
// the frontier follows the knobs: with BF16 gradients and states policy A fits at 288 GPUs
type('oB', '6'); type('gB', '2');
const r288 = [...document.querySelectorAll('.frontier tbody tr')].find((tr) => tr.cells[0].textContent === '288');
T.check('g_B 2, o_B 6: the frontier re-builds, policy A now fits at 288', /PP \d+ × EP \d+/.test(r288.cells[2].textContent), r288.textContent);
type('oB', '8'); type('gB', '4');
T.done();
