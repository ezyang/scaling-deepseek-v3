// @page studies/03-roofline.html
// @args --width 430
// 03 mobile: the static formula sheets (.cellsheet: fixed column grids) and
// trace tallies crush below their natural width, so they preview like the
// widgets — including the sheets folded inside <details>, measured open.
await T.tick(300);
T.check('no horizontal page scroll', document.documentElement.scrollWidth <= innerWidth + 1,
  `${document.documentElement.scrollWidth} vs ${innerWidth}`);
const sheets = [...document.querySelectorAll('main .cellsheet')];
T.check('every formula sheet previews', sheets.every((s) => s.classList.contains('mprev')),
  sheets.filter((s) => !s.classList.contains('mprev')).length + ' live');
T.check('sheets keep their desktop 760px box (row wrapping identical to desktop)',
  sheets.every((x) => x.offsetWidth === 760), [...new Set(sheets.map((x) => x.offsetWidth))].join(','));
const tallies = [...document.querySelectorAll('main .tally')];
T.check('both trace tallies preview', tallies.length === 2 && tallies.every((t) => t.classList.contains('mprev')), tallies.length);
T.check('no tally cell wraps (measured at max-content, not a rounded px short)',
  tallies.every((t) => [...t.querySelectorAll('tbody td')].every((d) => d.offsetHeight < 24)), '');
const W = 'dsv3-mmfig, dsv3-ladder, dsv3-epsim, dsv3-mesh, dsv3-fsdpsched, dsv3-fsdpcurve, dsv3-ppcurve';
T.check("03's own widgets preview", [...document.querySelectorAll(W)].every((w) => w.classList.contains('mprev')),
  [...document.querySelectorAll(W)].filter((w) => !w.classList.contains('mprev')).map((w) => w.tagName).join(','));

// a folded sheet was probed open, then folded back; opening it shows a
// correctly scaled preview, not a zero-width one
const fold = document.querySelector('details.fold');
const fs = fold.querySelector('.cellsheet');
T.check('fold stays folded after the probe', !fold.open, '');
fold.open = true;
await T.tick();
const r = fs.getBoundingClientRect();
T.check('opened fold: preview spans the column', fs.classList.contains('mprev') && r.width > 380 && r.width < 392
  && fs.nextElementSibling?.classList.contains('mopen'), `w=${Math.round(r.width)}`);
fs.dispatchEvent(new MouseEvent('click', { bubbles: true }));
await T.tick();
T.check('focused fold hides its explore button', document.body.classList.contains('mfocus')
  && getComputedStyle(fs.nextElementSibling).display === 'none', '');
T.click('.mclose');
await T.tick();
fold.open = false;

// focus a sheet: its editable value cells take clicks again
const s = sheets.find((x) => x.querySelector('td.vl.edv'));
s.dispatchEvent(new MouseEvent('click', { bubbles: true }));
await T.tick();
const td = s.querySelector('td.vl.edv');
T.check('tap focuses the sheet at natural width', document.body.classList.contains('mfocus') && s.style.transform === ''
  && getComputedStyle(td).pointerEvents !== 'none', '');
T.click('.mclose');
await T.tick();
T.check('closed clean', !document.body.classList.contains('mfocus') && s.classList.contains('mprev'), '');

// focus the kernel tally: the SM toggle works there
const kern = document.querySelector('.tally.kern');
kern.dispatchEvent(new MouseEvent('click', { bubbles: true }));
await T.tick();
T.click('.tally.kern [data-sm="112"]');
await T.tick();
T.check('focused tally: SM toggle presses', kern.querySelector('[data-sm="112"]').getAttribute('aria-pressed') === 'true', '');
T.click('.tally.kern [data-sm="132"]');
T.click('.mclose');
await T.tick();
T.done();
