// @page studies/03-roofline.html
// @args --width 430
// 03 mobile: the static formula sheets (.cellsheet: fixed column grids) and
// trace tallies don't preview — their boxes stay in the column and the
// tables scroll sideways inside at desktop width, live (editable values, the
// SM toggle). The widgets preview, each with an explore pill floated over
// its bottom-right corner that takes no space.
await T.tick(300);
T.check('no horizontal page scroll', document.documentElement.scrollWidth <= innerWidth + 1,
  `${document.documentElement.scrollWidth} vs ${innerWidth}`);
const sheets = [...document.querySelectorAll('main .cellsheet')];
const tallies = [...document.querySelectorAll('main .tally')];
T.check('sheets + tallies stay live (no preview)', [...sheets, ...tallies].every((s) => !s.classList.contains('mprev'))
  && tallies.length === 2, '');
const open = sheets.filter((s) => s.offsetWidth);   // the folded ones measure later
T.check('sheet boxes fit the column; their tables scroll sideways at desktop width (734px)',
  open.every((s) => s.offsetWidth <= 392 && s.scrollWidth > s.clientWidth && s.querySelector('table').offsetWidth === 734),
  [...new Set(open.map((s) => `${s.offsetWidth}/${s.querySelector('table').offsetWidth}`))].join(','));
T.check('tallies scroll sideways, no cell wraps',
  tallies.every((t) => t.offsetWidth <= 392 && t.scrollWidth > t.clientWidth
    && [...t.querySelectorAll('tbody td')].every((d) => d.offsetHeight < 24)), '');
const fold = document.querySelector('details.fold');
fold.open = true;
await T.tick();
const fs = fold.querySelector('.cellsheet');
T.check('a folded sheet opens into the same scroller', fs.offsetWidth <= 392 && fs.querySelector('table').offsetWidth === 734, fs.offsetWidth);
fold.open = false;
T.click('.tally.kern [data-sm="112"]');
await T.tick();
T.check('tally SM toggle works in place', T.el('.tally.kern [data-sm="112"]').getAttribute('aria-pressed') === 'true', '');
T.click('.tally.kern [data-sm="132"]');
await T.tick();

const W = 'dsv3-mmfig, dsv3-ladder, dsv3-epsim, dsv3-mesh, dsv3-fsdpsched, dsv3-fsdpcurve, dsv3-ppcurve';
T.check("03's own widgets preview", [...document.querySelectorAll(W)].every((w) => w.classList.contains('mprev')),
  [...document.querySelectorAll(W)].filter((w) => !w.classList.contains('mprev')).map((w) => w.tagName).join(','));
// the pill sits inside the preview's visual box, bottom-right, and the
// prose after it starts where it would without a pill
const bad = [];
for (const el of document.querySelectorAll('.mprev')) {
  const pill = el.nextElementSibling, pr = pill.getBoundingClientRect(), er = el.getBoundingClientRect();
  const next = pill.nextElementSibling?.getBoundingClientRect();
  if (!(pr.bottom <= er.bottom - 4 && pr.top >= er.top && pr.right <= er.right && pr.right >= er.right - 12)
    || (next && Math.abs(next.top - parseFloat(getComputedStyle(pill.nextElementSibling).marginTop) - (er.bottom + 26)) > 2))
    bad.push(`${el.tagName}#${el.id} pill ${Math.round(pr.top - er.bottom)}..${Math.round(pr.bottom - er.bottom)} next ${next ? Math.round(next.top - er.bottom) : '-'}`);
}
T.check('pills float in the bottom-right corner, zero layout height', !bad.length, bad.slice(0, 3).join('; '));
T.done();
