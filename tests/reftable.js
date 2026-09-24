// @page studies/03-roofline.html
// the reference-scale ladder (#reftable): every printed number rounds from its
// data-raw; hover shows the raw value; a multi-cell copy yields raw-number TSV
const t = document.getElementById('reftable');
await T.tick(100);
const SCALE = { 'µs': 1e-6, s: 1, hours: 3600, days: 86400, 'TFLOP/s': 1e12, 'EFLOP/s': 1e18 };
// printed "0.45" agrees with v if v is within half a unit of its last digit
const agrees = (txt, v) => { const d = (txt.split('.')[1] ?? '').length; return Math.abs(v - +txt.replace(/,/g, '')) <= 0.5 * 10 ** -d * (1 + 1e-9); };
const bad = [];
let n = 0;
for (const c of t.querySelectorAll('td.t, td.m')) {
  const raw = +c.dataset.raw;
  if (c.classList.contains('m')) {
    const exp = +c.nextElementSibling.querySelector('sup').textContent;
    n++; if (!agrees(c.textContent, raw / 10 ** exp)) bad.push(`${c.textContent}e${exp} vs ${raw}`);
  } else {
    const [num, u] = c.textContent.split(' ');
    n++; if (!agrees(num, raw / SCALE[u])) bad.push(`${c.textContent} vs ${raw}`);
  }
}
T.check('every printed number carries a raw value', n === 9 * 3 + 3 * 2 && [...t.querySelectorAll('td.t, td.m')].every((c) => c.dataset.raw), n);
T.check('every printed number rounds from its raw value', bad.length === 0, bad.join('; '));
T.check('DeepSeek sustained = 2.664M GPU-hours for 14.8T tokens (≈339 TFLOP/s)', /^339\d{12}/.test(t.querySelector('tr[data-r="h800"] td:last-child').dataset.raw.replace('.', '')), t.querySelector('tr[data-r="h800"] td:last-child').dataset.raw);
T.check('step FLOPs exact: 6N × 62,914,560', t.querySelector('tr.hero td.m').dataset.raw === String(6n * 36625618432n * 62914560n), t.querySelector('tr.hero td.m').dataset.raw);
// hover
const cell = t.querySelector('tr.hero td.t');
const r = cell.getBoundingClientRect();
cell.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: r.left + 5, clientY: r.top + 5 }));
const tip = document.querySelector('.ref-tip');
T.check('hover: tooltip shows unrounded seconds', tip.style.display === 'block' && /^3\.41\d* s/.test(tip.textContent), tip.textContent);
t.dispatchEvent(new MouseEvent('mouseleave'));
T.check('mouseleave hides it', tip.style.display === 'none', '');
// click pins: survives hovering another cell and clicks inside the tip; any other click closes
const tick = (el, type) => { const q = el.getBoundingClientRect(); el.dispatchEvent(new MouseEvent(type, { bubbles: true, clientX: q.left + 5, clientY: q.top + 5 })); };
getSelection().removeAllRanges();
tick(cell, 'click');
T.check('click pins the tip', tip.classList.contains('pinned') && tip.style.display === 'block' && getComputedStyle(tip).pointerEvents === 'auto', tip.className);
const other = t.querySelector('tr[data-f="token"] td.m');
tick(other, 'mousemove'); t.dispatchEvent(new MouseEvent('mouseleave'));
T.check('pinned: hover elsewhere and mouseleave leave it', /^3\.41/.test(tip.textContent) && tip.style.display === 'block', tip.textContent);
tick(tip, 'click');
T.check('pinned: a click inside the tip keeps it (so its text can be selected)', tip.classList.contains('pinned'), '');
tick(other, 'click');
T.check('any other click unpins and closes', !tip.classList.contains('pinned') && tip.style.display === 'none', tip.className);
tick(other, 'click');
T.check('…and the next click pins afresh', tip.classList.contains('pinned') && /219,753,710,592 FLOPs/.test(tip.textContent), tip.textContent);
tick(document.querySelector('h1'), 'click');
T.check('click on the page closes it', tip.style.display === 'none', '');
// copy: header through the first band's first row → TSV of raw numbers
const sel = getSelection(), range = document.createRange();
range.setStart(t.rows[0].cells[1], 0);
range.setEnd(t.querySelector('tr[data-f="token"]').cells[4], 1);
sel.removeAllRanges(); sel.addRange(range);
const dt = new DataTransfer();
document.dispatchEvent(new ClipboardEvent('copy', { clipboardData: dt, bubbles: true, cancelable: true }));
const rows = dt.getData('text/plain').trimEnd().split('\n').map((l) => l.split('\t'));
T.check('copy: 3 rows × 4 columns', rows.length === 3 && rows.every((r) => r.length === 4), JSON.stringify(rows));
T.check('copy: header names the time unit', rows[0][2] === 'at FP8 peak (s)' && rows[0][3] === 'sustained (s)', rows[0]);
T.check('copy: band row carries FLOP/s', /H800 \(FLOP\/s\)/.test(rows[1][0]) && +rows[1][2] === 1979e12, rows[1]);
T.check('copy: token row is raw numbers', rows[2][1] === String(6n * 36625618432n) && Math.abs(+rows[2][2] - 111.04e-6) < 1e-7, rows[2]);
// a selection inside one cell copies as-is
range.setStart(cell.firstChild, 0); range.setEnd(cell.firstChild, 3);
sel.removeAllRanges(); sel.addRange(range);
const dt2 = new DataTransfer();
document.dispatchEvent(new ClipboardEvent('copy', { clipboardData: dt2, bubbles: true, cancelable: true }));
T.check('single-cell copy untouched', dt2.getData('text/plain') === '', dt2.getData('text/plain'));
T.done();
