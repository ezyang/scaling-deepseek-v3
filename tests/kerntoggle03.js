// @page studies/03-roofline.html
// 03's kernel table: the SM toggle sets n_c (sheetedit knob, in the hash); the
// SOL cells reprice by 132/112, the ratios follow from the exact trace times,
// both turn amber, the columns don't shift, and the first button undoes it all
await T.tick(100);
const kern = document.querySelector('.tally.kern');
const [b132, b112] = kern.querySelectorAll('.smtog button');
const total = [...kern.querySelectorAll('tr.hl td')].map((td) => td.textContent.replace(/\s+/g, ' ').trim());
const cols = () => [...kern.querySelectorAll('thead tr:last-child th')].map((th) => Math.round(th.getBoundingClientRect().left));
const pub = total.slice(1), x0 = cols();
T.check('published: 132 pressed, totals at all SMs', b132.getAttribute('aria-pressed') === 'true' && pub.join('|') === '4,210 µs|9,267|2.20|5,970 µs|14,922|2.50|3,300 µs|7,593|2.30', pub.join('|'));
b112.click();
await T.tick(50);
const now = [...kern.querySelectorAll('tr.hl td')].slice(1).map((td) => td.textContent.replace(/\s+/g, ' ').trim());
T.check('112: SOL × 132/112, ratios ÷ 132/112', now.join('|') === '4,960 µs|9,267|1.87|7,030 µs|14,922|2.12|3,890 µs|7,593|1.95', now.join('|'));
T.check('112: button state, hash, amber', b112.getAttribute('aria-pressed') === 'true' && b132.getAttribute('aria-pressed') === 'false'
  && /SMc/.test(decodeURIComponent(location.hash)) && kern.querySelectorAll('td.off').length === 37 && kern.querySelectorAll('[data-cellref].off').length === 37, location.hash);
T.check('112: the sheet row reads 112 and turns amber', [...document.querySelectorAll('.cellsheet tr')].find((tr) => tr.cells[0]?.textContent === 'nc')?.classList.contains('off'), '');
T.check('the toggle shifts no column', cols().join() === x0.join(), `${x0} → ${cols()}`);
b132.click();
await T.tick(50);
T.check('132 again: back to published, hash clean, nothing amber', location.hash === '' && !kern.querySelector('.off')
  && [...kern.querySelectorAll('tr.hl td')].slice(1).map((td) => td.textContent.replace(/\s+/g, ' ').trim()).join('|') === pub.join('|'), location.hash);
T.done();
