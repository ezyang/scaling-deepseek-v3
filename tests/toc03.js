// @page studies/03-roofline.html
// @args --width 1366
// section rail on 03: h3 subsections join the rail as indented .sub items
// (in document order, under their h2), and the scrollspy lights them
await T.tick(200);
const nav = document.querySelector('nav.toc');
const want = [...document.querySelectorAll('main > h2, main > h3')].map(h => (h.tagName === 'H3' ? '>' : '') + h.textContent);
const got = [...nav.children].map(a => (a.classList.contains('sub') ? '>' : '') + a.textContent);
T.check('rail = h2s + h3s in document order, h3s marked sub', got.join('|') === want.join('|'), got.join('|'));
T.check('03 has subsections on the rail', nav.querySelectorAll('a.sub').length >= 2, '');
const h = [...document.querySelectorAll('main > h3')][1];
h.scrollIntoView(); await T.tick(150);
document.dispatchEvent(new Event('scroll')); await T.tick(50);
T.check('scrollspy lights the subsection under the reading line', nav.querySelector('a.on')?.textContent === h.textContent, nav.querySelector('a.on')?.textContent);
T.done();
