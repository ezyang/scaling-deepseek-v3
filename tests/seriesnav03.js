// @page studies/03-roofline.html
// series interlinking: the strip under the h1 and the prev/next cards
await T.tick(200);
const strip = document.querySelector('.series-strip');
T.check('series strip present with the ordinal', strip?.textContent.includes('post 3 of 3')
  && strip.querySelector('a[href="../index.html"]'), strip?.textContent);
const nav = document.querySelector('.series-nav');
T.check('prev card links to 02', nav?.querySelector('a.card.prev')?.getAttribute('href') === './02-hopper-memory.html', '');
T.check('up card links to the front page', nav?.querySelector('a.card.up')?.getAttribute('href') === '../index.html', '');
T.check('no next card past the last post', !nav?.querySelector('a.card.next'), '');
// titles stay synchronized: manifest ≡ <title> ≡ <h1>
const { SERIES } = await import('/studies/series.js');
const me = SERIES.find(p => location.pathname.endsWith('/' + p.href));
T.check('manifest title ≡ page title ≡ h1', me.title === document.title
  && me.title === document.querySelector('h1').textContent, `${me.title} vs ${document.title}`);
const raw = await (await fetch(location.pathname)).text();
T.check('strip ships as static markup (no load-time page shove)', /<main>\s*<nav class="series-strip">/.test(raw), '');
T.done();
