// @page studies/01-deepseek-diagram.html
// the LIVE first post gains the next-link to 02 (and only nav — no content change)
await T.tick(200);
T.check('post 1 of 2 strip', document.querySelector('.series-strip')?.textContent.includes('post 1 of 2'), '');
const nav = document.querySelector('.series-nav');
T.check('next card links to 02', nav?.querySelector('a.card.next')?.getAttribute('href') === './02-hopper-memory.html', '');
T.check('no prev card on the first post', !nav?.querySelector('a.card.prev'), '');
const { SERIES } = await import('/studies/series.js');
const me = SERIES.find(p => location.pathname.endsWith('/' + p.href));
T.check('manifest title ≡ page title ≡ h1', me.title === document.title
  && me.title === document.querySelector('h1').textContent, `${me.title} vs ${document.title}`);
T.done();
