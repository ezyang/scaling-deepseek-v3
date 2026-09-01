// @page index.html
await T.tick(200);
const links = [...document.querySelectorAll('ul.posts a')].map(a => a.getAttribute('href'));
T.check('front page lists both posts', links.includes('./studies/01-deepseek-diagram.html')
  && links.includes('./studies/02-hopper-memory.html'), links.join(' '));
const t2 = [...document.querySelectorAll('ul.posts a')].map(a => a.textContent.trim());
T.check('front-page link text matches the page titles', t2.includes('Memory: a Hopper case study')
  && t2.includes('An infra-oriented diagram of the DeepSeek-V3 architecture'), t2.join(' | '));
T.done();
