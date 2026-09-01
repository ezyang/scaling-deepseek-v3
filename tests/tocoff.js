// @page studies/01-deepseek-diagram.html
// the rail's gate: post 01 imports toc.js via series.js but has one section —
// the LIVE page's DOM must be untouched
await T.tick(200);
T.check('no section rail on the short (LIVE) post', !document.querySelector('nav.toc'), '');
T.done();
