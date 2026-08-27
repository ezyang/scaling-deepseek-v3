// @page index.html
T.check('posts list present', document.querySelectorAll('.posts li').length >= 1, document.querySelectorAll('.posts li').length);
T.check('no stray widget tags', !document.querySelector('dsv3-controls, dsv3-memory, dsv3-stack'), '');
T.done();
