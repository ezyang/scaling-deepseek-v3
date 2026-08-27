// @page studies/02-hopper-memory.html
const t = document.querySelector('#inventory table.xcheck');
T.check('inventory table renders', !!t && t.rows.length === 4, t?.rows.length);
T.check('no stray widget tags', !document.querySelector('dsv3-controls, dsv3-memory, dsv3-stack'), '');
T.done();
