// @page studies/02-hopper-memory.html
// hollows: pooled remainder cells — slices stay inside a 5x4 outer footprint
const dia = document.getElementById('local-diagram').querySelector('.lv-scroll svg');
const hollows = [...dia.querySelectorAll('rect[fill="none"][stroke-width="0.8"]')];
T.log('hollow slice count', hollows.length);
T.check('hollow slices exist', hollows.length > 0, hollows.length);
const ok = hollows.every(r => +r.getAttribute('height') === 3.2 && +r.getAttribute('width') <= 4.21);
T.check('slices inset (h 3.2, w <= 4.2)', ok, '');
// the unit swatch floats right on the chart header
const bar = document.getElementById('local-diagram').querySelector('.lv-bar');
T.check('chart header grey unit swatch', !!bar.querySelector('svg rect[fill="#898781"]'), '');
T.done();
