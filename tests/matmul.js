// @page studies/03-roofline.html
// prep diagram hover, across the diagram and its y = xW expansion (e-prefixed):
// a weight lights the x it reads, the y it adds into, and the rest of that sum;
// hitboxes tile the gaps between neighbors (no hover dead zone, no flicker)
const fig = document.getElementById('mmfig');
const tag = (e) => (e.closest('.mmeq') ? 'e' : '') + (e.classList.contains('w') ? `w${e.dataset.i}${e.dataset.j}`
  : e.tagName === 'path' ? `a${e.dataset.j}` : e.dataset.i !== undefined ? `x${e.dataset.i}` : `y${e.dataset.j}`);
const on = (c) => [...fig.querySelectorAll('.' + c)].map(tag).sort().join(' ');
const ov = (el) => el.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
const cell = (sel) => ov(T.el(`svg ${sel}`).querySelector('rect'));
cell('.w[data-i="2"][data-j="4"]');
T.check('weight (2,4): it and its term', on('src') === 'ew24 w24', on('src'));
T.check('weight (2,4): reads x2, adds into y4 via arrow 4', on('dep') === 'a4 ey4 x2 y4', on('dep'));
T.check('weight (2,4): rest of y4\'s sum', on('sib') === 'ew04 ew14 ew34 ew44 w04 w14 w34 w44', on('sib'));
cell('.v[data-i="1"]');
T.check('x1: feeds row 1', on('dep') === 'ew10 ew11 ew12 ew13 ew14 w10 w11 w12 w13 w14', on('dep'));
T.check('x1: reaches every y', on('sib') === 'ey0 ey1 ey2 ey3 ey4 y0 y1 y2 y3 y4', on('sib'));
cell('.v[data-j="3"]');
T.check('y3: sums column 3', on('dep') === 'a3 ew03 ew13 ew23 ew33 ew43 w03 w13 w23 w33 w43', on('dep'));
T.check('y3: reads every x', on('sib') === 'x0 x1 x2 x3 x4', on('sib'));
ov(T.el('.mmeq .w[data-i="1"][data-j="2"]'));
T.check('a term lights its cell', on('src') === 'ew12 w12', on('src'));
T.check('term (1,2): reads x1, adds into y2', on('dep') === 'a2 ey2 x1 y2', on('dep'));
fig.dispatchEvent(new MouseEvent('mouseleave'));
T.check('leave clears', on('src') + on('dep') + on('sib') === '', on('dep'));
fig.scrollIntoView({ block: 'start' });   // elementFromPoint only sees the viewport
const tgt = (x, y) => document.elementFromPoint(x, y)?.closest('.w, .v');
const probe = (a, b, name) => {
  const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
  const horiz = Math.abs(ra.top - rb.top) < 2;
  let miss = [];
  if (horiz) { const y = ra.top + ra.height / 2; for (let x = ra.left + 1; x < rb.right - 1; x += 0.5) if (!tgt(x, y)) miss.push(x.toFixed(1)); }
  else { const x = ra.left + ra.width / 2; for (let y = ra.top + 1; y < rb.bottom - 1; y += 0.5) if (!tgt(x, y)) miss.push(y.toFixed(1)); }
  T.check(`no dead zone: ${name}`, miss.length === 0, miss.slice(0, 6).join(','));
};
const q = (s) => T.el(s);
probe(q('svg .w[data-i="1"][data-j="1"] rect:not(.hit)'), q('svg .w[data-i="1"][data-j="2"] rect:not(.hit)'), 'W cells side by side');
probe(q('svg .w[data-i="1"][data-j="1"] rect:not(.hit)'), q('svg .w[data-i="2"][data-j="1"] rect:not(.hit)'), 'W cells stacked');
probe(q('svg .v[data-i="1"] rect:not(.hit)'), q('svg .v[data-i="2"] rect:not(.hit)'), 'x cells');
probe(q('.mmeq .w[data-i="1"][data-j="2"]'), q('.mmeq .w[data-i="2"][data-j="2"]'), 'terms across a +');
probe(q('.mmeq .v[data-j="2"]'), q('.mmeq .w[data-i="0"][data-j="2"]'), 'y chip across =');
probe(q('.mmeq .w[data-i="1"][data-j="1"]'), q('.mmeq .w[data-i="1"][data-j="2"]'), 'terms across lines');
T.done();
