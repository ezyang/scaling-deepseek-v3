// @page studies/scratch-bwd.html
// the backward-unrolled precision diagram (scratch): toggles, unique names, wire-hover families, + junctions
const w = document.querySelector('dsv3-bwd');
const names = () => [...w.querySelectorAll('svg text.name')].map(t => t.textContent);
const dups = names().filter((n, i, a) => a.indexOf(n) !== i);
T.check('every box uniquely named', dups.length === 0, dups.join(' | '));
T.check('no identity boxes', !names().some(n => /identity/.test(n)), '');
const w0 = +w.querySelector('svg').getAttribute('width');
T.check('wgrad column present by default', names().some(n => /wgrad/.test(n)), '');
w.querySelector('.bw-head input').click(); await T.tick(50);
T.check('toggle hides wgrad boxes and shrinks', !names().some(n => /wgrad|dγ/.test(n)) && +w.querySelector('svg').getAttribute('width') < w0, names().filter(n => /wgrad/.test(n)).length);
T.check('no amber edges land in the hidden column', ![...w.querySelectorAll('svg path.save')].some(p => +p.getAttribute('d').split('H').pop().split(' ')[1] > +w.querySelector('svg').getAttribute('width')), '');
w.querySelector('.bw-head input').click(); await T.tick(50);
T.check('toggle back restores', names().some(n => /wgrad/.test(n)), '');
const sh = () => w.querySelector('.bw-head input[data-k="sh"]');
sh().click(); await T.tick(50);
T.check('shared expert hidden: no shared boxes, no shared rail', !names().some(n => /shared/.test(n) && !/routed \+ shared/.test(n))
  && ![...w.querySelectorAll('svg text.raillab')].some(t => /shared/.test(t.textContent)), names().filter(n => /shared/.test(n)).join('|'));
T.check('combine flows straight into the routed+shared add', !!names().find(n => n === '+ routed + shared'), '');
sh().click(); await T.tick(50);
T.check('shared expert restored', names().some(n => /shared gate\/up$/.test(n)), '');

const svg = w.querySelector('svg');
const hits = [...svg.querySelectorAll('path.hit')];
T.check('hit paths exist', hits.length > 100, hits.length);
const ev = (el, t) => el.dispatchEvent(new MouseEvent(t, { bubbles: true }));
// hover a save edge (amber): it and the forward wire it forks from light up
const amber = svg.querySelectorAll('path.save')[1];   // [0] leaves the x0 chip row, which has no source box
const start = amber.getAttribute('d').split(' ').slice(0, 3).join(' ');
const hA = hits.find(h => h.getAttribute('d').startsWith(start));
ev(hA, 'mouseenter'); await T.tick(20);
const lit = [...svg.querySelectorAll('.hl')];
T.check('hover lights a component of >1 wire', lit.length > 1, lit.length);
T.check('the source box lights with it', lit.some(e => e.tagName === 'rect' && e.classList.contains('box')), '');
T.check('the fork dot lights with it', lit.some(e => e.tagName === 'circle' && e.getAttribute('r') === '2.5'), '');
T.check('lit set = the amber edge + the forward wire it forks from, nothing else', lit.some(e => e.classList.contains('save')) && lit.some(e => e.classList.contains('wire')) && !lit.some(e => /grad|savew/.test(e.getAttribute('class'))), lit.map(e => e.getAttribute('class')).join(','));
ev(hA, 'mouseleave'); await T.tick(20);
T.check('unhover clears', svg.querySelectorAll('.hl').length === 0, '');
// a dY edge's component must not include a blue/amber edge running along the lane above it
for (const h of hits) { h.dispatchEvent(new MouseEvent('mouseenter', { bubbles: true })); const l = [...svg.querySelectorAll('.hl')];
  const bad = l.some(e => e.classList.contains('grad')) && l.some(e => /save/.test(e.getAttribute('class')));
  h.dispatchEvent(new MouseEvent('mouseleave', { bubbles: true }));
  if (bad) { T.check('no component mixes a dY edge with a save edge', false, h.getAttribute('d')); break; } }
T.check('component scan done', true, '');
// converse: hovering a producer box lights its outgoing wire family
const box = [...svg.querySelectorAll('g.fop')].find(g => /q\/kv down-proj/.test(g.querySelector('text.name').textContent));
box.dispatchEvent(new MouseEvent('mouseenter', { bubbles: true })); await T.tick(20);
const litB = [...svg.querySelectorAll('.hl')];
T.check('box hover lights its output wire(s) and itself', litB.some(e => e.tagName === 'path') && litB.includes(box.querySelector('rect.box')), litB.length);
box.dispatchEvent(new MouseEvent('mouseleave', { bubbles: true })); await T.tick(20);
T.check('box unhover clears', svg.querySelectorAll('.hl').length === 0, '');

// click pins; background click clears
const hit0 = svg.querySelectorAll('path.hit')[3];
hit0.dispatchEvent(new MouseEvent('click', { bubbles: true })); await T.tick(20);
T.check('click pins the family', svg.querySelectorAll('.pin').length > 0, svg.querySelectorAll('.pin').length);
svg.dispatchEvent(new MouseEvent('click', { bubbles: true })); await T.tick(20);
T.check('background click clears pins', svg.querySelectorAll('.pin').length === 0, '');
// geometric grammar: a wire ending at a + rim carries an arrowhead; no wire ends under a + center
const plusC = [...svg.querySelectorAll('text.plus')].map(t => [+t.getAttribute('x'), +t.getAttribute('y') - 3.5]);
const endOf = (d) => { const t = d.trim().split(/\s+/); let cmd = 'M', p = [0, 0];
  for (let i = 0; i < t.length;) { if (/^[MHVL]$/.test(t[i])) { cmd = t[i++]; continue; }
    if (cmd === 'H') p = [+t[i++], p[1]]; else if (cmd === 'V') p = [p[0], +t[i++]]; else { p = [+t[i], +t[i + 1]]; i += 2; } }
  return p; };
const uniq = plusC.filter(([x, y], i) => plusC.findIndex(([a, b]) => a === x && b === y) === i);
const ends = [...svg.querySelectorAll('path.wire, path.save, path.savew, path.grad, path.elide')].map(p => ({ p, end: endOf(p.getAttribute('d')) }));
const atRim = ends.filter(({ end }) => plusC.some(([x, y]) => Math.abs(Math.hypot(end[0] - x, end[1] - y) - 8) < 1.5));
const under = ends.filter(({ end }) => plusC.some(([x, y]) => Math.hypot(end[0] - x, end[1] - y) < 6));
T.check('every + has ≥2 arriving wires, all with arrowheads', atRim.length >= 2 * uniq.length && atRim.every(({ p }) => p.hasAttribute('marker-end')), `${atRim.length} arrivals / ${uniq.length} junctions`);
T.check('no wire ends hidden under a +', under.length === 0, under.length);
// grammar: a wire never crosses a + circle (its arrivals stop at the rim), and every arrowhead lands on a
// box edge, a + rim, or the SIDE of another wire — never collinear in the middle of one
const parsePts = (d) => { const t = d.trim().split(/\s+/); let cmd = 'M'; const pts = [];
  for (let i = 0; i < t.length;) { if (/^[MHVL]$/.test(t[i])) { cmd = t[i++]; continue; } const [px, py] = pts[pts.length - 1] ?? [0, 0];
    if (cmd === 'H') pts.push([+t[i++], py]); else if (cmd === 'V') pts.push([px, +t[i++]]); else { pts.push([+t[i], +t[i + 1]]); i += 2; } }
  return pts; };
const dSeg = (p, a, b) => { const dx = b[0] - a[0], dy = b[1] - a[1], l2 = dx * dx + dy * dy || 1;
  const u = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2)); return Math.hypot(p[0] - (a[0] + u * dx), p[1] - (a[1] + u * dy)); };
const wires = [...svg.querySelectorAll('path.wire, path.save, path.savew, path.grad, path.elide')].map(p => ({ p, pts: parsePts(p.getAttribute('d')) }));
const rects = [...svg.querySelectorAll('rect.box')].map(r => ({ x: +r.getAttribute('x'), y: +r.getAttribute('y'), w: +r.getAttribute('width'), h: +r.getAttribute('height') }));
const crossers = [];
for (const [qx, qy] of uniq) for (const w of wires) {
  const endsAtRim = [w.pts[0], w.pts[w.pts.length - 1]].some(e => Math.abs(Math.hypot(e[0] - qx, e[1] - qy) - 8) < 1.5);
  if (endsAtRim) continue;
  for (let i = 1; i < w.pts.length; i++) if (dSeg([qx, qy], w.pts[i - 1], w.pts[i]) < 7.5) { crossers.push(w.p.getAttribute('d')); break; }
}
T.check('no wire crosses a + circle', crossers.length === 0, crossers.slice(0, 2).join(' | '));
const onEdge = (e) => rects.some(({ x, y, w, h }) => (e[0] >= x - 2 && e[0] <= x + w + 2 && (Math.abs(e[1] - y) <= 3 || Math.abs(e[1] - (y + h)) <= 3))
  || (e[1] >= y - 2 && e[1] <= y + h + 2 && (Math.abs(e[0] - x) <= 3 || Math.abs(e[0] - (x + w)) <= 3)));
const bad = [];
for (const w of wires) {
  if (!w.p.hasAttribute('marker-end') || w.pts.length < 2) continue;
  const e = w.pts[w.pts.length - 1], a = w.pts[w.pts.length - 2], L = Math.hypot(e[0] - a[0], e[1] - a[1]) || 1, dir = [(e[0] - a[0]) / L, (e[1] - a[1]) / L];
  if (onEdge(e) || uniq.some(([qx, qy]) => Math.abs(Math.hypot(e[0] - qx, e[1] - qy) - 8) < 1.5)) continue;
  let landed = false, collinear = false;
  for (const o of wires) { if (o === w) continue;
    for (let i = 1; i < o.pts.length; i++) if (dSeg(e, o.pts[i - 1], o.pts[i]) <= 5) { landed = true;
      const ox = o.pts[i][0] - o.pts[i - 1][0], oy = o.pts[i][1] - o.pts[i - 1][1], ol = Math.hypot(ox, oy) || 1;
      if (Math.abs(dir[0] * oy / ol - dir[1] * ox / ol) < 0.1) collinear = true; } }
  if (!landed || collinear) bad.push((collinear ? 'mid-wire: ' : 'into nothing: ') + w.p.getAttribute('d'));
}
T.check('every arrowhead lands on a box edge, a + rim, or the side of a wire', bad.length === 0, bad.slice(0, 3).join(' | '));
T.done();
