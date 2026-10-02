// @page studies/03-roofline.html
// the ZeRO-3 timeline: two DSv3 MoE layers, a BF16 move (399 MB of IB per
// GPU, 7.99 ms) per all-gather and two (FP32) per reduce-scatter; compute is
// conserved across microbatch counts while the collectives multiply (3 per
// layer per microbatch, 4 moves' worth).
// Compute is scheduled with gathers prefetched a layer ahead, then each
// gather is drawn as late as that allows (it ends where its consumer starts
// or the next collective begins): at one microbatch only the first gather
// and the last reduce-scatter are exposed; at 8 the IB queue can't keep up.
// Every collective carries its layer digit. Every compute op starts after
// its own gather, neither track double-books, flips tween and land exactly,
// the tip prices a block, hovering links a gather and its consumer, state
// rides the hash. The page's m="1"/m="2"
// figures are pinned (no knob, no hash); the closing figure's knob drives
// its fit chart too. Plus the 'doesn't fit' bars: a PP1 · ZeRO-3 · one
// microbatch snapshot
const M = await import('/src/fsdpsched.js');
for (const [id, m] of [['fsdpsched', 1], ['fsdpsched-2mb', 2]]) {
  const p = document.getElementById(id);
  T.check(`${id}: pinned at ${m}, no knob`, p.m === m && !p.querySelector('[data-knob]') && p.querySelectorAll('rect[data-op=RS]').length === 2 * m, p.m);
}
const w = document.getElementById('fsdpsched-fit');
const q = (s) => w.querySelector(s), qa = (s) => [...w.querySelectorAll(s)];
const btn = (k, v) => q(`[data-knob=${k}] [data-v="${v}"]`);
const near = (a, b, e = 1e-9) => Math.abs(a - b) < e;

T.check('move = 399.3 MB, 7.99 ms', near(M.MOVE_B, 399333231.75, 1) && near(M.MOVE_S * 1e3, 7.9867, 1e-3), M.MOVE_B);
T.check('resting: 8 microbatches', btn('m', 8).classList.contains('on'), '');
T.check('resting readout: 61% exposed', /514\.6 ms.*312\.6 ms \(61%\)/.test(q('[data-readout]').textContent), q('[data-readout]').textContent);
T.check('resting header fits', q('[data-hdr]').getComputedTextLength() < 738, q('[data-hdr]').getComputedTextLength());
T.check('resting: 48 moves drawn', qa('rect[data-op=AG]').length + qa('rect[data-op=RS]').length === 48, qa('rect[data-op=AG]').length);
T.check('readout fits', q('[data-readout]').getComputedTextLength() < 738, q('[data-readout]').getComputedTextLength());
const comp0 = M.schedule(8).comp;
for (const m of M.MS) {
  const S = M.schedule(M.LOCAL / m), tag = `m ${m}`;
  const byT = (tr) => S.blocks.filter((b) => b.track === tr && b.op !== 'idle').sort((a, b) => a.t0 - b.t0);
  const cs = byT('c'), ns = byT('n');
  T.check(`${tag}: compute conserved`, near(S.comp, comp0, 1e-12), S.comp);
  T.check(`${tag}: 6 moves per microbatch`, ns.length === 6 * S.m && ns.filter((b) => b.op === 'RS').length === 2 * S.m, ns.length);
  T.check(`${tag}: no track double-books`, [cs, ns].every((a) => a.every((b, i) => !i || b.t0 >= a[i - 1].t1 - 1e-12)), '');
  T.check(`${tag}: each compute op after its gather`, cs.every((c) => { const g = S.blocks.find((b) => b.k === `AG${c.layer}${c.op}:${c.mb}`); return g && c.t0 >= g.t1 - 1e-12; }), '');
  T.check(`${tag}: each gather as late as possible`, ns.every((g, i) => g.op !== 'AG' || near(g.t1, Math.min(ns[i + 1]?.t0 ?? Infinity, S.blocks.find((b) => b.k === `${g.k[3]}${g.layer}:${g.mb}`).t0), 1e-12)), '');
  T.check(`${tag}: each reduce-scatter after its backward`, ns.filter((b) => b.op === 'RS').every((r) => r.t0 >= S.blocks.find((b) => b.k === `B${r.layer}:${r.mb}`).t1 - 1e-12), '');
}
T.check('overlap @8: exposed = first gather + last (FP32) reduce-scatter', near(M.schedule(8).exposed, 3 * M.MOVE_S, 1e-12), '');
T.check('fixed axis covers every state', M.T_MAX === 0.52, M.T_MAX);

T.check('m 1: each gather abuts its consumer', M.schedule(8).blocks.filter((b) => b.op === 'AG').every((g) => near(g.t1, M.schedule(8).blocks.find((b) => b.k === `${g.k[3]}${g.layer}:${g.mb}`).t0, 1e-12)), '');

// memory: preallocated staging (two BF16 layers for gathers, two FP32 gradient layers for reduce-scatters)
// under activations that ramp in over each forward and out over its backward;
// every figure shares one unlabeled axis (the m 1 peak fills it)
const topY = (el) => Math.min(...[...el.querySelectorAll('polygon[data-mem]')].flatMap((p) => p.getAttribute('points').split(' ').map((xy) => +xy.split(',')[1])));
const myOf = (v) => 22 + 56 - 52 * v / M.MEM_MAX;
const pk = (S) => Math.max(...M.memPts(S).map(([, v]) => v[0] + v[1] + v[2]));
for (const m of M.MS) {
  const S = M.schedule(M.LOCAL / m), P = M.memPts(S);
  T.check(`m ${m}: memory starts and ends empty`, P[0][1].every((v) => v === 0) && P.at(-1)[1].every((v) => near(v, 0, 1)), JSON.stringify([P[0][1], P.at(-1)[1]]));
  T.check(`m ${m}: staging holds for the whole step`, P.filter(([t], i) => (i % 2 ? t >= 0 && t < S.step : t > 0 && t <= S.step)).every(([, v]) => v[0] === 2 * M.LAYER_B && v[1] === 2 * M.GRAD_B), '');
  T.check(`m ${m}: activation peak = two layers of one microbatch`, near(Math.max(...P.map(([, v]) => v[2])), 2 * (M.LOCAL / m) * M.ACT_SEQ, 1e3), '');
  // the staging is enough: a gathered layer is live from its gather's start to
  // its consumer's end, a gradient from its backward's start to its reduce-scatter's end
  const at = (k) => S.blocks.find((b) => b.k === k), iv = { AG: [], RS: [] };
  for (const b of S.blocks) {
    if (b.op === 'AG') iv.AG.push([b.t0, at(`${b.k[3]}${b.layer}:${b.mb}`).t1]);
    if (b.op === 'RS') iv.RS.push([at(`B${b.layer}:${b.mb}`).t0, b.t1]);
  }
  const most = (xs) => Math.max(...xs.map(([t]) => xs.filter(([a, b]) => a <= t + 1e-12 && t < b - 1e-12).length));
  T.check(`m ${m}: two staging layers each suffice (and are needed)`, most(iv.AG) === 2 && most(iv.RS) === 2, `${most(iv.AG)} ${most(iv.RS)}`);
}
T.check('memory axis = the m 1 peak', M.MEM_MAX === Math.max(...M.MS.map((m) => pk(M.schedule(M.LOCAL / m)))) && M.MEM_MAX === pk(M.schedule(8)), M.MEM_MAX);
for (const [id, m] of [['fsdpsched', 1], ['fsdpsched-2mb', 2], ['fsdpsched-fit', 8]])
  T.check(`${id}: memory on the shared axis`, near(topY(document.getElementById(id)), +myOf(pk(M.schedule(M.LOCAL / m))).toFixed(2), 0.006), topY(document.getElementById(id)));
T.check('legend fits', q('svg').getBBox().width <= 738, q('svg').getBBox().width);

const L2 = w.querySelector('dsv3-layer');
T.check('fit bars: ZeRO-3 · PP1 · EP64 · m 8, no knobs', L2.pp === 1 && L2.ep === 64 && L2.zero === 3 && L2.mb === 8 && !L2.querySelector('[data-knob]'), `${L2.pp} ${L2.ep} ${L2.zero} ${L2.mb}`);
const acts8 = [...L2.querySelectorAll('.lv-bar text')].map((t) => t.textContent).join('|');
btn('m', 1).click();
await T.tick(400);
T.check('fit bars follow the knob', L2.mb === 1 && L2.mbs === 8 && [...L2.querySelectorAll('.lv-bar text')].map((t) => t.textContent).join('|') !== acts8, `${L2.mb} ${L2.mbs}`);
T.check('m 1: 6 moves drawn', qa('rect[data-op=AG]').length + qa('rect[data-op=RS]').length === 6, qa('rect[data-op=AG]').length);
T.check('m 1 readout', /226\.0 ms = compute 202\.0 ms \+ exposed IB 24\.0 ms \(11%\)/.test(q('[data-readout]').textContent), q('[data-readout]').textContent);
const S1 = M.schedule(8), sx = (t) => 64 + (738 - 22 - 64) * t / M.T_MAX;
T.check('tween lands exactly', S1.blocks.every((b) => near(+q(`rect[data-k="${b.k}"]`).getAttribute('x'), +sx(b.t0).toFixed(2), 0.006)), '');
T.check('memory tween lands exactly', near(topY(w), +myOf(pk(M.schedule(8))).toFixed(2), 0.006) && q('polygon[data-mem="2"]').getAttribute('points') === (w.render(), q('polygon[data-mem="2"]').getAttribute('points')), topY(w));
T.check('hash state', decodeURIComponent(location.hash).includes('f:fsdpsched-fit={"m":1}') && !/f:fsdpsched=|f:fsdpsched-2mb/.test(decodeURIComponent(location.hash)), location.hash);
btn('m', 8).click();
await T.tick(400);
T.check('every collective labelled', qa('rect[data-op=AG], rect[data-op=RS]').length === 48 && qa('text').filter((t) => /^[12]$/.test(t.textContent)).length === 48, qa('text').filter((t) => /^[12]$/.test(t.textContent)).length);
const r = q('rect[data-k="AG2B:3"]'), b = r.getBoundingClientRect();
r.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: b.x + b.width / 2, clientY: b.y + b.height / 2 }));
await T.tick(30);
const tip = w.querySelector('.fsc > .dsv3-tip')?.textContent ?? '';
T.check('tip: gather before backward, microbatch, price', /layer 2's weights, before its backward · microbatch 4 of 8/.test(tip) && /399\.3 MB[^]*÷ 50 GB\/s = 8\.0 ms/.test(tip), tip);
const rr = q('rect[data-k="RS1:3"]'), rb = rr.getBoundingClientRect();
rr.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: rb.x + rb.width / 2, clientY: rb.y + rb.height / 2 }));
await T.tick(30);
const rtip = w.querySelector('.fsc > .dsv3-tip')?.textContent ?? '';
T.check('tip: a reduce-scatter moves FP32 gradients, two moves\' worth', /reduce-scatter of layer 1's gradients · microbatch 4 of 8/.test(rtip) && /798\.7 MB \(FP32\)[^]*= 16\.0 ms/.test(rtip), rtip);

// hover link: a gather and the op it feeds tint together, both ways
const hl = () => qa('rect.hl').map((r) => r.dataset.k).sort().join(' ');
q('rect[data-k="B1:5"]').dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
T.check('hover compute → its gather', hl() === 'AG1B:5 B1:5', hl());
q('rect[data-k="AG2F:0"]').dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
T.check('hover gather → its consumer', hl() === 'AG2F:0 F2:0', hl());
q('rect[data-k="RS1:0"]').dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
T.check('hover elsewhere clears', hl() === '', hl());

// the ZeRO-3 · one-microbatch bars that don't fit: a static snapshot,
// prose-width so margin notes flow beside it
const L = document.getElementById('fsdp-nofit');
T.check('nofit: PP1 · ZeRO-3 · one microbatch of the whole 8', L.pp === 1 && L.zero === 3 && L.mb === 1 && L.mbs === 8, `${L.pp} ${L.zero} ${L.mb} ${L.mbs}`);
T.check('nofit: prose width, no float clearing', L.getBoundingClientRect().width <= 760 && L.querySelector('.lv').scrollWidth <= 760 && getComputedStyle(L).clear === 'none', L.querySelector('.lv').scrollWidth);
T.check('nofit: no knobs', !L.querySelector('[data-knob]'), '');
T.check('nofit: no wire diagram', !L.querySelector('svg.lv-svg, .lv-diagram') && !L.querySelector('[data-op]'), '');
T.done();
