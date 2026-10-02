// @page studies/03-roofline.html
// the comms-vs-compute line: the whole step on the average GPU (the cells'
// 7.5 sequences) against microbatches per step. Its three terms are the
// page's cells (T⁺c, T⁺EP, Tmb); EP is flat, ZeRO-3 pays one Tmb per
// microbatch, so comms is straight; the excess over compute is exposed, plotted
// as its share of the step (one 0–80% scale with <dsv3-ppcurve>). The curve,
// dots and labels sit where the numbers say; the tip prices a point.
// Below, the busiest GPU's peak memory on the same m axis against a soft 80 GiB line
const M = await import('/src/fsdpsched.js');
const w = document.querySelector('dsv3-fsdpcurve'), q = (s) => w.querySelector(s), qa = (s) => [...w.querySelectorAll(s)];
const near = (a, b, e = 1e-9) => Math.abs(a - b) < e;
T.check('T⁺c = 6.389277268 s (6N + MLA up-projection replay at π^sol_fp8; head, routers, attention core with FlashAttention QKᵀ replay at π^sol_bf16)', near(M.STEP.comp, 6.389277268, 1e-9), M.STEP.comp);
T.check('T⁺EP = 5.419971379 s', near(M.STEP.ep, 5.419971379, 1e-9), M.STEP.ep);
T.check('Tmb = 1.924700956 s', near(M.STEP.mb, 1.924700956, 1e-9), M.STEP.mb);
T.check('crossover: comms = compute at m = 0.50, off the axis (below one microbatch)', near(M.at(M.CROSS).comms, M.STEP.comp, 1e-12) && near(M.CROSS, 0.503614, 1e-6) && !q('circle[data-cross]'), M.CROSS);
const pct = (m) => Math.round(100 * M.at(m).exposed / M.at(m).step);
T.check('exposed shares: 13 · 31 · 51 · 69 %', [1, 2, 4, 8].map(pct).join(' ') === '13 31 51 69', [1, 2, 4, 8].map(pct).join(' '));
T.check('labels follow', qa('text[data-pct]').map((t) => t.textContent).join(' ') === '13% 31% 51% 69%', qa('text[data-pct]').map((t) => t.textContent).join(' '));
T.check('one Tmb per microbatch', [1, 2, 4, 8].every((m) => near(M.at(m).fsdp, m * M.STEP.mb, 1e-12) && M.at(m).ep === M.STEP.ep), '');
// geometry: the exposed share as a curve, the dots on it, the crossover at 0
const cx = (m) => 52 + (570 - 52) * (m - 1) / 7, cy = (s) => 236 - 200 * s / 0.8, sh = (m) => M.at(m).exposed / M.at(m).step;
T.check('the four points sit on the exposed-share curve', [1, 2, 4, 8].every((m) => { const c = q(`circle[data-m="${m}"]`); return near(+c.getAttribute('cx'), cx(m), 0.006) && near(+c.getAttribute('cy'), cy(sh(m)), 0.006); }), '');
const cur = q('polyline[data-idle]').getAttribute('points').split(' ').map((p) => p.split(',').map(Number));
T.check('the curve: 1 − compute ÷ comms (0 below a crossover)', cur.every(([x, y]) => { const m = 1 + 7 * (x - 52) / 518; return near(y, cy(Math.max(0, 1 - M.STEP.comp / M.at(m).comms)), 0.02); }) && near(cur.at(-1)[1], cy(sh(8)), 0.006), cur.length);
T.check('second x axis: microbatch size 7.5 ÷ m sequences, ticks at m = 7.5 ÷ size', qa('line[data-mbs]').every((l) => near(+l.getAttribute('x1'), cx(7.5 / +l.dataset.mbs), 0.006)) && qa('line[data-mbs]').map((l) => l.dataset.mbs).join(' ') === '7.5 5 4 3 2 1.5 1.25 1', qa('line[data-mbs]').length);
T.check('the y axis covers the m 8 peak', sh(8) < 0.8, sh(8));
T.check('header fits', q('[data-hdr]').getComputedTextLength() <= 700, q('[data-hdr]').getComputedTextLength());
T.check('nothing spills past the svg', q('svg').getBBox().x >= 0 && q('svg').getBBox().x + q('svg').getBBox().width <= 700, JSON.stringify(q('svg').getBBox()));
// the memory panel: the page's cells (T1 at mb 1 ÷ m in its activations) + the schedule figures' staging
const GiB = 2 ** 30, r = (m) => M.memAt(m);
T.check('shards 4.27 GiB, staging 4.57 GiB (BF16 gathers + FP32 reduce-scatters), A1 311.11 GiB at one microbatch', near(M.MEMB.shards / GiB, 4.27, 0.005) && near(M.MEMB.staging / GiB, 4.57, 0.005) && near(M.MEMB.acts1 / GiB, 311.11, 0.005), JSON.stringify(M.MEMB));
T.check('m 8 = the fit bars\' 43.16 GiB + staging', near((r(8).total - M.MEMB.staging) / GiB, 43.16, 0.005), r(8).total / GiB);
T.check('of the MBs states only 8 is under 80 GiB (4: 86.62 GiB)', [1, 2, 4].every((m) => r(m).total > M.CAP) && r(8).total < M.CAP && near(r(4).total / GiB, 86.62, 0.005), [1, 2, 4, 8].map((m) => (r(m).total / GiB).toFixed(2)).join(' '));
T.check('no crossing is marked: the line is soft', !q('[data-fit]') && !/fits from/.test(q('svg').textContent) && /only m = 8 fits/.test(q('[data-only]').textContent), '');
const my = (b) => 454 - 150 * Math.min(b, 200 * GiB) / (200 * GiB);
T.check('memory dots on the curve', [2, 4, 8].every((m) => near(+q(`circle[data-mm="${m}"]`).getAttribute('cy'), my(r(m).total), 0.006)), '');
T.check('no washout; the over-HBM zone is tinted down to the line', !q('[data-wash]') && near(+q('rect[data-over]').getAttribute('y') + +q('rect[data-over]').getAttribute('height'), +q('line[data-cap]').getAttribute('y1'), 0.01), '');
T.check('memory header fits', q('[data-mhdr]').getComputedTextLength() <= 570 - 52, q('[data-mhdr]').getComputedTextLength());
const c = q('circle[data-m="4"]'), b = c.getBoundingClientRect();
c.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: b.x + b.width / 2, clientY: b.y + b.height / 2 }));
await T.tick(30);
const tip = w.querySelector('.dsv3-tip')?.textContent ?? '';
T.check('tip prices the point', /4 microbatches of 1\.875 sequences/.test(tip) && /exposed 51% of the step: 6\.73 s of 13\.12 s/.test(tip) && /comms 13\.12 s = EP 5\.42 s \+ 4 × Tmb 1\.92 s; compute 6\.39 s/.test(tip) && /memory 86\.62 GiB on the busiest GPU: 6\.62 GiB over/.test(tip), tip);
const G = q('[data-guide]'), gx = () => +/translate\(([\d.]+)/.exec(G.getAttribute('transform'))[1];
T.check('the dot\'s guide: exposed point down through memory to the axis', G.getAttribute('display') === 'inline' && near(gx(), cx(4), 0.006)
  && near(+G.querySelector('[data-g=t]').getAttribute('cy'), cy(sh(4)), 0.006) && near(+G.querySelector('[data-g=m]').getAttribute('cy'), my(r(4).total), 0.006) && +G.querySelector('line').getAttribute('y2') === 454, G.outerHTML);
// off the dots, the whole column snaps to the nearest MBs state: here empty memory panel at m ≈ 1.2 → m 1, whose memory is off-scale
const sv = q('svg').getBoundingClientRect(), at_ = (x, y) => { const ev = new MouseEvent('mousemove', { bubbles: true, clientX: sv.x + x, clientY: sv.y + y }); q('svg').dispatchEvent(ev); };
at_(cx(1.2), 300); await T.tick(30);
T.check('a non-dot snaps: m 1, ring on its point, none on the off-scale memory', near(gx(), cx(1), 0.006) && near(+G.querySelector('[data-g=t]').getAttribute('cy'), cy(sh(1)), 0.006)
  && G.querySelector('[data-g=m]').getAttribute('visibility') === 'hidden' && /^1 microbatch of 7\.5 sequences/.test(w.querySelector('.dsv3-tip').textContent), w.querySelector('.dsv3-tip').textContent.slice(0, 40));
at_(cx(5.9), 150); await T.tick(30);
T.check('m ≈ 5.9 in the top panel snaps to 4', near(gx(), cx(4), 0.006) && /^4 microbatches/.test(w.querySelector('.dsv3-tip').textContent), gx());
q('svg').parentElement.dispatchEvent(new MouseEvent('mouseleave')); await T.tick(10);
T.check('leaving hides the guide', G.getAttribute('display') === 'none', G.getAttribute('display'));
T.done();
