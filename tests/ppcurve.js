// @page studies/03-roofline.html
// the pipelined sibling of the comms-vs-compute line: EP64, ZeRO-1, PP8 under
// DualPipeV, same panels and scales. Comms are flat in m and assumed hidden
// (at β^sol_IB they exceed compute by 0.64 s: unmodeled);
// the bubble, (PP − 1)/(3m) of compute, is the only time m buys back, and the
// busiest rank's memory falls ∝ 1/m: only one sequence per microbatch fits.
// Dashed over it, plain 1F1B's bubble, (PP − 1)/m: three times DualPipeV's.
const M = await import('/src/fsdpsched.js');
const w = document.querySelector('dsv3-ppcurve'), q = (s) => w.querySelector(s), qa = (s) => [...w.querySelectorAll(s)];
const near = (a, b, e = 1e-9) => Math.abs(a - b) < e, GiB = 2 ** 30;
T.check('PP8 × DP256, 60 sequences per pipeline, MBs states 20 · 30 · 60', M.PPR === 8 && M.PDP === 256 && M.PLB === 60 && M.PMIN === 16 && M.PMS.join() === '20,30,60', M.PMS.join());
T.check('compute and EP are the FSDP chart\'s', M.PSTEP.comp === M.STEP.comp && M.PSTEP.ep === M.STEP.ep, '');
T.check('PP sends: 15/8 boundaries × 60 × 4,096 tokens × 2 ways × BF16 7,168 over 50 GB/s = 0.2642 s', near(M.PSTEP.pp, 15 / 8 * 60 * 4096 * 2 * 7168 * 2 / 50e9, 1e-12), M.PSTEP.pp);
T.check('ZeRO-1 sync 0.146 s', near(M.PSTEP.z1, 0.1460375046, 1e-9), M.PSTEP.z1);
T.check('comms 5.83 s, flat in m (under compute, 6.39 s)', M.PMS.every((m) => near(M.atP(m).comms, 5.83, 0.005)), M.atP(20).comms);
T.check('bubble = 7/(3m) of compute', M.PMS.every((m) => near(M.atP(m).bubble, M.PSTEP.comp * 7 / (3 * m), 1e-12)), '');
const pct = (m) => (100 * M.atP(m).bubble / M.atP(m).step).toFixed(1);
T.check('bubble shares of the step: 10.4 · 7.2 · 3.7 %', M.PMS.map(pct).join(' ') === '10.4 7.2 3.7', M.PMS.map(pct).join(' '));
T.check('1F1B bubble = 7/m of compute, its shares 25.9 · 18.9 · 10.4 %', M.PMS.every((m) => near(M.atP(m).bubble1, M.PSTEP.comp * 7 / m, 1e-12) && near(M.atP(m).step1, M.PSTEP.comp + M.atP(m).bubble1, 1e-12))
  && M.PMS.map((m) => (100 * M.atP(m).bubble1 / M.atP(m).step1).toFixed(1)).join(' ') === '25.9 18.9 10.4', '');
T.check('1F1B at m = 60 idles exactly what DualPipeV does at 20', near(M.atP(60).bubble1 / M.atP(60).step1, M.atP(20).bubble / M.atP(20).step, 1e-12), '');
T.check('1F1B labels follow', qa('text[data-pct1]').map((t) => t.textContent).join(' ') === '25.9% 18.9% 10.4%', qa('text[data-pct1]').map((t) => t.textContent).join(' '));
T.check('the hidden-comms flattery is disclosed', /no paired microbatch to hide the all-to-alls/.test(q('[data-flatter]')?.textContent ?? ''), '');
T.check('labels follow', qa('text[data-pct]').map((t) => t.textContent).join(' ') === '10.4% 7.2% 3.7%', qa('text[data-pct]').map((t) => t.textContent).join(' '));
// geometry
const px = (m) => 52 + (570 - 52) * (m - 16) / 44, cy = (s) => 236 - 200 * s / 0.8, sh = (m) => M.atP(m).bubble / M.atP(m).step, my = (b) => 454 - 150 * Math.min(b, 200 * GiB) / (200 * GiB);
T.check('dots on the bubble-share curve, 7 ÷ (3m + 7)', M.PMS.every((m) => { const c = q(`circle[data-m="${m}"]`); return near(+c.getAttribute('cx'), px(m), 0.006) && near(+c.getAttribute('cy'), cy(sh(m)), 0.006) && near(sh(m), 7 / (3 * m + 7), 1e-12); }), '');
const sh1 = (m) => M.atP(m).bubble1 / M.atP(m).step1;
T.check('hollow dots on the dashed 1F1B curve, 7 ÷ (m + 7)', M.PMS.every((m) => { const c = q(`circle[data-m1="${m}"]`); return near(+c.getAttribute('cx'), px(m), 0.006) && near(+c.getAttribute('cy'), cy(sh1(m)), 0.006) && near(sh1(m), 7 / (m + 7), 1e-12); }), '');
const cur1 = q('polyline[data-idle1]').getAttribute('points').split(' ').map((p) => p.split(',').map(Number));
T.check('the dashed curve spans m 16 → 60', q('polyline[data-idle1]').getAttribute('stroke-dasharray') && near(cur1[0][0], px(16), 0.006) && near(cur1[0][1], cy(sh1(16)), 0.006) && near(cur1.at(-1)[1], cy(sh1(60)), 0.006), JSON.stringify(cur1[0]));
const cur = q('polyline[data-idle]').getAttribute('points').split(' ').map((p) => p.split(',').map(Number));
T.check('the curve spans m 16 → 60', near(cur[0][0], px(16), 0.006) && near(cur[0][1], cy(sh(16)), 0.006) && near(cur.at(-1)[1], cy(sh(60)), 0.006), JSON.stringify(cur[0]));
T.check('bubble area closes on the 0% axis', q('polygon[data-area=bubble]').getAttribute('points').endsWith(`${px(16).toFixed(2)},236`), '');
T.check('header fits', q('[data-hdr]').getComputedTextLength() <= 700, q('[data-hdr]').getComputedTextLength());
T.check('memory header fits', q('[data-mhdr]').getComputedTextLength() <= 570 - 52, q('[data-mhdr]').getComputedTextLength());
T.check('nothing spills past the svg', q('svg').getBBox().x >= 0 && q('svg').getBBox().x + q('svg').getBBox().width <= 700, JSON.stringify(q('svg').getBBox()));
// memory: the cells' T1 on the busiest rank (rank 1, 8 MoE layers)
const r = (m) => M.memP(m).total / GiB;
T.check('busiest rank 1: shards 20.97 GiB, 40.77 GiB of activations per sequence', M.MEMP.rank === 1 && near(M.MEMP.shards / GiB, 20.97, 0.005) && near(M.MEMP.actsSeq / GiB, 40.77, 0.005), JSON.stringify(M.MEMP));
T.check('143.29 · 102.52 · 61.74 GiB: only m 60 is under 80 GiB', [20, 30, 60].map((m) => r(m).toFixed(2)).join(' ') === '143.29 102.52 61.74' && /only m = 60 fits/.test(q('[data-only]').textContent), [20, 30, 60].map((m) => r(m).toFixed(2)).join(' '));
T.check('memory dots on the curve', M.PMS.every((m) => near(+q(`circle[data-mm="${m}"]`).getAttribute('cy'), my(M.memP(m).total), 0.006)), '');
// the tip and guide
const c = q('circle[data-m="30"]'), b = c.getBoundingClientRect();
c.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: b.x + b.width / 2, clientY: b.y + b.height / 2 }));
await T.tick(30);
const tip = w.querySelector('.dsv3-tip')?.textContent ?? '';
T.check('tip prices the point', /^30 microbatches of 2 sequences/.test(tip) && /bubble 7\.2% of the step: 0\.50 s of 6\.89 s/.test(tip) && /= 7 ÷ \(3 × 30\) of compute 6\.39 s; comms 5\.83 s stay under it/.test(tip) && /1F1B: bubble 18\.9% of 7\.88 s, 7 ÷ 30 of compute \(comms assumed hidden\)/.test(tip) && /memory 102\.52 GiB on rank 1: 22\.52 GiB over/.test(tip), tip);
const G = q('[data-guide]'), gx = () => +/translate\(([\d.]+)/.exec(G.getAttribute('transform'))[1];
T.check('the dot\'s guide: bubble point down through memory', G.getAttribute('display') === 'inline' && near(gx(), px(30), 0.006)
  && near(+G.querySelector('[data-g=t]').getAttribute('cy'), cy(sh(30)), 0.006) && near(+G.querySelector('[data-g=t1]').getAttribute('cy'), cy(sh1(30)), 0.006)
  && near(+G.querySelector('line').getAttribute('y1'), cy(sh1(30)) + 6, 0.006) && near(+G.querySelector('[data-g=m]').getAttribute('cy'), my(M.memP(30).total), 0.006), G.outerHTML);
const sv = q('svg').getBoundingClientRect(), at_ = (x, y) => q('svg').dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: sv.x + x, clientY: sv.y + y }));
at_(px(24.5), 150); await T.tick(30);
T.check('m ≈ 24.5 snaps to 20', near(gx(), px(20), 0.006) && /^20 microbatches of 3 sequences/.test(w.querySelector('.dsv3-tip').textContent), gx());
at_(px(46), 300); await T.tick(30);
T.check('m ≈ 46 in the memory panel snaps to 60, ring on its memory', near(gx(), px(60), 0.006) && G.querySelector('[data-g=m]').getAttribute('visibility') === 'visible' && /^60 microbatches of 1 sequence\n/.test(w.querySelector('.dsv3-tip').textContent), gx());
const h = q('circle[data-m1="20"]'), hb = h.getBoundingClientRect();
h.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: hb.x + hb.width / 2, clientY: hb.y + hb.height / 2 })); await T.tick(30);
T.check('a hollow 1F1B dot prices its m', near(gx(), px(20), 0.006) && /^20 microbatches of 3 sequences/.test(w.querySelector('.dsv3-tip').textContent) && /1F1B: bubble 25\.9%/.test(w.querySelector('.dsv3-tip').textContent), w.querySelector('.dsv3-tip').textContent);
q('svg').parentElement.dispatchEvent(new MouseEvent('mouseleave')); await T.tick(10);
T.check('leaving hides the guide', G.getAttribute('display') === 'none', G.getAttribute('display'));
T.done();
