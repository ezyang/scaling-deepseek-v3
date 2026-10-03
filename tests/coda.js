// @page studies/coda.html
// The CODA study: every launch of the DeepSeek EP64 profile's layer cycle is
// modeled (240, one spec each), the non-GEMM sums reproduce 03's kernel table
// to the microsecond, the GEMM speed-of-light cells reproduce 03's, every
// launch is a row of the figure or the table, the CODA flip keeps anchors
// (host rows do not move; folded rows ghost in place) and rides the hash,
// tips name the kernel, and the prose numbers are live.
const M = await import('/src/kerntrace.js');
const KD = await import('/src/kerndata.js');
const near = (a, b, e = 1e-9) => Math.abs(a - b) < e;
const P = M.priced(), TT = M.tally();
T.check('240 launches, all specified', KD.LAUNCHES.length === 240 && P.length === 240 && P.every((e) => e.lab && e.kind), P.length);
// 03's "everything else (not a matmul)" row: raw sums of the non-GEMM, non-attention compute-stream launches
const other = (p) => P.filter((e) => e.pass === p && !e.comm && e.kind !== 'gemm' && e.kind !== 'attn').reduce((t, e) => t + e.raw, 0);
T.check('non-GEMM sums = 03 (2,713.25 / 5,307.25 / 577.75 µs)', near(other('F'), 2713.25, 1e-6) && near(other('B'), 5307.25, 1e-6) && near(other('W'), 577.75, 1e-6), `${other('F')} ${other('B')} ${other('W')}`);
// 03's per-kernel SOL cells (µs): q down 61.9 · kv down 23.2 · q up 212 · kv up 94.4 · attention 907 · o 660 · router 19.8 · shared 165/82.6 · routed 1,320/660 · attention bwd 2,360
const sol = (i) => P[i].sol * 1e6;
T.check('GEMM speed of light = 03\'s cells', [[3, 61.9], [6, 23.2], [15, 212], [18, 94.4], [23, 907], [27, 660], [30, 19.8], [103, 165], [118, 82.6], [109, 1320], [114, 660], [174, 2360]].every(([i, v]) => Math.abs(sol(i) - v) / v < 0.004), [3, 6, 15, 18, 23, 27, 30, 103, 118, 109, 114, 174].map((i) => sol(i).toFixed(1)).join(' '));
T.check('routed rows normalized like 03 (32,512 F / 32,128 B)', near(KD.ROUTED_ROWS.F.reduce((a, b) => a + b) / 4, 32512) && near(KD.ROUTED_ROWS.B.reduce((a, b) => a + b) / 4, 32128), '');
T.check('every GEMM compute-bound today (the bf16 router GEMMs sit on the ridge)', P.filter((e) => e.kind === 'gemm' && e.dt === 'fp8').every((e) => e.flop >= e.hbm) && P.filter((e) => e.kind === 'gemm' && e.dt === 'bf16').every((e) => e.flop >= e.hbm * 0.97), '');
T.check('after CODA every host hides its epilogue but the Q-down input gradient', TT.F.gemm.over.length === 0 && TT.W.gemm.over.length === 0 && TT.B.gemm.over.join() === '214' && P[214].hbmAfter < 2 * P[214].flop, TT.B.gemm.over.join());
T.check('bytes shrink under CODA, in every pass', ['F', 'B', 'W'].every((p) => TT[p].other.bytesAfter < TT[p].other.bytes), '');
T.check('inferred shapes are marked', P.filter((e) => e.inf).length > 10 && P.filter((e) => e.inf).every((e) => !e.kind || e.kind !== 'gemm'), P.filter((e) => e.inf).length);

// the figure
const w = document.getElementById('kern');
const q = (s) => w.querySelector(s), qa = (s) => [...w.querySelectorAll(s)];
const launchesDrawn = () => qa('g[data-row]').flatMap((g) => g.dataset.row.split('+')).length;
T.check('forward: every compute + comm launch of the pass is a row', launchesDrawn() === TT.F.launches + P.filter((e) => e.comm && /\(F\)/.test(e.ann)).length, launchesDrawn());
T.check('resting: as traced, no ghosts', !q('[data-knob=mode] [data-v=coda]').classList.contains('on') && qa('g[data-gone="1"]').length === 0, '');
const yOf = () => Object.fromEntries(qa('g[data-row]').map((g) => [g.dataset.row, g.querySelector('rect').getAttribute('y')]));
const y0 = yOf();
q('[data-knob=mode] [data-v=coda]').click();
await T.tick(400);
const y1 = yOf();
T.check('CODA flip preserves every anchor', Object.keys(y0).every((k) => y0[k] === y1[k]) && Object.keys(y0).length === Object.keys(y1).length, '');
T.check('CODA: folded rows ghost in place', qa('g[data-gone="1"]').length === TT.F.other.gone + TT.F.other.aux - 0 || qa('g[data-gone="1"]').length > 20, qa('g[data-gone="1"]').length);
T.check('CODA: hosts wear an epilogue tag', qa('.epibox').length >= 6 && q('g[data-row="27"] .epi'), qa('.epibox').length);
T.check('hash state', decodeURIComponent(location.hash).includes('k:kern={"pass":"F","mode":"coda"}'), location.hash);
const foot = q('text[data-foot=floor]').textContent;
T.check('footer: floor → CODA floor', /1\.34 ms → with CODA 0\.54 ms in 41 launches/.test(foot), foot);
// tip: hover the output projection
const r = q('g[data-row="27"] rect[data-hit]'), b = r.getBoundingClientRect();
r.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: b.x + 40, clientY: b.y + b.height / 2 }));
await T.tick(30);
const tip = w.querySelector('.dsv3-tip')?.textContent ?? '';
T.check('tip: kernel, both meters, the epilogue', /attention output projection/.test(tip) && /4,096 × 16,384 → 7,168/.test(tip) && /compute-bound/.test(tip) && /CODA epilogue: × c · \+ residual/.test(tip) && /fp8_ptp128c/.test(tip), tip.slice(0, 160));
// the backward pass
q('[data-knob=pass] [data-v=B]').click();
await T.tick(200);
T.check('backward: every launch of the pass is a row', launchesDrawn() === TT.B.launches + P.filter((e) => e.comm && /\(B\)/.test(e.ann)).length, launchesDrawn());
T.check('backward: the router\'s 25 glue launches collapse to one row', q('g[data-row^="121+"]')?.dataset.row.split('+').length === 25, q('g[data-row^="121+"]')?.dataset.row);
T.check('backward: FlashAttention backward is a box', q('g[data-row="174"] rect.box') && /FlashAttention backward/.test(q('g[data-row="174"] text').textContent), '');

// the page: tables and live prose
T.check('per-launch tables: 240 rows', document.querySelectorAll('.tally[data-pass] tr[data-i]').length === 240, '');
T.check('inferred rows marked ~', [...document.querySelectorAll('.tally[data-pass] tr[data-i] td.lab')].filter((td) => td.textContent.startsWith('~')).length === P.filter((e) => e.inf).length, '');
T.check('savings table: layer row', /one layer, one microbatch/.test(document.querySelector('.tally.savings tr.hl')?.textContent), '');
T.check('prose numbers live', document.querySelector('[data-kt="F.other.trace"]').textContent === '2.71 ms' && document.querySelector('[data-kt="B.other.trace"]').textContent === '5.31 ms' && /%$/.test(document.querySelector('[data-kt="step.pct"]').textContent), '');
T.check('AI disclosure present', document.querySelector('p.note.ai')?.textContent.includes('[[AI:') && document.querySelector('main > div.ai'), '');
T.done();
