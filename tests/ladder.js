// @page studies/03-roofline.html
// the scaling ladder: op-class shares of the counted FLOPs up the DSv3 family vs 6ND
const L = await import('/src/ladder.js');
const { PARAMS } = await import('/src/params.js');
const w = document.querySelector('dsv3-ladder');
await T.tick(300);
const share = (k) => +w.querySelector(`[data-share="${k}"]`).dataset.true;
T.check('family formula reproduces DeepSeek’s 36.6B activated exactly', L.N_DSV3 === PARAMS.activeTotal, L.N_DSV3);
T.check('DSv3 is exactly s = 1', L.ladderPoint(L.LOG_C_DSV3, 4096).s === 1, '');
const pct = (k) => w.querySelector(`[data-share="${k}"]`).textContent;
T.check('resting: 6ND counts 87.7% (misses 12.3%)', pct('six') === '87.7%', pct('six'));
T.check('DSv3 preset lit', w.querySelector('[data-knob="preset"] button').classList.contains('on'), '');
const sum = L.CLASSES.reduce((t, k) => t + share(k.id), 0);
T.check('class shares sum to 1', Math.abs(sum - 1) < 1e-12, sum);
// 6N = the in-N matmuls + 6 × (norm weights + router bias): a hair above their share
const inN = share('proj') + share('router') + share('head');
T.check('6ND share = in-N matmuls + a norm-weight sliver (<0.01%)', share('six') > inN && share('six') - inN < 1e-4, share('six') - inN);
// asymptotics: at 4K, 6ND's miss shrinks monotonically from 1e21 up; the head's share collapses
const miss = (lc, seq) => { const p = L.ladderPoint(lc, seq); return 1 - p.sixN / p.total; };
const ms = [21, 22, 23, 24, 25, 26, 27].map((e) => miss(e, 4096));
T.check('6ND miss shrinks with C (1e21 → 1e27)', ms.every((m, i) => !i || m < ms[i - 1]), ms.map((m) => m.toFixed(3)));
T.check('small rungs: the fixed-vocab head dominates (6ND accurate again at 1e19)', miss(19, 4096) < miss(20, 4096), `${miss(19, 4096)} vs ${miss(20, 4096)}`);
// sequence length: + → 8K grows attention; height reserved through the tween
const h0 = w.getBoundingClientRect().height, a0 = share('attn');
w.querySelector('[data-knob="seq"] button[data-dir="1"]').click(); await T.tick(350);
T.check('S = 8,192: attention share grows', share('attn') > a0 && w.querySelector('[data-knob="seq"] select').value === '8192', share('attn'));
T.check('knob flip: height reserved', Math.abs(w.getBoundingClientRect().height - h0) < 1, w.getBoundingClientRect().height);
w.querySelector('[data-knob="seq"] select').value = '131072'; w.querySelector('[data-knob="seq"] select').dispatchEvent(new Event('change')); await T.tick(350);
T.check('S = 131,072: attention the majority at DSv3', share('attn') > 0.75, share('attn'));
w.querySelector('[data-knob="seq"] select').value = '1048576'; w.querySelector('[data-knob="seq"] select').dispatchEvent(new Event('change')); await T.tick(350);
T.check('S = 1,048,576 (the top): + disabled', w.querySelector('[data-knob="seq"] button[data-dir="1"]').disabled, '');
w.querySelector('[data-knob="seq"] select').value = '4096'; w.querySelector('[data-knob="seq"] select').dispatchEvent(new Event('change')); await T.tick(350);
// compute slider: 1e27 → smaller miss, preset unlit; near DSv3 it snaps back on
const r = w.querySelector('input[data-knob="c"]');
r.value = '27'; r.dispatchEvent(new Event('input')); await T.tick(50);
T.check('C = 1e27: preset unlit, 6ND closer', !w.querySelector('[data-knob="preset"] button').classList.contains('on') && share('attn') < a0 && w.querySelector('.cv').textContent === '1.00×10²⁷', w.querySelector('.cv').textContent);
r.value = (L.LOG_C_DSV3 + 0.02).toFixed(2); r.dispatchEvent(new Event('input')); await T.tick(50);
T.check('slider snaps onto DSv3', w.querySelector('[data-knob="preset"] button').classList.contains('on') && +w.querySelector('line[data-cursor]').dataset.cursor === L.LOG_C_DSV3, '');
// drag on the plot: press at the 1e20 tick
const hit = w.querySelector('rect[data-hit]'), b = hit.getBoundingClientRect();
const x20 = b.left + b.width / 8;
hit.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientX: x20, clientY: b.top + 50, pointerId: 1 }));
w.querySelector('.ld > div:nth-child(2)').dispatchEvent(new PointerEvent('pointerup', { bubbles: true, pointerId: 1 })); await T.tick(50);
T.check('press on the plot moves the cursor (1e20)', Math.abs(+w.querySelector('line[data-cursor]').dataset.cursor - 20) < 0.02, w.querySelector('line[data-cursor]').dataset.cursor);
// the S panel: the same stack across S at the cursor's rung; a press sets S continuously
// (snapping onto powers of two within 0.04 octaves), C untouched
const sel = w.querySelector('[data-knob="seq"] select');
const pressS = async (oct) => {   // oct: octaves above 512, of 11 (the svg re-renders: re-query the hit rect)
  const hs = w.querySelector('rect[data-hit="s"]'), bs = hs.getBoundingClientRect();
  hs.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientX: bs.left + bs.width * oct / 11, clientY: bs.top + 50, pointerId: 1 }));
  w.querySelector('.ld > div:nth-child(2)').dispatchEvent(new PointerEvent('pointerup', { bubbles: true, pointerId: 1 })); await T.tick(50);
};
await pressS(6.5);
const sOff = +w.querySelector('line[data-cursor-s]').dataset.cursorS;
T.check('press between ticks: S lands off-grid (~46K)', Math.abs(sOff - 512 * 2 ** 6.5) < 200 && Number.isInteger(sOff), sOff);
T.check('… the stepper shows it as a transient option', sel.value === String(sOff) && sel.querySelectorAll('option[data-off]').length === 1 && sel.options.length === 13, sel.value);
w.querySelector('[data-knob="seq"] button[data-dir="-1"]').click(); await T.tick(350);
T.check('… − steps down to the power of two below (32K), transient option gone', sel.value === '32768' && !sel.querySelector('option[data-off]'), sel.value);
await pressS(6.5); w.querySelector('[data-knob="seq"] button[data-dir="1"]').click(); await T.tick(350);
T.check('… + steps up to 64K', sel.value === '65536', sel.value);
await pressS(11);
T.check('right edge: S = 1,048,576, + disabled', sel.value === '1048576' && w.querySelector('[data-knob="seq"] button[data-dir="1"]').disabled, sel.value);
await pressS(6.02);
T.check('press near a tick snaps onto it (32,768), C stays', sel.value === '32768' && +w.querySelector('line[data-cursor-s]').dataset.cursorS === 32768
  && Math.abs(+w.querySelector('line[data-cursor]').dataset.cursor - 20) < 0.02, sel.value);
const P32 = L.ladderPoint(20, 32768);
T.check('legend = the (C, S) crosshair', Math.abs(share('attn') - P32.f.attn / P32.total) < 1e-12, share('attn'));
T.check('captions name each panel\'s fixed slice', /at S = 32,768/.test(w.textContent) && /at C = 1\.00×10²⁰/.test(w.textContent), '');
// the S panel's right edge (1M) at C = 1e20 = ladderPoint(20, 2²⁰)'s attention share
const P1M = L.ladderPoint(20, 2 ** 20), pts = w.querySelector('g[data-x="s"] polygon[data-band="attn"]').getAttribute('points').split(' ');
const attnTop = +pts[160].split(',')[1], attnBot = +pts[161].split(',')[1];   // the band's top and bottom edges at the last sample
T.check('S panel at 1M = the ladder point', Math.abs((attnBot - attnTop) / 200 - P1M.f.attn / P1M.total) < 0.003, (attnBot - attnTop) / 200);
w.querySelector('[data-knob="seq"] select').value = '4096'; w.querySelector('[data-knob="seq"] select').dispatchEvent(new Event('change')); await T.tick(350);
T.check('S stepper moves the S panel cursor', +w.querySelector('line[data-cursor-s]').dataset.cursorS === 4096, '');
T.check('URL hash carries the state', /l:ladder=/.test(decodeURIComponent(location.hash)), location.hash);
r.value = String(L.LOG_C_DSV3); r.dispatchEvent(new Event('input')); await pressS(7.3);
T.check('preset unlit at DSv3\'s C but off its S', +w.querySelector('line[data-cursor]').dataset.cursor === L.LOG_C_DSV3 && !w.querySelector('[data-knob="preset"] button').classList.contains('on'), '');
w.querySelector('[data-knob="preset"] button').click(); await T.tick(350);
T.check('preset returns to DSv3: its rung and 4K', +w.querySelector('line[data-cursor]').dataset.cursor === L.LOG_C_DSV3 && sel.value === '4096' && pct('six') === '87.7%'
  && w.querySelector('[data-knob="preset"] button').classList.contains('on'), pct('six'));
const pb = w.querySelector('[data-knob="preset"]').getBoundingClientRect(), tb = w.querySelector('.ld .top').getBoundingClientRect();
T.check('preset sits top right', tb.right - pb.right < 30 && pb.top - tb.top < 30, `${tb.right - pb.right} ${pb.top - tb.top}`);
// y axis: share ↔ log FLOP/token crossfades; lines + legend in absolute FLOP/token; height held
const yb = (v) => w.querySelector(`[data-knob="y"] button[data-y="${v}"]`);
T.check('resting y axis: share', yb('share').classList.contains('on') && !w.querySelector('[data-view="log"]'), '');
yb('log').click(); await T.tick(80);
T.check('mid-flip: both views drawn, crossfading', w.querySelector('[data-view="share"]') && w.querySelector('[data-view="log"]'), '');
await T.tick(300);
const P = L.ladderPoint(L.LOG_C_DSV3, 4096), abs = (k) => +w.querySelector(`[data-abs="${k}"]`).dataset.true;
T.check('log: one line per class, share view gone', L.CLASSES.every((k) => w.querySelector(`polyline[data-line="${k.id}"]`)) && !w.querySelector('[data-view="share"]'), '');
T.check('log legend = FLOP/token at the cursor', L.CLASSES.every((k) => abs(k.id) === P.f[k.id]) && abs('six') === P.sixN, abs('proj'));
T.check('y flip: height reserved', Math.abs(w.getBoundingClientRect().height - h0) < 1, w.getBoundingClientRect().height);
T.check('hash carries y', /"y":"log"/.test(decodeURIComponent(location.hash)), location.hash);
w.querySelector('[data-knob="seq"] button[data-dir="1"]').click(); await T.tick(350);
T.check('S flip in log mode stays in log', !w.querySelector('[data-view="share"]') && abs('attn') === L.ladderPoint(L.LOG_C_DSV3, 8192).f.attn, abs('attn'));
w.querySelector('[data-knob="seq"] button[data-dir="-1"]').click(); await T.tick(350);
yb('lin').click(); await T.tick(350);
T.check('linear: stacked FLOP/token areas, 6N dashed, legend still absolute', L.CLASSES.every((k) => w.querySelector(`polygon[data-area="${k.id}"]`)) && w.querySelector('polyline[data-six-lin]') && !w.querySelector('[data-view="log"]') && abs('proj') === P.f.proj, '');
// linear ceiling: continuous, 1.05 × the tallest stack over both panels (at DSv3, S = 4K: the S panel's 1M edge, 8.08 TFLOP, tops the C panel's 4.06)
const linTop = () => +w.querySelector('[data-view="lin"]').dataset.top;
const tot = (p) => Object.values(p.f).reduce((x, y) => x + y, 0), P1Mdsv3 = L.ladderPoint(L.LOG_C_DSV3, 2 ** 20);
T.check('linear ceiling = 1.05 × the S panel\'s 1M stack (8.08 TFLOP)', Math.abs(linTop() / (1.05 * tot(P1Mdsv3)) - 1) < 1e-12, linTop());
// gridlines under a moving ceiling: sweep it over 10¹⁰ … 10¹⁴ in 0.1% steps — every line's opacity
// changes by < 0.03 per step (lines slide and crossfade, never pop), and at least 4 opaque lines always remain
let worstPop = 0, fewest = 99, prevT = null;
for (let lt = 10; lt <= 14; lt += Math.log10(1.001)) {
  const t = new Map(L.linTicks(10 ** lt));
  if (prevT) for (const k of new Set([...t.keys(), ...prevT.keys()])) worstPop = Math.max(worstPop, Math.abs((t.get(k) ?? 0) - (prevT.get(k) ?? 0)));
  fewest = Math.min(fewest, [...t.values()].filter((o) => o >= 0.5).length); prevT = t;
}
T.check('linear gridlines never pop as the ceiling moves (< 0.03 per 0.1%)', worstPop < 0.03, worstPop);
T.check('… and always keep ≥ 2 lines at ≥ 50% opacity (worst: a ceiling of 3 coarse steps, the finer level half-faded)', fewest >= 2, fewest);
// a drag in lin mode follows the ceiling frame by frame (no settle)
const pressNow = (oct) => {
  const hs = w.querySelector('rect[data-hit="s"]'), bs = hs.getBoundingClientRect();
  hs.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientX: bs.left + bs.width * oct / 11, clientY: bs.top + 50, pointerId: 1 }));
  w.querySelector('.ld > div:nth-child(2)').dispatchEvent(new PointerEvent('pointerup', { bubbles: true, pointerId: 1 }));
};
pressNow(10.5); const t1 = linTop(); pressNow(10.45);
const P2 = L.ladderPoint(L.LOG_C_DSV3, +w.querySelector('line[data-cursor-s]').dataset.cursorS), P27 = L.ladderPoint(27, +w.querySelector('line[data-cursor-s]').dataset.cursorS);
T.check('S drag in lin mode: the ceiling tracks the drag (1.05 × max stack), no snapping to round values', linTop() < t1 && Math.abs(linTop() / (1.05 * Math.max(tot(P2), tot(P27))) - 1) < 1e-9, linTop());
w.querySelector('[data-knob="preset"] button').click(); await T.tick(350);
T.check('linear flip: height reserved', Math.abs(w.getBoundingClientRect().height - h0) < 1, w.getBoundingClientRect().height);
yb('share').click(); await T.tick(350);
T.check('back to share', !w.querySelector('[data-view="log"]') && yb('share').classList.contains('on'), '');
T.log('widget height'
, w.getBoundingClientRect().height);
T.done();
