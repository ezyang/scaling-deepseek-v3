// @page studies/03-roofline.html
// the scaling ladder: op-class shares of the counted FLOPs up the DSv3 family vs 6ND
const L = await import('/src/ladder.js');
const { PARAMS } = await import('/src/params.js');
const w = document.querySelector('dsv3-ladder');
await T.tick(300);
const ro = () => w.querySelector('.ro').textContent;
const share = (k) => +w.querySelector(`[data-share="${k}"]`).dataset.true;
T.check('family formula reproduces DeepSeek’s 36.6B activated exactly', L.N_DSV3 === PARAMS.activeTotal, L.N_DSV3);
T.check('DSv3 is exactly s = 1', L.ladderPoint(L.LOG_C_DSV3, 4096).s === 1, '');
T.check('resting: DSv3 itself, ×1.14, 6ND misses 12.3%', /DeepSeek-V3 itself/.test(ro()) && /×1\.14/.test(ro()) && /misses 12\.3%/.test(ro()), ro());
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
T.check('S = 8,192: attention share grows', share('attn') > a0 && /S = 8,192/.test(ro()), share('attn'));
T.check('knob flip: height reserved', Math.abs(w.getBoundingClientRect().height - h0) < 1, w.getBoundingClientRect().height);
w.querySelector('[data-knob="seq"] select').value = '131072'; w.querySelector('[data-knob="seq"] select').dispatchEvent(new Event('change')); await T.tick(350);
T.check('S = 131,072: + disabled, attention the majority at DSv3', w.querySelector('[data-knob="seq"] button[data-dir="1"]').disabled && share('attn') > 0.75, share('attn'));
w.querySelector('[data-knob="seq"] select').value = '4096'; w.querySelector('[data-knob="seq"] select').dispatchEvent(new Event('change')); await T.tick(350);
// compute slider: 1e27 → smaller miss, preset unlit; near DSv3 it snaps back on
const r = w.querySelector('input[data-knob="c"]');
r.value = '27'; r.dispatchEvent(new Event('input')); await T.tick(50);
T.check('C = 1e27: preset unlit, 6ND closer', !w.querySelector('[data-knob="preset"] button').classList.contains('on') && share('attn') < a0 && /1\.00×10²⁷/.test(ro()), ro());
r.value = (L.LOG_C_DSV3 + 0.02).toFixed(2); r.dispatchEvent(new Event('input')); await T.tick(50);
T.check('slider snaps onto DSv3', w.querySelector('[data-knob="preset"] button').classList.contains('on') && /DeepSeek-V3 itself/.test(ro()), ro());
// drag on the plot: press at the 1e20 tick
const hit = w.querySelector('rect[data-hit]'), b = hit.getBoundingClientRect();
const x20 = b.left + b.width / 8;
hit.dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, clientX: x20, clientY: b.top + 50, pointerId: 1 }));
w.querySelector('.ld > div:nth-child(2)').dispatchEvent(new PointerEvent('pointerup', { bubbles: true, pointerId: 1 })); await T.tick(50);
T.check('press on the plot moves the cursor (1e20)', Math.abs(+w.querySelector('line[data-cursor]').dataset.cursor - 20) < 0.02, w.querySelector('line[data-cursor]').dataset.cursor);
T.check('URL hash carries the state', /l:ladder=/.test(decodeURIComponent(location.hash)), location.hash);
w.querySelector('[data-knob="preset"] button').click(); await T.tick(350);
T.check('preset returns to DSv3', /DeepSeek-V3 itself/.test(ro()), ro());
T.log('widget height', w.getBoundingClientRect().height);
T.done();
