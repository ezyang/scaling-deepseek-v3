// @page studies/scratch-04.html
// the EP-scaling figure: at DeepSeek-V3 its ratio is 03's sheets' T_EP′ ÷ T
// (the same model, not a re-derivation that could drift); the break-even
// crossing really sits at 1, below DSv3's budget; past the lm-head hump the
// curve only falls, and its tail slope is −1/6 per decade of C
const { epRatio } = await import('/src/epscale.js');
const { ladderPoint, LOG_C_DSV3 } = await import('/src/ladder.js');
const w = document.getElementById('epscale');
const doc03 = new DOMParser().parseFromString(await (await fetch('/studies/03-roofline.html')).text(), 'text/html');
const cell = (nm) => +[...doc03.querySelectorAll('.cellsheet tr')].find((r) => r.querySelector('.nm')?.textContent === nm).querySelector('.vl').textContent.replace(/,/g, '');
const want = cell('TEP′') / cell('T'), got = +w.querySelector('[data-dsv3]').dataset.true;
T.check('DSv3 point = T_EP′ ÷ T', Math.abs(got / want - 1) < 1e-9, `${got} vs ${want}`);
const lc = +w.querySelector('[data-cross]').dataset.true;
T.check('break-even at ratio 1', Math.abs(epRatio(ladderPoint(lc, 4096)) - 1) < 1e-9, lc);
T.check('break-even below DSv3', lc < LOG_C_DSV3, lc);
const u = (x) => epRatio(ladderPoint(x, 4096));
let falls = true; for (let x = 20.5; x < 27; x += 0.25) falls &&= u(x + 0.25) < u(x);
T.check('falls past the hump', falls, '');
const slope = (Math.log10(u(27)) - Math.log10(u(26))) / 1;
T.check('tail slope ≈ −1/6', Math.abs(slope + 1 / 6) < 0.01, slope);
T.done();
