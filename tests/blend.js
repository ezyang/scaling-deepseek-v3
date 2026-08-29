// @page studies/02-hopper-memory.html
// fit-chart layout blend: transitions interpolate the LAYOUT (pixel space),
// so retargets are continuous, one-sided rows fade, and ghosts tween in
const deck = document.getElementById('fitdeck');
const dl = () => deck.querySelector('dsv3-layer');
const totW = () => +dl().querySelector('.lv-bar rect[data-bar="total"]')?.getBoundingClientRect().width;
const next = deck.querySelector('button.deck-next');

// (a) retarget continuity: interrupt a slide tween mid-flight — the new tween
// starts from the on-screen geometry, not a snap to either endpoint
next.click(); await T.tick(150);            // mid-tween of step 1 → 2
const wMid = totW();
next.click(); await T.tick(30);             // retarget to step 3, ~2 frames in
const wAfter = totW();
T.check('retarget starts from on-screen geometry (no jump)',
  Math.abs(wAfter - wMid) < 25, `${wMid.toFixed(1)} → ${wAfter.toFixed(1)}`);
await T.tick(700);

// (b) TWO PHASES: the ghosts plant themselves first (quick fade/slide over
// the opening share) while the bars hold still; then the bars pour
const wBefore = totW();
next.click(); await T.tick(50);             // inside the ghost phase
const gOp = [...dl().querySelectorAll('.lv-bar rect[data-ghost]')].map((g) => +g.getAttribute('opacity'));
T.check('ghosts mid-fade during the ghost phase', gOp.length > 0 && gOp.some((o) => o > 0 && o < 0.7), gOp.join('|'));
T.check('bars hold still until the ghosts have planted', Math.abs(totW() - wBefore) < 1,
  `${wBefore.toFixed(1)} → ${totW().toFixed(1)}`);
await T.tick(250);                          // well into the bar phase
T.check('bars pour after the ghost phase', Math.abs(totW() - wBefore) > 5,
  `${wBefore.toFixed(1)} → ${totW().toFixed(1)}`);
await T.tick(700);
const gOp2 = [...dl().querySelectorAll('.lv-bar rect[data-ghost]')].map((g) => +g.getAttribute('opacity'));
T.check('ghosts land at 0.7', gOp2.every((o) => o === 0.7), gOp2.join('|'));

// (c) accordion close: sub-rows are one-sided rows — they fade + collapse
// into their parent instead of vanishing on the first frame
const layer = () => document.getElementById('local-diagram');
const md = (el2) => el2.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
md(layer().querySelectorAll('.lv-bar g[data-prop]')[0]); await T.tick(700);   // solo weights: breakdown opens
const nSub = layer().querySelectorAll('.lv-bar g[data-part]').length;
T.check('breakdown open under solo', nSub >= 2, nSub);
md(layer().querySelectorAll('.lv-bar g[data-prop]')[0]); await T.tick(60);    // unsolo: close begins
const subMid = [...layer().querySelectorAll('.lv-bar g[data-part]')].map((g) => +g.getAttribute('opacity'));
T.check('sub-rows still present mid-close, fading', subMid.length === nSub && subMid.every((o) => o < 1), subMid.join('|'));
await T.tick(700);
T.check('sub-rows gone once settled', layer().querySelectorAll('.lv-bar g[data-part]').length === 0, '');
T.done();
