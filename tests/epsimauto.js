// @page studies/03-roofline.html
// the EP sim's autoplay: on a fresh load the first full view samples from token
// 1 to the resting 1,000 in ~3 s — the tallies move nearly every frame and the
// bars ease, while token 1 stays drawn as the example routing (blue experts,
// black IB arrows, header, the histogram's blue bin) and crossfades to token
// 1,000 at the end, so nothing token-specific flickers mid-run; 1,000 tokens is the
// default, so the run leaves the hash clean and the page's ↺ dot off; it plays
// once per load (reset is by hand), "redo animation" replays it, and any knob
// interrupts it
const { simulate } = await import('/src/epsim.js');
const w = document.getElementById('epsim');
const q = (s) => w.querySelector(s), qa = (s) => [...w.querySelectorAll(s)];
const token = () => +q('[data-token]').dataset.token, routed = () => +q('[data-routed]').dataset.routed;
const dot = () => document.querySelector('.resetb')?.classList.contains('mod');
// headless Chrome under virtual time runs no rendering frames, so the IntersectionObserver
// reports scrolls late or never: unhook it, scroll for real, and hand the widget the ratio it'd see
w._io.disconnect();
const away = () => { scrollTo(0, 0); w._view(0); }, into = () => { scrollTo(0, w.getBoundingClientRect().top + scrollY - 40); w._view(1); };
const BLUE = '#2a78d6', ink = () => {   // token-specific ink on screen, by opacity
  const hdr = q('[data-ink]') ? +q('[data-ink]').getAttribute('opacity') : 1;
  const left = qa('svg > g[opacity]').reduce((m, g) => Math.max(m, +g.getAttribute('opacity')), 0);
  const bin = qa('g[data-seek] rect[opacity]').filter((r) => r.getAttribute('fill') === BLUE).reduce((m, r) => Math.max(m, +r.getAttribute('opacity')), 0);
  const label = qa('g[data-seek] text[style]').length;
  return { hdr, left, bin, label };
};
T.check('fresh load: out of view, at token 1, hash clean', w.getBoundingClientRect().top > innerHeight && token() === 1 && location.hash === '' && !dot(), w.getBoundingClientRect().top);
await T.tick(200);
T.check('no autoplay while out of view', routed() === 1, routed());

into();
const t0 = performance.now(), seen = new Set(), swaps = [], drawn = () => qa('rect[data-bar]').map((r) => +r.getAttribute('height') / 62);
let last = token(), prev = drawn(), jump = 0, inkMid = 1, tokMid = new Set(), dotMid = false;
while (performance.now() - t0 < 4000) {
  await T.tick(16);
  seen.add(routed());
  const now = drawn(); jump = Math.max(jump, ...now.map((v, k) => Math.abs(v - prev[k]))); prev = now;
  const el = performance.now() - t0;
  if (el > 400 && el < 2700) { const i = ink(); inkMid = Math.min(inkMid, i.hdr, i.left, i.bin, i.label / 2); tokMid.add(token()); }
  dotMid ||= dot();
  if (token() !== last) { swaps.push(el); last = token(); }
}
T.check('the bars ease: no share moves over 0.25 between frames (token 2 alone swings 0.5)', jump < 0.25, jump.toFixed(3));
T.check('… and settle on the exact shares', qa('rect[data-bar]').every((r) => Math.abs(+r.getAttribute('height') - (+r.dataset.share * 62)) < 0.06), '');
T.check('the tallies glide: >100 distinct counts on the way', seen.size > 100, seen.size);
T.check('mid-run: token 1 fully inked throughout (left, header, blue bin, blue labels)', inkMid === 1 && tokMid.size === 1 && tokMid.has(1), `${inkMid} ${[...tokMid]}`);
T.check('the header switches once, at the end', swaps.length === 1 && swaps[0] > 2500, swaps.map(Math.round).join());
const ref = simulate(1000);
T.check('lands on token 1,000, tallies match simulate()', token() === 1000 && routed() === 1000 && +q('[data-mean=ib]').dataset.true === ref.ib / ref.n, `${token()} ${routed()}`);
const i1 = ink();
T.check('token 1,000\'s ink fully in', i1.hdr === 1 && i1.left === 1 && i1.bin === 1 && qa('svg > g[opacity]').length === 1 && qa('text[data-cur]').length === 2, JSON.stringify(i1));
T.check('the run leaves the hash clean and the ↺ dot off, throughout', location.hash === '' && !dot() && !dotMid, location.hash);

away(); await T.tick(100); into(); await T.tick(800);
T.check('another full view does not replay it', token() === 1000 && routed() === 1000, routed());
q('[data-knob="step"] [data-v="reset"]').click(); await T.tick(400);
away(); await T.tick(100); into(); await T.tick(1500);
T.check('reset: token 1, by hand (no replay on the next full view); a departure, so the dot', token() === 1 && routed() === 1 && dot(), `${routed()} ${location.hash}`);

q('[data-knob="anim"] [data-v="redo"]').click(); await T.tick(1500);
const mid = routed();
T.check('redo animation replays it, from token 1, which it shows', mid > 1 && mid < 1000 && token() === 1 && ink().left === 1 && qa('svg > g[opacity]').length === 1, mid);
await T.tick(2500);
T.check('… to the resting 1,000, hash clean again', routed() === 1000 && token() === 1000 && location.hash === '' && !dot(), `${routed()} ${location.hash}`);
q('[data-knob="anim"] [data-v="redo"]').click(); await T.tick(1500);
const mid2 = routed();
q('[data-knob="step"] [data-v="1"]').click(); await T.tick(1500);
T.check('a knob interrupts it: +1 from where it was, then stays, in the hash', routed() === mid2 + 1 && token() === mid2 + 1 && location.hash.includes(String(mid2 + 1)), `${mid2} → ${routed()}`);
T.check('… its ink eases back in from nothing', ink().left === 1 && ink().hdr === 1, JSON.stringify(ink()));
T.done();
