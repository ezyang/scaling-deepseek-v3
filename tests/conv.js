// @page studies/02-hopper-memory.html
// mark-flip tween: the chip grid CONVERTS square-by-square (no ghost flash)
const ac = document.querySelector('dsv3-anatomy[controls="marks"]');
const dia = () => ac.querySelector('.lv-scroll svg');
// swiglu is recomputed under dsv3: its chip shows a hollow counterfactual
// grid. Flip it to save and inspect mid-tween.
const mb = () => dia().querySelector('button[data-mark="swiglu"]');
T.check('swiglu mark button present (state ↻)', mb()?.textContent === '↻', mb()?.textContent);
// count the amber squares near the swiglu chip: solid (fill) vs hollow (stroke)
const counts = () => {
  const solid = [...dia().querySelectorAll('rect[fill="#eda100"][width="5"][height="5"]')].length;
  const hollow = [...dia().querySelectorAll('rect[stroke="#eda100"][width="4.2"]')].length;
  return { solid, hollow };
};
const before = counts();
T.log('before', JSON.stringify(before));
mb().click(); await T.tick(90);   // mid-tween (12 frames ≈ 200 ms)
const mid = counts();
T.log('mid', JSON.stringify(mid));
await T.tick(500);
const after = counts();
T.log('after', JSON.stringify(after));
// mid-tween: a MIX (some converted, some still hollow), and no doubled grid
T.check('mid-tween shows a partial conversion (solid grew, hollow shrank, both present)',
  mid.solid > before.solid && mid.hollow < before.hollow && mid.solid < after.solid,
  JSON.stringify({ before, mid, after }));
T.check('no doubled squares mid-tween (converted in place, not crossfaded)',
  mid.solid + mid.hollow <= before.solid + before.hollow + 1,
  `${mid.solid + mid.hollow} vs ${before.solid + before.hollow}`);
T.check('flip completes: all solid gain, hollows gone from this chip',
  after.solid > before.solid && after.hollow < before.hollow, JSON.stringify(after));
// and back: solid converts back to hollow through a mixed state
mb().click(); await T.tick(90);
const mid2 = counts();
T.check('reverse flip also converts through a mix', mid2.solid < after.solid && mid2.hollow > after.hollow,
  JSON.stringify(mid2));
await T.tick(500);
T.done();
