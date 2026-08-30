// @page studies/scratch-fold.html
// the V-fold stage map: 16 virtual chunks ↔ 8 physical ranks (animated),
// exact params, minimap highlighting — rank 0 lights BOTH ends of the model
const f = document.querySelector('dsv3-pp-fold');
const segs = () => [...f.querySelectorAll('g[data-chunk]')];
const mm = () => f.querySelectorAll('rect[data-layer]');
const hot = () => [...mm()].filter((r) => r.getAttribute('fill') === '#eda100').map((r) => +r.dataset.layer);

T.check('16 chunk segments drawn', segs().length === 16, segs().length);
T.check('61 minimap layer cells', mm().length === 61, mm().length);
// exactness: chunk params partition the EP64-local total — and the two
// views claim the same masses (rank totals = the pairwise sums)
const p = segs().map((g) => +g.dataset.params);
const moeTotal = p.reduce((a, b) => a + b, 0);
const rk = [];
for (const t2 of f.querySelectorAll('text[data-ranktotal]')) rk[+t2.dataset.ranktotal] = +t2.dataset.params;
T.check('rank totals = pairwise chunk sums (fold conserves mass)',
  rk.length === 8 && rk.every((v, r) => v === p[r] + p[15 - r]), '');
T.check('rank 0 is the heaviest (emb + head both land there)',
  rk[0] === Math.max(...rk), rk.map((v) => (v / 1e9).toFixed(2)).join('|'));
// hover chunk 0 (virtual view): emb cap + its layers light, head does not
segs()[0].dispatchEvent(new MouseEvent('mouseover', { bubbles: true })); await T.tick(50);
T.check('virtual hover: chunk 0 lights its own range only', hot().length > 0 && Math.max(...hot()) < 8
  && f.querySelector('.ro').textContent.includes('v0'), f.querySelector('.ro').textContent);
// fold it (ONE cycle button): the animation runs, height is reserved
const h0 = f.getBoundingClientRect().height;
const btn = f.querySelector('.top button');
T.check('single cycle button, labeled with the action', f.querySelectorAll('.top button').length === 1
  && btn.textContent.includes('fold'), btn.textContent);
btn.click();
await T.tick(150);   // mid-fold
T.check('mid-fold: height reserved (no reflow)', Math.abs(f.getBoundingClientRect().height - h0) < 1, '');
await T.tick(400);   // settled
// hover rank 0: BOTH ends of the model light up — the fold's signature
segs()[15].dispatchEvent(new MouseEvent('mouseover', { bubbles: true })); await T.tick(50);
const hs = hot();
T.check('folded hover on s0: both ends light (emb-side layers AND tail layers)',
  hs.some((l) => l < 4) && hs.some((l) => l > 56)
  && f.querySelector('.ro').textContent.includes('s0 = v0 + v15'), f.querySelector('.ro').textContent);
T.check('readout names the heaviest rank', f.querySelector('.ro').textContent.includes('heaviest'), '');
T.check('button relabels for the way back', btn.textContent.includes('unroll'), btn.textContent);
// folded: up-pass bars POP onto their partner's row — same y, docked after
const barY = (c) => +f.querySelectorAll('g[data-chunk] rect')[0].ownerSVGElement
  .querySelector(`g[data-chunk="${c}"] rect`).getAttribute('y');
T.check('v15 docks on row 0 (no re-pitching)', barY(15) === barY(0), `${barY(15)} vs ${barY(0)}`);
T.check('vacated rows keep their spans (all 16 span rows drawn)',
  f.querySelectorAll('rect[data-layer]').length === 61, '');
// whole-ROW hitboxes: hovering row 12's empty band highlights rank 3
f.querySelector('rect[data-row="12"]').dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
await T.tick(50);
T.check('row hitbox hover: row 12 lights rank 3 (v3 + v12)',
  f.querySelector('.ro').textContent.includes('s3 = v3 + v12'), f.querySelector('.ro').textContent);
// the second instance opens folded with EP1: experts whole, bars near-uniform
const f2 = document.querySelectorAll('dsv3-pp-fold')[1];
const p2 = [...f2.querySelectorAll('g[data-chunk]')].map((g) => +g.dataset.params);
T.check('EP1 instance holds whole experts (chunks ≫ EP64 chunks)', p2[4] > p[4] * 10, `${p2[4]} vs ${p[4]}`);
T.done();
