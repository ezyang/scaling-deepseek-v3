// @page studies/03-roofline.html
// the EP routing sim: the drawn copies match the token's tally (one IB arrow
// per remote node hosting an expert, one NVLink arc per other GPU on a node
// that needs it), the tallies match the node-importable simulate(), a token
// only ever costs 3 or 4 IB copies under the sheet's routing, the experts
// don't split evenly across nodes, both means settle on the exact ones;
// DeepSeek-V3's routing replays the same token count and settles below 3.5
// (some tokens touch only 2 remote nodes); clicking a histogram column
// routes tokens until one lands in it and says how many that took (the sheet
// routing's 0/1/2 are impossible; 0 under DeepSeek-V3 is ~1 in 14M, past the
// cap); state rides the hash
const { simulate, copies, route, EXACT } = await import('/src/epsim.js');
const w = document.getElementById('epsim');
const q = (s) => w.querySelector(s), qa = (s) => [...w.querySelectorAll(s)];
const btn = (knob, v) => q(`[data-knob=${knob}] [data-v="${v}"]`);
const drawn = () => {
  const ib = +q('[data-cur-ib]').dataset.curIb, nv = +q('[data-cur-nv]').dataset.curNv;
  return { ib, nv, ibPaths: qa('path[data-ib]').length, nvPaths: qa('path[data-nv]').length, experts: qa('rect[data-expert]').map((r) => +r.dataset.expert) };
};
let d = drawn();
T.check('resting: token 1, sheet routing', q('[data-token]').textContent === 'token 1' && btn('mode', 'sheet').classList.contains('on'), q('[data-token]').textContent);
const click = (k, sel = '.hit') => q(`[data-seek="${k}"] ${sel}`).dispatchEvent(new MouseEvent('click', { bubbles: true }));
const ibOf = (t, m) => copies(route(t, m)).ib;
const msg = () => q('[data-seek-msg]')?.textContent ?? '';
T.check('token 1: its 8 experts drawn', d.experts.join() === route(1).join(), d.experts.join());
T.check('one IB arrow per remote node', d.ibPaths === d.ib && d.ib === copies(route(1)).ib, `${d.ibPaths} vs ${d.ib}`);
T.check('one NVLink arc per copy', d.nvPaths === d.nv, `${d.nvPaths} vs ${d.nv}`);
T.check('reset disabled at 1 token', btn('step', 'reset').disabled, '');

click(1);
await T.tick(50);
T.check('sheet routing: seeking 1 IB copy is impossible, nothing routed', /impossible/.test(msg()) && q('[data-token]').textContent === 'token 1', msg());
click(4, 'rect[data-bar]');
await T.tick(400);
let want = 2; while (ibOf(want, 'sheet') !== 4) want++;
T.check('seek 4: routes to the first token with 4 IB copies', q('[data-token]').textContent === `token ${want}` && +q('[data-cur-ib]').dataset.curIb === 4, `${q('[data-token]').textContent} vs ${want}; ${msg()}`);
T.check('seek: the count is said', msg() === (want === 2 ? 'the next token had 4 IB copies' : `sampled ${want - 1} tokens until one had 4 IB copies`), msg());
btn('step', 'reset').click();
await T.tick(400);
T.check('a step clears the seek note', msg() === '', msg());

btn('step', 10000).click();
await T.tick(400);
const ref = simulate(10001), mean = +q('[data-mean=ib]').dataset.true;
T.check('+10,000: tallies match simulate()', mean === ref.ib / ref.n && +q('[data-mean=nv]').dataset.true === ref.nv / ref.n, `${mean}`);
T.check('only 3 or 4 IB copies', qa('rect[data-bar]').every((r) => (+r.dataset.bar >= 3) === (+r.dataset.share > 0)), qa('rect[data-bar]').map((r) => r.dataset.share).join());
T.check('IB mean near 3.5', Math.abs(mean - EXACT.ib) < 0.01, mean);
const nvMean = +q('[data-mean=nv]').dataset.true;
T.check('NVLink mean near the exact 6.528', Math.abs(nvMean - EXACT.nv) < 0.02, nvMean);
const uneven = Array.from({ length: 100 }, (_, t) => route(t + 1)).filter((ex) => ex.some((e) => ex.filter((f) => f >> 5 === e >> 5).length !== 2)).length;
T.check('experts split unevenly across nodes (not always 2-2-2-2)', uneven > 50, uneven);
d = drawn();
T.check("this token's bin highlighted", qa('text[data-cur]').length === 2 && qa('text[data-cur]').every((t) => t.parentNode && t.textContent.length) && q('text[data-cur]:not(.num)').textContent === String(d.ib), qa('text[data-cur]').map((t) => t.textContent).join());
T.check('the last token drawn, arrows match', d.experts.join() === route(10001).join() && d.ibPaths === d.ib && d.nvPaths === d.nv, d.experts.join());
T.check('the fade finished (one overlay, opaque)', qa('svg > g[opacity]').length === 1 && q('svg > g[opacity]').getAttribute('opacity') === '1.000', qa('svg > g[opacity]').map((g) => g.getAttribute('opacity')).join());
T.check('state in the hash', /e%3Aepsim=.*10001.*sheet/.test(location.hash), location.hash);

btn('mode', 'dsv3').click();
await T.tick(400);
const ref2 = simulate(10001, 'dsv3'), mean2 = +q('[data-mean=ib]').dataset.true;
T.check('DeepSeek-V3 routing: same token count replayed', q('[data-token]').textContent === 'token 10,001' && mean2 === ref2.ib / ref2.n, `${mean2}`);
T.check('DeepSeek-V3 routing: settles below 3.5', mean2 < 3.49 && mean2 > 3.4, mean2);
T.check('DeepSeek-V3 routing: some tokens touch only 2 remote nodes', +q('rect[data-bar="2"]').dataset.share > 0, q('rect[data-bar="2"]').dataset.share);
d = drawn();
T.check('DeepSeek-V3 routing: its last token drawn', d.experts.join() === route(10001, 'dsv3').join() && d.ibPaths === d.ib && d.nvPaths === d.nv, d.experts.join());

click(1);
await T.tick(400);
let at = 10002; while (ibOf(at, 'dsv3') !== 1) at++;
const ref3 = simulate(at, 'dsv3');
T.check('DeepSeek-V3 routing: seek 1 finds the next 1-copy token', q('[data-token]').textContent === `token ${at.toLocaleString('en-US')}` && +q('[data-cur-ib]').dataset.curIb === 1 && +q('[data-mean=ib]').dataset.true === ref3.ib / ref3.n, `${q('[data-token]').textContent} vs ${at}`);
T.check('DeepSeek-V3 routing: says how many it took', msg() === `sampled ${(at - 10001).toLocaleString('en-US')} tokens until one had 1 IB copy`, msg());
T.check("the seek note fits the widget", q('[data-seek-msg]').getBBox().x + q('[data-seek-msg]').getBBox().width < 738, q('[data-seek-msg]').getBBox().width);
click(0, 'text');
await T.tick(400);
T.check('DeepSeek-V3 routing: seek 0 runs to the cap and says so', q('[data-token]').textContent === 'token 100,000' && /no 0 in the next .* tokens: the sim stops at 100K/.test(msg()), `${q('[data-token]').textContent}; ${msg()}`);

btn('step', 'reset').click();
await T.tick(400);
T.check('reset: back to token 1, routing kept', q('[data-token]').textContent === 'token 1' && btn('mode', 'dsv3').classList.contains('on') && drawn().experts.join() === route(1, 'dsv3').join(), q('[data-token]').textContent);
T.done();
