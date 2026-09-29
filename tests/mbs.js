// @page studies/02-hopper-memory.html
// 02's final free-play widget: the microbatches-per-step knob (gbs attribute) and its readouts.
const layer = () => document.getElementById('local-diagram');
const host = () => layer().parentElement;
const val = (id) => parseFloat(layer().querySelector(`.lv-bar text[data-role="val:${id}"]`)?.textContent);
const ro = (k) => host().querySelector(`[data-readout="${k}"]`)?.textContent;
const knob = (k = 'mb') => host().querySelector(`.stp[data-knob="${k}"]`);
const opts = () => [...knob().querySelectorAll('option')].map((o) => o.textContent).join();
const head = () => Math.round(layer().querySelector('.lv-head').getBoundingClientRect().height);
const pick = async (k, v) => { [...knob(k).querySelectorAll('button')].find((b) => b.textContent === v).click(); await T.tick(700); };
const choose = async (k, v) => { const s = knob(k).querySelector('select'); s.value = String(v); s.dispatchEvent(new Event('change')); await T.tick(700); };
const sheetRow = (id) => document.querySelector(`dsv3-sheet[layer="local-diagram"] tr[data-cell="${id}"]`)?.textContent ?? '';
host().scrollIntoView(); await T.tick(100);

// resting state = the published PP8 numbers: one sequence per microbatch
T.check('PP8 rests at m = 60 (one sequence each)', layer().mb === 60 && ro('mb') === 'of 1 seq', `${layer().mb} ${ro('mb')}`);
T.check('PP8 choices: divisors of 60 with m ≥ 2·PP', opts() === '20,30,60', opts());
T.check('sheet carries m and the busiest GPU’s batch', sheetRow('P12').includes('60') && sheetRow('P14').includes('60'), sheetRow('P14').slice(0, 80));
const acts1 = val('3'), h0 = head();
knob().querySelectorAll('button')[0].click(); await T.tick(700);
T.check('m 30: two sequences each, stash doubles', layer().mb === 30 && ro('mb') === 'of 2 seq' && Math.abs(val('3') / acts1 - 2) < 0.01, `${acts1} → ${val('3')}`);
T.check('m in URL', decodeURIComponent(location.hash).includes('"mb":30'), 'mb:30');
for (const z of ['off', '1', '2', '3']) { await pick('zero', z); T.check(`head height fixed at zero ${z}`, head() === h0, `${head()} vs ${h0}`); }
T.check('ZeRO-3 readout', ro('zero').includes('weights gathered 60× per step') && ro('zero').includes('gradients reduce-scattered 30× per step'), ro('zero'));
await pick('zero', '1');
T.check('ZeRO-1 readout', ro('zero').includes('weights resident') && ro('zero').includes('synced once per step'), ro('zero'));
// sticky: a cluster resize that keeps m valid keeps it
await choose('gpus', 4096);
T.check('4096 GPUs: m 30 sticks (30 each)', layer().mb === 30 && opts() === '30' && ro('mb') === 'of 1 seq', `${layer().mb} ${opts()}`);
await choose('gpus', 2048);
T.check('back to 2048: m 30 still sticks', layer().mb === 30 && ro('mb') === 'of 2 seq', layer().mb);
// PP re-defaults: no pipeline = no microbatching; the busiest GPU carries ⌈15360 ÷ 2048⌉ = 8
await choose('pp', 1);
T.check('pp1: m re-defaults to 1 (8 seq)', layer().mb === 1 && ro('mb') === 'of 8 seq', `${layer().mb} ${ro('mb')}`);
T.check('pp1 choices divide 8', opts() === '1,2,4,8', opts());
T.check('pp1 whole batch stash = 8 × one sequence', Math.abs(val('3') - 8 * 38.893) < 0.2, val('3'));
await choose('mb', 4);
await pick('zero', '3');
T.check('m sticks across a ZeRO flip', layer().mb === 4 && ro('mb') === 'of 2 seq', layer().mb);
// tween frames patch the diagram svg in place: the patched result must equal a from-scratch render
const svgTxt = () => layer().querySelector('.lv-scroll svg').outerHTML.replace(/<!---->/g, '');
const patched = svgTxt(); layer().render();
T.check('patched svg == full rebuild', svgTxt() === patched, `${patched.length} vs ${svgTxt().length}`);
await choose('pp', 8);
T.check('PP8 again: re-defaults to one sequence each', layer().mb === 60, layer().mb);
[...host().querySelectorAll('button')].find((b) => b.textContent === 'reset all').click(); await T.tick(700);
T.check('reset all: pp1, no microbatching', layer().pp === 1 && layer().mb === 1, `${layer().pp} ${layer().mb}`);
T.done();
