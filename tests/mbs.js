// @page studies/02-hopper-memory.html
// 02's final free-play widget: the microbatch-size knob (gbs attribute) and its readouts.
const layer = () => document.getElementById('local-diagram');
const host = () => layer().parentElement;
const val = (id) => parseFloat(layer().querySelector(`.lv-bar text[data-role="val:${id}"]`)?.textContent);
const ro = (k) => host().querySelector(`[data-readout="${k}"]`)?.textContent;
const knob = () => host().querySelector('.stp[data-knob="mbs"]');
const opts = () => [...knob().querySelectorAll('option')].map((o) => o.textContent);
const head = () => Math.round(layer().querySelector('.lv-head').getBoundingClientRect().height);
const pick = async (k, v) => { [...host().querySelector(`.stp[data-knob="${k}"]`).querySelectorAll('button')].find((b) => b.textContent === v).click(); await T.tick(700); };
host().scrollIntoView(); await T.tick(100);

T.check('default mbs 1 (published numbers)', layer().mbs === 1, layer().mbs);
T.check('PP8: 60 microbatches per step', ro('mbs') === '× 60 per step', ro('mbs'));
T.check('PP8 choices keep ≥ 2·PP microbatches', opts().join() === '1 seq,2 seq', opts().join());
const acts1 = val('3'), h0 = head();
knob().querySelectorAll('button')[1].click(); await T.tick(700);
T.check('mbs 2 doubles the stash', Math.abs(val('3') / acts1 - 2) < 0.01, `${acts1} → ${val('3')}`);
T.check('mbs 2: 30 per step', ro('mbs') === '× 30 per step', ro('mbs'));
T.check('mbs in URL', decodeURIComponent(location.hash).includes('"mbs":2'), 'mbs:2');
for (const z of ['off', '1', '2']) { await pick('zero', z); T.check(`head height fixed at zero ${z}`, head() === h0, `${head()} vs ${h0}`); }
await pick('zero', '3');
T.check('head height fixed at zero 3', head() === h0, `${head()} vs ${h0}`);
T.check('ZeRO-3 readout', ro('zero').includes('weights gathered 60× per step') && ro('zero').includes('gradients reduce-scattered 30× per step'), ro('zero'));
await pick('zero', '1');
T.check('ZeRO-1 readout', ro('zero').includes('weights resident') && ro('zero').includes('synced once per step'), ro('zero'));
host().querySelector('.stp[data-knob="pp"]').querySelectorAll('button')[0].click(); await T.tick(700);
T.check('pp1 choices end at the whole local batch', opts().join() === '1 seq,2 seq,4 seq,7.5 seq', opts().join());
const sel = knob().querySelector('select'); sel.value = '7.5'; sel.dispatchEvent(new Event('change')); await T.tick(700);
T.check('whole batch: one microbatch', ro('mbs') === '× 1 per step', ro('mbs'));
T.check('whole batch stash 291.7 GiB', Math.abs(val('3') - 291.7) < 0.1, val('3'));
host().querySelector('.stp[data-knob="pp"]').querySelectorAll('button')[1].click(); await T.tick(700);
T.check('pp back up clamps mbs into the schedule', [1, 2].includes(layer().mbs), layer().mbs);
T.done();
