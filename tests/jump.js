// @page studies/02-hopper-memory.html
const layer = () => document.getElementById('local-diagram');
const sheet = document.querySelector('dsv3-sheet');
await T.tick(400);
const trOf = (id) => [...sheet.querySelectorAll('tr')].find(r => r.querySelector('.nm')?.textContent === id);
// model input → controlling knob
trOf('P3').querySelector('td.nm').click(); await T.tick(100);
const ep = layer().parentElement.querySelector('.stp[data-knob="ep"]');
T.check('P3 jumps to the EP stepper (highlighted)', ep.classList.contains('jump-hl'), ep.className);
// activation → its chip
trOf('A8').querySelector('td.nm').click(); await T.tick(100);
T.check('A8 jumps to the dispatched-tokens chip', layer().querySelector('g[data-chip="dispatch"]')?.classList.contains('jump-hl'), '');
// the split-out lse → the aux label
trOf('A6b').querySelector('td.nm').click(); await T.tick(100);
T.check('lse row jumps to the aux label', layer().querySelector('g[data-chip="attn:aux"]')?.classList.contains('jump-hl'), '');
// S inputs → the ZeRO segment
trOf('S6').querySelector('td.nm').click(); await T.tick(100);
T.check('S6 jumps to the ZeRO segment', layer().parentElement.querySelector('.stp[data-knob="zero"]')?.classList.contains('jump-hl'), '');
// consistent sizes: redo chips report the would-be bytes
const dia = layer().querySelector('.lv-scroll svg');
const redo = [...dia.querySelectorAll('text.tredo')].find(t => t.textContent.includes('norm1 out'));
T.check('redo chips report their would-be size', /\(\d[\d.]* [GMK]iB\)/.test(redo?.textContent ?? ''), redo?.textContent);
// tooltip split: tip on the name tspan; the byte value carries raw only
const sv = [...dia.querySelectorAll('text.tsave')].find(t => t.querySelector('tspan[data-tip]') && t.querySelector('tspan[data-raw]'));
const nameTs = sv.querySelector('tspan[data-tip]'), rawTs = sv.querySelector('tspan[data-raw]');
T.check('saved tip is brief and name-scoped', nameTs.dataset.tip.includes('kept alive by')
  && !nameTs.dataset.tip.includes('Direction:') && !rawTs.closest('[data-tip]'), nameTs.dataset.tip.slice(0, 50));
T.done();
