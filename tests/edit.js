// @page studies/02-hopper-memory.html
const layer = () => document.getElementById('local-diagram');
const sheet = document.querySelector('dsv3-sheet');
await T.tick(400);
const trOf = (id) => [...sheet.querySelectorAll('tr')].find(r => r.querySelector('.nm')?.textContent === id);
const rowOf = (id) => trOf(id)?.textContent ?? '';
const btn = (id, ed) => trOf(id)?.querySelector(`.sbtn[data-ed="${ed}"]`);
const valOf = (id) => trOf(id)?.querySelector('td.vl')?.textContent.trim();
const total = () => +layer().querySelector('.lv-bar text[data-role="val:total"]').dataset.true;
const t0 = total();
// stepper: EP down halves it — the widget's own stepper moves
btn('P3', 'dn').click(); await T.tick(700);
T.check('P3 − steps EP 64 → 32 (linked: chart repriced)', valOf('P3') === '32' && total() !== t0,
  `${rowOf('P3').slice(0, 40)} total ${total()}`);
T.check('the widget stepper shows 32 too', layer().parentElement.querySelector('.stp[data-knob="ep"] select.v').value === '32', '');
btn('P3', 'up').click(); await T.tick(700);
T.check('and back (total restores exactly)', total() === t0, total());
// seg: ZeRO down to off
btn('Z1', 'dn').click(); await T.tick(700);
T.check('Z1 − steps ZeRO 1 → off', valOf('Z1') === '0' && total() > t0, valOf('Z1'));
btn('Z1', 'up').click(); await T.tick(700);
// toggle: F1
btn('F1', 'tg').click(); await T.tick(700);
T.check('F1 ⇄ flips the e4m3+ᵀ params checkbox', layer().parentElement.querySelector('input[data-knob="fp8params"]').checked
  && valOf('F1') === '1', valOf('F1'));
btn('F1', 'tg').click(); await T.tick(400);
// precision: B8 (dispatched) toggles via the diagram's ffn gate/up dtype button
btn('B8', 'tg').click(); await T.tick(700);
T.check('B8 ⇄ flips dispatched to bf16 (2 B/elem) via the gate/up dtype button',
  rowOf('B8').includes('2 B/elem') && layer().querySelector('button[data-dt="ffn_gate_up"]').textContent === 'bf16', rowOf('B8'));
btn('B8', 'tg').click(); await T.tick(400);
T.check('and back to 1.03125', rowOf('B8').includes('1.03125'), rowOf('B8'));
// kept?: R8 flips the dispatch mark (policy goes custom; preset restores)
btn('R8', 'tg').click(); await T.tick(700);
T.check('R8 ⇄ replays dispatched tokens (kept? → 0)', valOf('R8') === '0', valOf('R8'));
const rseg = layer().parentElement.querySelector('.stp[data-knob="recompute"]');
[...rseg.querySelectorAll('button')].find(b => b.textContent === 'dsv3').click(); await T.tick(700);
T.check('the dsv3 preset restores (total back to start)', total() === t0 && valOf('R8') === '1', total());
T.done();
