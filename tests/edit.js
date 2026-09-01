// @page studies/02-hopper-memory.html
const layer = () => document.getElementById('local-diagram');
const sheet = document.querySelector('dsv3-sheet');
await T.tick(400);
const trOf = (id) => [...sheet.querySelectorAll('tr')].find(r => r.querySelector('.nm')?.textContent === id);
const rowOf = (id) => trOf(id)?.textContent ?? '';
const md = (el2) => el2.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
const step = (id, dir) => md(trOf(id).querySelector(`.sb.${dir}`));
const tgl = (id) => md(trOf(id).querySelector('td.vl.tg'));
const valOf = (id) => trOf(id)?.querySelector('td.vl')?.textContent.trim();
const total = () => +layer().querySelector('.lv-bar text[data-role="val:total"]').dataset.true;
const t0 = total();
// stepper: EP down halves it — the widget's own stepper moves
step('P3', 'dn'); await T.tick(700);
T.check('P3 − steps EP 64 → 32 (linked: chart repriced)', valOf('P3') === '32' && total() !== t0,
  `${rowOf('P3').slice(0, 40)} total ${total()}`);
T.check('the widget stepper shows 32 too', layer().parentElement.querySelector('.stp[data-knob="ep"] select.v').value === '32', '');
step('P3', 'up'); await T.tick(700);
T.check('and back (total restores exactly)', total() === t0, total());
// seg: ZeRO down to off
step('S1', 'dn'); await T.tick(700);
T.check('S1 − steps ZeRO 1 → off', valOf('S1') === '0' && total() > t0, valOf('S1'));
step('S1', 'up'); await T.tick(700);
// toggle: F1
tgl('F1'); await T.tick(700);
T.check('F1 ⇄ flips the e4m3+ᵀ params checkbox', layer().parentElement.querySelector('input[data-knob="fp8params"]').checked
  && valOf('F1') === '1', valOf('F1'));
tgl('F1'); await T.tick(400);
// precision: B8 (dispatched) toggles via the diagram's ffn gate/up dtype button
tgl('B8'); await T.tick(700);
T.check('B8 ⇄ flips dispatched to bf16 (2 B/elem) via the gate/up dtype button',
  rowOf('B8').includes('2 B/elem') && layer().querySelector('button[data-dt="ffn_gate_up"]').textContent === 'bf16', rowOf('B8'));
tgl('B8'); await T.tick(400);
T.check('and back to 1.03125', rowOf('B8').includes('1.03125'), rowOf('B8'));
// kept?: R8 flips the dispatch mark (policy goes custom; preset restores)
tgl('R8'); await T.tick(700);
T.check('R8 ⇄ replays dispatched tokens (kept? → 0)', valOf('R8') === '0', valOf('R8'));
const rseg = layer().parentElement.querySelector('.stp[data-knob="recompute"]');
[...rseg.querySelectorAll('button')].find(b => b.textContent === 'dsv3').click(); await T.tick(700);
T.check('the dsv3 preset restores (total back to start)', total() === t0 && valOf('R8') === '1', total());
// the ± glyphs are pseudo-content: copy/paste never sees them
T.check('stepper cell copies clean (no ± in textContent)',
  !/[−+]/.test(trOf('P3').querySelector('td.vl').textContent), trOf('P3').querySelector('td.vl').textContent);
// RAPID clicks all land (mousedown-driven; re-query per press — the sheet
// resyncs between them)
step('P3', 'dn'); step('P3', 'dn'); step('P3', 'dn'); await T.tick(900);
T.check('three fast − presses = three halvings (EP 64 → 8)', valOf('P3') === '8', valOf('P3'));
step('P3', 'up'); step('P3', 'up'); step('P3', 'up'); await T.tick(900);
T.check('three fast + presses restore (total exact)', valOf('P3') === '64' && total() === t0, `${valOf('P3')} ${total()}`);
T.done();
