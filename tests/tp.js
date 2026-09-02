// @page studies/03-blackwell-memory.html
// tensor parallelism in the local model (Megatron's layout: SP on, expert-TP 1):
// the TP stepper, DP = GPUs/PP/TP, expert-DP unchanged, sharded vs replicated
// parameter classes, every stash ÷ TP — checked against the cells
const l = () => document.getElementById('local-diagram');
const knob = (k) => l().parentElement.querySelector(`.stp[data-knob="${k}"]`);
const txt = () => [...l().querySelectorAll('.lv-bar text')].map((t) => t.textContent).join('|');
await T.tick(600);
T.check('TP stepper offers 1/2/4/8 and opens at 1', knob('tp') && [...knob('tp').querySelectorAll('option')].map((o) => o.value).join() === '1,2,4,8' && l().tp === 1, '');
const c1 = l()._cells();
const before = { T1: c1.get('T1'), W1: c1.get('W1'), G1: c1.get('G1'), O1: c1.get('O1'), A1: c1.get('A1'), P4: c1.get('P4'), P5: c1.get('P5') };
const meshTxt = () => l().parentElement.querySelector('.stp[data-knob="tp"]').parentElement.textContent;
T.check('mesh row reads TP 1 × DP 128 at PP2 on 256 GPUs', meshTxt().includes('DP 128'), meshTxt());
// TP 2
knob('tp').querySelector('button:last-child').click(); await T.tick(500);
const c2 = l()._cells();
T.check('TP2: P11 = 2, DP 64, expert-DP still 4, token share 2048', c2.get('P11') === 2 && c2.get('P4') === 64 && c2.get('P5') === 4 && c2.get('P7') === 2048 && meshTxt().includes('DP 64'), `${c2.get('P4')} ${c2.get('P5')} ${c2.get('P7')}`);
T.check('TP2: activations exactly halve', Math.abs(c2.get('A1') * 2 - before.A1) < 1e-6, `${c2.get('A1')} vs ${before.A1}`);
// weights: sharded parts halve, replicated parts and experts do not
const N5 = c2.get('N5'), N6 = c2.get('N6'), N7 = c2.get('N7'), N8 = c2.get('N8'), L1 = c2.get('L1'), L2 = c2.get('L2');
T.check('TP2: Q2 = L1 × (N5/2 + N6) + L2 × (N7/2 + N8)', Math.abs(c2.get('Q2') - (L1 * (N5 / 2 + N6) + L2 * (N7 / 2 + N8))) < 1e-6, String(c2.get('Q2')));
T.check('TP2: expert params per GPU unchanged (EP’s business), experts’ optimizer shard group unchanged', c2.get('Q1') === c1.get('Q1') && c2.get('S6') === c1.get('S6'), '');
T.check('TP2: weights fall by less than half (replicated norms/router/down-proj + whole experts)', c2.get('W1') < before.W1 && c2.get('W1') > before.W1 / 2, `${c2.get('W1')} vs ${before.W1}`);
// optimizer state per GPU RISES under TP: sharded params keep their footprint
// (half the params over half the DP group), experts are untouched, but the
// replicated params' whole copies now shard over a DP group half the size —
// the increase is exactly 8 B × (replicated params on this rank) / 128
const Qrep = L1 * N6 + L2 * N8 + c2.get('L5') * c2.get('H1');
T.check('TP2: optimizer state grows by exactly the replicated params’ extra shard (8 × Qrep / 128)',
  Math.abs((c2.get('O1') - before.O1) - 8 * Qrep / 128) < 1e-6 && c2.get('O1') > before.O1, `${c2.get('O1') - before.O1} vs ${8 * Qrep / 128}`);
T.check('bars/readouts: 2048 tokens, activations label carries the in-flight count', txt().includes('activations ×') && l()._tok() === 2048, txt().slice(0, 120));
T.check('save label names TP2', (l()._saveBaseline(), l()._pinCfg.label.includes('·TP2·')), l()._pinCfg.label);
// the diagram chips: the norm1 weight number is REPLICATED (whole), the q up-proj number is sharded
const paramTxt = (op) => l().querySelector(`g[data-op="${op}"]`)?.textContent ?? '';
T.log('q_up box', paramTxt('q_up').slice(0, 60));
// back to TP1: everything returns
knob('tp').querySelector('button:first-child').click(); await T.tick(500);
const c3 = l()._cells();
T.check('TP1 again: totals restored exactly', c3.get('T1') === before.T1 && c3.get('A1') === before.A1 && l().tp === 1, `${c3.get('T1')} vs ${before.T1}`);
T.done();
