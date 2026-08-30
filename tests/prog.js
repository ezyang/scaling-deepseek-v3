// @page studies/02-hopper-memory.html
// the strip's program knob + compact notation editor: official DualPipeV vs
// greedy vs a reader-written program (strip-local — the model charges the law)
const w = document.querySelector('dsv3-pp-schedule');
const l = document.getElementById('local-diagram');
const knob = (k) => w.querySelector(`[data-knob="${k}"]`);
const cellN = (ph) => w.querySelectorAll(`rect[data-cell^="${ph}"]`).length;
const segBtn = (t) => [...knob('prog').querySelectorAll('button')].find((b) => b.textContent === t);
{ const msel = knob('mb'); msel.value = 'auto'; msel.dispatchEvent(new Event('change')); await T.tick(200); }

T.check('program knob offers DSv3 + greedy, DSv3 on', segBtn('DSv3').classList.contains('on')
  && !!segBtn('greedy'), '');
const offCounts = [cellN('F'), cellN('B'), cellN('W')].join('/');
segBtn('greedy').click(); await T.tick(400);
T.check('greedy: no zero-bubble split, no fused blocks', cellN('W') === 0
  && ![...w.querySelectorAll('rect[data-cell^="F"]')].some((r) => +r.getAttribute('width') === 9), '');
T.check('greedy still draws the modeled residency (no charge tag)',
  w.querySelector('text[data-peak]').textContent.includes('16.5 mb in flight')
  && !w.querySelector('text[data-peak]').textContent.includes('charges'), w.querySelector('text[data-peak]').textContent);
segBtn('DSv3').click(); await T.tick(400);

// ✎ editor: serialize → apply unmodified → identical drawing, tagged custom
knob('edit').click(); await T.tick(100);
const ta = () => w.querySelector('.ped textarea');
T.check('editor prefills the drawn program (one line per rank, RLE)',
  ta().value.split('\n').length === l.pp && /F0x\d+/.test(ta().value), ta().value.slice(0, 60));
w.querySelector('.ped button').click(); await T.tick(400);
T.check('round-trip: the custom program redraws the official cells',
  [cellN('F'), cellN('B'), cellN('W')].join('/') === offCounts
  && segBtn('custom')?.classList.contains('on'), [cellN('F'), cellN('B'), cellN('W')].join('/'));
T.check('no gap warnings on a complete program', !w.querySelector('.pwarn:not(:empty)')
  || [...w.querySelectorAll('.pwarn')].every((x) => !x.textContent), '');

// break it: drop s0's last weight-grad → the gap is NAMED, still drawn
ta().value = ta().value.split('\n').map((ln, i) => i === 0 ? ln.replace(/ \S+$/, '') : ln).join('\n');
w.querySelector('.ped button').click(); await T.tick(400);
T.check('incomplete program names its gap', [...w.querySelectorAll('.pwarn')]
  .some((x) => x.textContent.includes('weight grads')), '');
// nonsense token → a parse error, program unchanged
ta().value = 'garbage\n' + ta().value.split('\n').slice(1).join('\n');
w.querySelector('.ped button').click(); await T.tick(200);
T.check('a bad token is a parse error', [...w.querySelectorAll('.pwarn')]
  .some((x) => x.textContent.includes('can’t read') || x.textContent.includes("can't read")), '');

// a config change invalidates the custom program (its counts are stale)
knob('vpp').querySelectorAll('button')[0].click(); await T.tick(700);   // VPP 2 -> 1
T.check('config change drops the custom program', !segBtn('custom')
  && w.querySelector('.ped').style.display === 'none', '');
T.check('VPP1: no program seg choice needed (greedy only)', !segBtn('DSv3') && !!segBtn('greedy'), '');
knob('vpp').querySelectorAll('button')[1].click(); await T.tick(700);   // back to DualPipeV
T.check('back to VPP2: official again', segBtn('DSv3').classList.contains('on') && cellN('W') > 0, '');
T.done();
