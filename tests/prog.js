// @page studies/02-hopper-memory.html
// the strip's schedule-family knob: textbook programs on the plain pipeline
// (1F1B · ZB1P · GPipe), DSv3's official program vs greedy on the DualPipeV
// shape — strip-local; the peak label calls out any drawn-vs-law difference
const w = document.querySelector('dsv3-pp-schedule');
const knob = (k) => w.querySelector(`[data-knob="${k}"]`);
const segBtn = (t) => [...(knob('prog')?.querySelectorAll('button') ?? [])].find((b) => b.textContent === t);
const cellN = (ph) => w.querySelectorAll(`rect[data-cell^="${ph}"]`).length;
const peak = () => w.querySelector('text[data-peak]')?.textContent ?? '';
{ const msel = knob('mb'); msel.value = 'auto'; msel.dispatchEvent(new Event('change')); await T.tick(200); }

// DualPipeV shape: official on by default, greedy the alternative
T.check('DPV shape offers DSv3* / greedy', segBtn('DSv3').classList.contains('on') && !!segBtn('greedy'), '');
T.check('official draws the zero-bubble split', cellN('W') > 0, cellN('W'));
segBtn('greedy').click(); await T.tick(400);
T.check('greedy: no W cells, no fused blocks, same modeled residency', cellN('W') === 0
  && ![...w.querySelectorAll('rect[data-cell^="F"]')].some((r) => +r.getAttribute('width') === 9)
  && peak().includes('16.5') && !peak().includes('charges'), peak());
segBtn('DSv3').click(); await T.tick(400);

// plain pipeline: the textbook set
knob('vpp').querySelectorAll('button')[0].click(); await T.tick(700);   // VPP 2 -> 1
T.check('plain shape offers 1F1B* / ZB1P / GPipe', segBtn('1F1B').classList.contains('on')
  && !!segBtn('ZB1P') && !!segBtn('GPipe'), '');
T.check('1F1B draws the law (pp−s, no tag)', peak().includes('15 mb') && !peak().includes('charges'), peak());
segBtn('GPipe').click(); await T.tick(500);
T.check('GPipe stashes ALL m — and the label says the model charges the 1F1B law',
  peak().includes('20 mb') && peak().includes('charges 15'), peak());
const lastF = Math.max(...[...w.querySelectorAll('rect[data-cell^="F"]')].filter((r) => r.dataset.v === '0').map((r) => +r.dataset.t1));
const firstB = Math.min(...[...w.querySelectorAll('rect[data-cell^="B"]')].filter((r) => r.dataset.v === '0').map((r) => +r.dataset.t0));
T.check('GPipe on s0: every forward precedes every backward (the flush)', lastF <= firstB, `${lastF} vs ${firstB}`);
segBtn('ZB1P').click(); await T.tick(500);
T.check('ZB1P: b one slot + W cells, 1F1B residency (no tag)', cellN('W') > 0
  && [...w.querySelectorAll('rect[data-cell^="B"]')].every((r) => r.dataset.t1 - r.dataset.t0 === 1)
  && peak().includes('15 mb') && !peak().includes('charges'), peak());
// every deferred W runs: weight grads complete (one W per b)
T.check('ZB1P completes the weight grads', cellN('W') === cellN('B'), `${cellN('W')} vs ${cellN('B')}`);
knob('vpp').querySelectorAll('button')[1].click(); await T.tick(700);   // back to the DualPipeV shape
T.check('back to VPP2: DPV families again', !!segBtn('DSv3') && !segBtn('GPipe'), '');
T.done();
