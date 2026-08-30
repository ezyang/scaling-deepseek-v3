// @page studies/02-hopper-memory.html
// the strip's schedule-family knob: textbook programs on the plain pipeline
// (1F1B · ZB-H1 · GPipe), DSv3's official program vs greedy on the DualPipeV
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
T.check('plain shape offers 1F1B* / ZB-H1 / GPipe', segBtn('1F1B').classList.contains('on')
  && !!segBtn('ZB-H1') && !!segBtn('GPipe'), '');
T.check('1F1B draws the law (pp−s, no tag)', peak().includes('15 mb') && !peak().includes('charges'), peak());
segBtn('GPipe').click(); await T.tick(500);
T.check('GPipe stashes ALL m — and the label says the model charges the 1F1B law',
  peak().includes('20 mb') && peak().includes('charges 15'), peak());
const lastF = Math.max(...[...w.querySelectorAll('rect[data-cell^="F"]')].filter((r) => r.dataset.v === '0').map((r) => +r.dataset.t1));
const firstB = Math.min(...[...w.querySelectorAll('rect[data-cell^="B"]')].filter((r) => r.dataset.v === '0').map((r) => +r.dataset.t0));
T.check('GPipe on s0: every forward precedes every backward (the flush)', lastF <= firstB, `${lastF} vs ${firstB}`);
const t1f1bEnd = () => Math.max(...[...w.querySelectorAll('rect[data-cell]')].map((r) => +r.dataset.t1));
segBtn('1F1B').click(); await T.tick(500);
const T1 = t1f1bEnd();
segBtn('ZB-H1').click(); await T.tick(500);
const bcells = () => [...w.querySelectorAll('rect[data-cell^="B"]')];
T.check('ZB-H1: shorter than 1F1B (bubbles filled), 1F1B residency (no tag)',
  t1f1bEnd() < T1 && peak().includes('15 mb') && !peak().includes('charges'), `${t1f1bEnd()} vs ${T1} | ${peak()}`);
T.check('rank 0 keeps its backward fused (the authors\u2019 patch)',
  bcells().filter((r) => r.dataset.cell.endsWith('@0')).every((r) => r.dataset.t1 - r.dataset.t0 === 2), '');
T.check('every deferred W drains (one W per split b)',
  cellN('W') === bcells().filter((r) => r.dataset.t1 - r.dataset.t0 === 1).length,
  `${cellN('W')} W`);
knob('vpp').querySelectorAll('button')[1].click(); await T.tick(700);   // back to the DualPipeV shape
T.check('back to VPP2: DPV families again', !!segBtn('DSv3') && !segBtn('GPipe'), '');
T.done();
