// @page studies/02-hopper-memory.html
// DualPipeV schedule option: V-fold stage map (emb+head on rank 0), uniform
// PP+1/2 in-flight, sched knobs on both widgets
const l = () => document.getElementById('local-diagram');
const stps = () => [...l().parentElement.querySelectorAll('.stp')];
const schedBtns = () => stps().map(s => [...s.querySelectorAll('button')]).flat();
const plan = () => l().closest('.anat-grid').querySelector('dsv3-anatomy-plan');

T.check('layer sched knob has DPV', !!schedBtns().find(b => b.textContent === 'DPV'), '');
schedBtns().find(b => b.textContent === 'DPV').click(); await T.tick(500);
T.check('sched flips to dpv', l().sched === 'dpv', l().sched);

const pp = l().pp;
T.log('pp', pp);
// the in-flight law: uniform pp + 0.5 (the fit chart's activations row names it)
const barTxt = [...l().querySelectorAll('text')].map(t => t.textContent).join('|');
T.check('fit chart shows uniform PP+0.5 in flight', barTxt.includes(`activations ×${pp + 0.5}mb`), barTxt.slice(0, 120));
// stage select: stage 0 hosts two chunks + emb + head
const ssel = l().parentElement.querySelectorAll('.pargrp select')[1] ?? null;
const opts = [...l().parentElement.querySelectorAll('select')].flatMap(s => [...s.options].map(o => o.textContent));
const s0 = opts.find(t => t.startsWith('0: '));
T.check('stage 0 label = two ranges + emb+head', /\+L\d+/.test(s0) && s0.includes('emb+head'), s0);
const sLast = opts.find(t => t.startsWith(`${pp - 1}: `));
T.check('last stage label has no head tag', !sLast.includes('head'), sLast);
// visit stage 0: the plan shows embedding AND lm head resident
l().setLocal(() => { l().stage = 0; }); await T.tick(500);
const planTxt = plan().textContent;
T.check('stage 0 plan: lm head resident (dims not placeholder)',
  !planTxt.includes('(last stage only)'), '');
// visit the last stage: head is NOT here under the fold; placeholder names stage 0
l().setLocal(() => { l().stage = l().pp - 1; }); await T.tick(500);
T.check('last stage plan: head placeholder points at stage 0',
  plan().textContent.includes('(stage 0 only)'), '');
// numbers ≡ squares: total activations bytes in the fit chart use pp+0.5
const segs = l()._segTotals;
T.log('segTotals', segs && segs.map(v => (v / 2 ** 30).toFixed(1)).join('/'));
// the strip widget: DPV segment present and the V fold actually drawn
const w = document.querySelector('dsv3-pp-schedule');
const wDpv = [...w.querySelectorAll('.stp button')].find(b => b.textContent === 'DPV');
T.check('strip replica has DPV segment (on)', wDpv && wDpv.classList.contains('on'), '');
{
  const msel = [...w.querySelectorAll('select')].at(-1);
  msel.value = 'auto'; msel.dispatchEvent(new Event('change')); await T.tick(150);
  const PP = l().pp, D = 2 * PP, M = D + 4;
  const all = [...w.querySelectorAll('rect[data-cell]')];
  T.check('every virtual-stage op drawn', all.length === 2 * D * M, `${all.length} vs ${2 * D * M}`);
  // rank rows host TWO chunks: row 0 carries cells of virtual stages 0 and D−1
  const row0v = new Set(all.filter(r => r.dataset.cell.endsWith('@0')).map(r => +r.dataset.v));
  T.check('rank 0 hosts virtual stages 0 and ' + (D - 1), row0v.has(0) && row0v.has(D - 1), [...row0v].join(','));
  // rank serialization: no overlapping cells within a row
  const row1 = all.filter(r => r.dataset.cell.endsWith('@1'))
    .map(r => [+r.dataset.t0, +r.dataset.t1]).sort((a, b) => a[0] - b[0]);
  T.check('rank rows never overlap in time', row1.every((c, i) => !i || c[0] >= row1[i - 1][1]), '');
  // residency: stashes live from F end to B end; the peak on a rank should
  // hit the modeled 2PP+1 half-rank chunks (±1 for chunk phase alignment)
  const fEnd = new Map(), bEnd = new Map();
  for (const r of all) {
    const [ph, rest] = [r.dataset.cell[0], r.dataset.cell.slice(1)];
    if (r.dataset.cell.endsWith('@1')) (ph === 'F' ? fEnd : bEnd).set(r.dataset.v + ':' + rest.split('@')[0], +r.dataset.t1);
  }
  let peak = 0;
  for (const t of new Set([...fEnd.values()])) {
    let live = 0;
    for (const [k, fe] of fEnd) if (fe <= t && (bEnd.get(k) ?? Infinity) > t) live++;
    peak = Math.max(peak, live);
  }
  T.log('drawn peak residency (rank 1, half-chunks)', `${peak} vs modeled ${2 * PP + 1}`);
  T.check('drawn residency matches the law', Math.abs(peak - (2 * PP + 1)) <= 1, peak);
}
// flip back to 1F1B from the strip: everything returns
[...w.querySelectorAll('.stp button')].find(b => b.textContent === '1F1B').click(); await T.tick(500);
T.check('back to 1f1b, strip redraws cells', l().sched === '1f1b' && w.querySelectorAll('rect[data-cell]').length > 0, '');
T.done();
