// @page studies/02-hopper-memory.html
// DualPipeV is THE schedule: VPP2·reflect derived from PP (never knobs) —
// V-fold stage map, emb+head on rank 0, uniform PP+½ in flight; PP1
// degenerates to one trivial chunk
const l = () => document.getElementById('local-diagram');
const knob = (k) => l().parentElement.querySelector(`.stp[data-knob="${k}"]`);
const plan = () => l().closest('.anat-grid').querySelector('dsv3-anatomy-plan');

T.check('no VPP or fold knobs anywhere', !knob('vpp') && !knob('fold'), '');
T.check('DualPipeV derived at PP8 (VPP2 · reflect)', l().vpp === 2 && l().fold === 'reflect',
  `${l().vpp}/${l().fold}`);
const pp = l().pp;
T.log('pp', pp);
// the in-flight law under the V: uniform pp + 0.5 on every stage
const barTxt = () => [...l().querySelectorAll('text')].map(t => t.textContent).join('|');
T.check('fit chart shows uniform PP+0.5 in flight', barTxt().includes(`activations ×${pp + 0.5}mb`), barTxt().slice(0, 120));
// stage select: stage 0 hosts two chunks + emb + head
const opts = () => [...l().parentElement.querySelectorAll('select')].flatMap(s => [...s.options].map(o => o.textContent));
const s0 = opts().find(t => t.startsWith('0: '));
T.check('stage 0 label = two ranges + emb+head', /\+L\d+/.test(s0) && s0.includes('emb+head'), s0);
T.check('last stage label has no head tag', !opts().find(t => t.startsWith(`${pp - 1}: `)).includes('head'), '');
// visit stage 0: the plan shows embedding AND lm head resident
l().setLocal(() => { l().stage = 0; }); await T.tick(500);
T.check('stage 0 plan: lm head resident', !plan().textContent.includes('(last stage only)'), '');
l().setLocal(() => { l().stage = l().pp - 1; }); await T.tick(500);
T.check('last stage plan: head placeholder points at stage 0',
  plan().textContent.includes('(stage 0 only)'), '');
// PP1: the derivation degenerates to one trivial chunk (no V at depth 1)
const stageSel = () => l().parentElement.querySelector('select[data-knob="stage"]');
const selW = stageSel().getBoundingClientRect().width;
knob('pp').querySelector('select.v').value = '1';
knob('pp').querySelector('select.v').dispatchEvent(new Event('change')); await T.tick(700);
T.check('PP1 → VPP1 (derived)', l().pp === 1 && l().vpp === 1, `${l().pp}/${l().vpp}`);
T.check('PP1 charges ×1 in flight', barTxt().includes('activations ×1mb'), '');
knob('pp').querySelector('select.v').value = '8';
knob('pp').querySelector('select.v').dispatchEvent(new Event('change')); await T.tick(700);
T.check('PP8 → VPP2 again, stage select width fixed', l().vpp === 2
  && stageSel().getBoundingClientRect().width === selW, '');

// ---- the strip draws the official program; residency counted off the cells
const w = document.querySelector('dsv3-pp-schedule');
{
  const msel = w.querySelector('[data-knob="mb"]');
  msel.value = 'auto'; msel.dispatchEvent(new Event('change')); await T.tick(150);
  const PP = l().pp, D = 2 * PP, M = D + 4;
  const all = [...w.querySelectorAll('rect[data-cell]')];
  const nOf = (ph) => all.filter(r => r.dataset.cell.startsWith(ph)).length;
  T.check('every F and B drawn', nOf('F') === D * M && nOf('B') === D * M, `${nOf('F')}/${nOf('B')} vs ${D * M}`);
  // official program: W (deferred weight grads) = sum over ranks of 2PP−r−1
  const wExpect = [...Array(PP).keys()].reduce((t, r) => t + 2 * PP - r - 1, 0);
  T.check('W cells match the zero-bubble count', nOf('W') === wExpect, `${nOf('W')} vs ${wExpect}`);
  // fusion cue = contiguity: a fused F touches its B (shared edge), while
  // ordinary neighbours keep a visible gap
  const fusedF = all.filter(r => r.dataset.cell.startsWith('F') && +r.getAttribute('width') === 9);
  T.check('fused F cells drawn flush (width 9 vs gapped 7)', fusedF.length > 0
    && all.some(r => r.dataset.cell.startsWith('F') && +r.getAttribute('width') === 7), fusedF.length);
  const touches = fusedF.every(f => all.some(b => b.dataset.cell.startsWith('B')
    && b.dataset.cell.endsWith('@' + f.dataset.cell.split('@')[1])
    && Math.abs((+f.getAttribute('x') + +f.getAttribute('width')) - +b.getAttribute('x')) < 0.01));
  T.check('every fused F shares an edge with a B', touches, '');
  const row0v = new Set(all.filter(r => r.dataset.cell.endsWith('@0')).map(r => +r.dataset.v));
  T.check('rank 0 hosts virtual stages 0 and ' + (D - 1), row0v.has(0) && row0v.has(D - 1), [...row0v].join(','));
  const row1 = all.filter(r => r.dataset.cell.endsWith('@1'))
    .map(r => [+r.dataset.t0, +r.dataset.t1]).sort((a, b) => a[0] - b[0]);
  T.check('rank rows never overlap in time', row1.every((c, i) => !i || c[0] >= row1[i - 1][1]), '');
  // stashes live from F end to B end; peak on a rank = the modeled 2PP+1
  const fEnd = new Map(), bEnd = new Map();
  for (const r of all) {
    const ph = r.dataset.cell[0];
    if (r.dataset.cell.endsWith('@1')) (ph === 'F' ? fEnd : bEnd).set(r.dataset.v + ':' + r.dataset.cell.slice(1).split('@')[0], +r.dataset.t1);
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
T.done();
