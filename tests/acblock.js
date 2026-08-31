// @page studies/02-hopper-memory.html
// AC + fp8 block widgets (controls="marks"/"dtype"): house knob segments,
// captionless foot with the full-width ribbon tally, FLOP bars in ONE fixed
// time unit, and the EP64·PP8 stash readout tallying with the fit chart
const ac = document.getElementById('ac-layer');
const f8 = document.getElementById('fp8-layer');
const seg = (l, k) => l.querySelector(`.lv-head .stp[data-knob="${k}"]`);
const btn = (l, k, t) => [...seg(l, k).querySelectorAll('button')].find(b => b.textContent === t);
const tally = (l) => l.querySelector('.lv-foot2 svg');
const tHead = (l) => tally(l).querySelector('text').textContent;

// ---- house knobs: one preset segment + the reserved custom chip
T.check('AC head carries the recompute segment only', seg(ac, 'recompute') && !seg(ac, 'recipe')
  && !ac.querySelector('.lv-head select'), '');
T.check('fp8 head carries the recipe segment only', seg(f8, 'recipe') && !seg(f8, 'recompute'), '');
T.check('dsv3 preset lights up', btn(ac, 'recompute', 'dsv3').classList.contains('on'), '');
T.check('custom chip reserved (present, disabled, off)', btn(ac, 'recompute', 'custom').disabled
  && !btn(ac, 'recompute', 'custom').classList.contains('on'), '');
// ---- captionless: the foot is just the tally
T.check('no prose caption on either widget', !ac.querySelector('.lv-foot2 .lv-note')
  && !f8.querySelector('.lv-foot2 .lv-note'), '');
T.check('tally spans the full runway', +tally(ac).getAttribute('width') === 1080, tally(ac).getAttribute('width'));
// ---- the stash readout prices the EP64·PP8 context: 8 MoE layers × 8.5 mb
T.check('readout carries the ×8 × 8.5 context + tag', /× 8 layers × 8.5 in flight/.test(tHead(ac))
  && tHead(ac).includes('interior rank of PP8'), tHead(ac));
const gib = () => +tHead(ac).match(/= ([\d.]+) GiB/)[1];
T.check('dsv3 policy stashes 66.6 GiB (deck: 87.5 total − 21.0 fixed)', gib() === 66.6, gib());
// ---- flipping the preset moves the readout AND the replay ribbon
const ribbons = (l) => [...tally(l).querySelectorAll('text')].map(t => t.textContent);
const replayX = (l) => +ribbons(l).find(t => /^\+[\d.]+×/.test(t)).match(/\+([\d.]+)×/)[1];
const r0 = replayX(ac);
btn(ac, 'recompute', 'attn-replay').click(); await T.tick(300);
T.check('attn-replay: stash drops below dsv3', gib() < 66.6, gib());
T.check('attn-replay: replay ribbon grows', replayX(ac) > r0, `${replayX(ac)} vs ${r0}`);
T.check('segment follows', btn(ac, 'recompute', 'attn-replay').classList.contains('on'), '');
// a hand-flipped mark lands on the reserved custom chip (no reflow)
ac.querySelector('button[data-mark="o_proj"]')?.click(); await T.tick(300);
T.check('hand-edited marks light the custom chip', btn(ac, 'recompute', 'custom').classList.contains('on'), '');
btn(ac, 'recompute', 'dsv3').click(); await T.tick(300);
T.check('back to dsv3', gib() === 66.6, gib());
// ---- exactness: readout = the fit chart's per-layer quantum × 8 × 8.5
{
  const l = document.getElementById('local-diagram');
  const kib = +tHead(ac).match(/([\d.]+) KiB\/token·layer/)[1];
  // the fit chart at PP8 prices interior ranks with the same analyze() —
  // recompute here is 'none' there, so compare the no-AC readout instead
  btn(ac, 'recompute', 'none').click(); await T.tick(300);
  const kibNone = +tHead(ac).match(/([\d.]+) KiB\/token·layer/)[1];
  T.check('no-AC quantum = 1786.6 MiB/mb·layer (the fit chart moe rate; readout rounds to whole KiB)',
    Math.abs(kibNone * 4096 / 1024 - 1786.6) < 2.1, (kibNone * 4096 / 1024).toFixed(1));
  T.check('dsv3 quantum is smaller', kib < kibNone, `${kib} vs ${kibNone}`);
  btn(ac, 'recompute', 'dsv3').click(); await T.tick(300);
}
// ---- FLOP pickets: one FIXED unit (20 MFLOP/token ≈ 83 µs/mb) — an fp8
// flip halves the COUNT, not the scale. Counting via the fwd ribbon (= the
// boxes' pickets laid end to end): bf16 fwd ≈ 1.34 GFLOP/tok = 67 pickets.
const fwdN = (l) => {
  const lab = [...tally(l).querySelectorAll('text')].find(t => t.textContent === 'fwd');
  const y0 = +lab.getAttribute('y') - 6;
  return [...tally(l).querySelectorAll('rect[height="5"]')]
    .filter(r => Math.abs(+r.getAttribute('y') - y0) < 3).length;
};
T.check('AC widget (bf16): fwd = 67 pickets at the fixed unit', fwdN(ac) === 67, fwdN(ac));
T.check('fp8 widget: mxfp8 shrinks the tally (44 pickets)', fwdN(f8) === 44, fwdN(f8));
{ // recipe flip to bf16 restores the count — the unit never renormalizes
  btn(f8, 'recipe', 'bf16').click(); await T.tick(400);
  T.check('recipe→bf16 restores the full count', fwdN(f8) === 67, fwdN(f8));
  T.check('fp8 readout follows the recipe', /= 118\.6 GiB/.test(tHead(f8)), tHead(f8));
  btn(f8, 'recipe', 'dsv3-fp8').click(); await T.tick(400);
  T.check('back to dsv3-fp8', /= 86\.2 GiB/.test(tHead(f8)), tHead(f8));
}
// vector ops keep the unpriced fig-leaf (hollow dashed); sub-picket GEMMs
// (router, kv down-proj half) wear the hollow trace
T.check('norms/swiglu wear the hollow dashed fig-leaf',
  ac.querySelectorAll('.lv-scroll rect[height="4"][stroke-dasharray]').length >= 3, '');
T.check('sub-picket GEMMs wear the hollow trace',
  ac.querySelectorAll('.lv-scroll rect[width="1.4"]').length >= 2, '');
// ---- the consolidated stash: SOLID linear bar over the GiB ruler
{
  const bar = tally(ac).querySelector('rect[data-true]');
  const ghost = tally(ac).querySelector('rect[stroke-dasharray="2 2"]');
  T.check('total bar carries its exact bytes', Math.abs(+bar.dataset.true / 2 ** 30 - 66.6) < 0.1, bar.dataset.true);
  // linear mapping: 6px per GiB from x=44
  T.check('bar length = 6px/GiB (linear)',
    Math.abs(+bar.getAttribute('width') - +bar.dataset.true / 2 ** 30 * 6) < 1, bar.getAttribute('width'));
  T.check('ghost = the dashed OVERHANG up to the untreated 118.6 GiB anchor',
    Math.abs(+ghost.getAttribute('x') + +ghost.getAttribute('width') - (44 + 118.6 * 6)) < 2, '');
  T.check('badge prices the lever vs the anchor', ribbons(ac).some(t => t.includes('▼×1.8')), '');
  T.check('the 80 GiB card line is drawn', ribbons(ac).some(t => t === '80 GiB'), '');
  T.check('both rulers present (GiB + GFLOP/token)',
    ribbons(ac).some(t => t.includes('minor tick = 1 GiB')) && ribbons(ac).some(t => t.includes('GFLOP/token')), '');
}
// ---- transitions ANIMATE (deterministic 12-frame tween, ~200 ms)
{
  const repN = () => { // replay ribbon picket count
    const lab = [...tally(ac).querySelectorAll('text')].find(t => t.textContent === 'replay');
    const y0 = +lab.getAttribute('y') - 6;
    return [...tally(ac).querySelectorAll('rect[height="5"]')]
      .filter(r => Math.abs(+r.getAttribute('y') - y0) < 3).length;
  };
  const w0 = repN();
  btn(ac, 'recompute', 'attn-replay').click(); await T.tick(60);   // mid-tween
  const wMid = repN();
  const dissolving = ac.querySelectorAll('.lv-scroll g[opacity] text.tredo').length;
  await T.tick(400);                                               // settled
  const w1 = repN();
  T.check('replay ribbon pours (mid-tween strictly between endpoints)',
    wMid > w0 + 1 && wMid < w1 - 1, `${w0} < ${wMid} < ${w1}`);
  T.check('chips dissolve mid-tween (both text forms present)', dissolving > 0, dissolving);
  T.check('tween settles clean (no opacity groups left)',
    !ac.querySelector('.lv-scroll g[opacity] text.tredo'), '');
  btn(ac, 'recompute', 'dsv3').click(); await T.tick(400);
}
{ // dtype flip: the picket count pours through the tween (fixed unit)
  btn(f8, 'recipe', 'bf16').click(); await T.tick(60);
  const mid = fwdN(f8);
  await T.tick(400);
  T.check('picket count lerps through the dtype flip', mid > 46 && mid < 65, mid);
  T.check('and lands on the full count', fwdN(f8) === 67, fwdN(f8));
  btn(f8, 'recipe', 'dsv3-fp8').click(); await T.tick(400);
}
T.done();
