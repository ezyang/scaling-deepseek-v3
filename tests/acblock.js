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
// ---- FLOP pickets: one FIXED unit (10 MFLOP/token ≈ 41 µs/mb) — an fp8
// flip halves the COUNT, not the scale. Counting via the fwd ribbon (= the
// boxes' pickets laid end to end): bf16 fwd ≈ 1.34 GFLOP/tok = 134 pickets.
const fwdN = (l) => {
  const lab = [...tally(l).querySelectorAll('text')].find(t => t.textContent === 'fwd');
  const y0 = +lab.getAttribute('y') - 6;
  return [...tally(l).querySelectorAll('rect[height="5"]')]
    .filter(r => Math.abs(+r.getAttribute('y') - y0) < 3).length;
};
T.check('AC widget (bf16): fwd = 134 pickets at the fixed unit', fwdN(ac) === 134, fwdN(ac));
T.check('fp8 widget: mxfp8 shrinks the tally (88 pickets)', fwdN(f8) === 88, fwdN(f8));
{ // recipe flip to bf16 restores the count — the unit never renormalizes
  btn(f8, 'recipe', 'bf16').click(); await T.tick(400);
  T.check('recipe→bf16 restores the full count', fwdN(f8) === 134, fwdN(f8));
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
  // the honesty line: what the ruler does NOT meter, per the CURRENT policy
  T.check('unmetered-vector line names this policy\'s replays (dsv3: RoPE ×2 + norms + SwiGLU)',
    ribbons(ac).some(t => t.startsWith('not priced') && t.includes('RoPE ×2')
      && t.includes('RMSNorm ×4') && t.includes('SwiGLU')), '');
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
// ---- always-detail: no elided-kernels toggle; the latent norms are markable
{
  T.check('AC tier is always the detail view (RoPE micros drawn)',
    [...ac.querySelectorAll('.lv-scroll text')].some(t => t.textContent === 'RoPE'), '');
  T.check('no elided-kernels checkbox on the house head',
    ![...ac.querySelectorAll('.lv-head label')].some(l => l.textContent.includes('elided')), '');
  btn(ac, 'recompute', 'none').click(); await T.tick(400);
  const qb = ac.querySelector('button[data-mark="q_norm"]');
  T.check('the q-latent RMSNorm micro carries a mark button', qb?.textContent === '💾', qb?.textContent);
  qb.click(); await T.tick(400);
  T.check('hand-flipping q_norm lands on custom and reprices',
    btn(ac, 'recompute', 'custom').classList.contains('on') && /= 117\.8 GiB/.test(tHead(ac)), tHead(ac));
  btn(ac, 'recompute', 'dsv3').click(); await T.tick(400);
}
// ---- the fixed-config readout + mirrored/fiat marks
{
  // the ctx READOUT row mirrors the full sim's knob layout, locked
  const rows = [...ac.querySelectorAll('.lv-head')];
  T.check('ctx row mirrors the sim groups, policy row below',
    rows.length >= 2 && !!rows[0].querySelector('[data-knob="gpus"]')
    && !!rows[1].querySelector('[data-knob="recompute"]')
    && ['cluster', 'pipeline', 'SPMD mesh', 'ZeRO'].every(x =>
      [...rows[0].querySelectorAll('.parlab')].some(l2 => l2.textContent === x)), '');
  T.check('locked chips carry the config (PP8 · r1–7 · DualPipeV · EP64 · ZeRO-1)',
    ac.querySelector('.stp[data-knob="pp"] button.on')?.textContent === '8'
    && ac.querySelector('.stp[data-knob="rank"] button.on')?.textContent.includes('r1–7')
    && ac.querySelector('.stp[data-knob="sched"] button.on')?.textContent === 'DualPipeV'
    && ac.querySelector('.stp[data-knob="ep"] button.on')?.textContent === '64'
    && ac.querySelector('.stp[data-knob="zero"] button.on')?.textContent === '1'
    && [...rows[0].querySelectorAll('button')].every(b => b.disabled), '');
  // custom MEMORY + toggle-back: leave custom, return to it, ping-pong
  ac.querySelector('button[data-mark="norm1"]').click(); await T.tick(400);   // dsv3 → custom
  const gX = tHead(ac).match(/= ([\d.]+) GiB/)[1];
  btn(ac, 'recompute', 'none').click(); await T.tick(400);
  T.check('custom chip stays ENABLED after leaving', !btn(ac, 'recompute', 'custom').disabled, '');
  btn(ac, 'recompute', 'custom').click(); await T.tick(400);
  T.check('custom restores the hand-edited state', tHead(ac).includes(`= ${gX} GiB`)
    && btn(ac, 'recompute', 'custom').classList.contains('on'), '');
  btn(ac, 'recompute', 'custom').click(); await T.tick(400);
  T.check('clicking the ACTIVE chip toggles back (to none)', btn(ac, 'recompute', 'none').classList.contains('on'), '');
  btn(ac, 'recompute', 'dsv3').click(); await T.tick(400);
}
// ---- kind pinned to MoE (the peak rank), tabs gone, REGION toggle on the
// enclosure; marks are LITERAL (torch_remat): full = exactly 1× fwd replay
{
  const rbtn = (a) => ac.querySelector(`button[data-regionact="${a}"][data-mem="ffnMixed"]`);
  const replayN = () => ribbons(ac).find(t => /^\+[\d.]+×/.test(t));
  T.check('no dense/MoE tab flaps (kind pinned)', !ac.querySelector('[data-kind]'), '');
  T.check('enclosure wears the static MoE FFN label',
    [...ac.querySelectorAll('.lv-scroll text')].some(t => t.textContent.includes('MoE FFN ×58')), '');
  T.check('region toggle: dsv3 reads as mixed (swiglu ↻, rest 💾)',
    rbtn('mixed')?.dataset.on === '1', '');
  rbtn('redo').click(); await T.tick(500);
  T.check('↻ all: the whole FFN replays (stash drops, a2a in the replay)',
    gib() < 66.6 && replayN().includes('a2a dispatch+combine'), `${gib()} ${replayN()}`);
  rbtn('save').click(); await T.tick(500);
  T.check('💾 all: every FFN output stashed', gib() > 66.6, gib());
  rbtn('mixed').click(); await T.tick(500);
  T.check('mixed restores the exact dsv3 marks (preset chip relights)',
    gib() === 66.6 && btn(ac, 'recompute', 'dsv3').classList.contains('on'), gib());
  btn(ac, 'recompute', 'full').click(); await T.tick(500);
  T.check('LITERAL remat: full replays exactly 1.00× fwd', replayN().startsWith('+1.00×'), replayN());
  // the MLA column carries the same toggle: attn-replay = ↻ all there
  const mbtn = (a) => ac.querySelector(`button[data-regionact="${a}"][data-mem="mlaMixed"]`);
  btn(ac, 'recompute', 'attn-replay').click(); await T.tick(500);
  T.check('MLA region toggle: attn-replay reads as ↻ all', mbtn('redo')?.dataset.on === '1', '');
  mbtn('save').click(); await T.tick(500);
  T.check('💾 all MLA: stash grows past dsv3', gib() > 66.6, gib());
  mbtn('mixed').click(); await T.tick(500);
  T.check('MLA mixed restores the dsv3-side composition', mbtn('mixed').dataset.on === '1', '');
  mbtn('mixed').click(); await T.tick(500);   // active chip → back to the previous pick (💾 all)
  T.check('region toggle-back: active chip returns to the previous pick', mbtn('save')?.dataset.on === '1', '');
  T.check('shared expert mirrors the grouped marks (two buttons per mark)',
    ac.querySelectorAll('button[data-mark="gate_up"]').length === 2
    && ac.querySelectorAll('button[data-mark="ffn_down"]').length === 2
    && ac.querySelectorAll('button[data-mark="swiglu"]').length === 2, '');
  const sh = [...ac.querySelectorAll('button[data-mark="gate_up"]')];
  const before = sh[0].textContent;
  sh[1].click(); await T.tick(400);
  T.check('clicking the shared button flips BOTH (one graph node)',
    [...ac.querySelectorAll('button[data-mark="gate_up"]')].every(b => b.textContent !== before), '');
  // RoPE is a REAL (zero-byte) mark: clickable; flipping it moves the stash
  // between the rotated and pre-RoPE tensors \u2014 the total holds exactly
  btn(ac, 'recompute', 'none').click(); await T.tick(400);
  const rq = () => ac.querySelector('button[data-mark="rope_q"]');
  T.check('RoPE pills carry LIVE mark buttons', !!rq() && !rq().disabled
    && !!ac.querySelector('button[data-mark="rope_kv"]'), '');
  T.check('kv down-proj mirrors qkv_down (2 buttons, one node)',
    ac.querySelectorAll('button[data-mark="qkv_down"]').length === 2, '');
  T.check('the block-output add wears the locked \ud83d\udd12 (its output IS next-x0)',
    [...ac.querySelectorAll('.lv-scroll button:disabled')].filter(b => b.textContent === '\ud83d\udd12').length === 1, '');
  // the routed+shared sum is a real node (no fusion modeled): its ↻ mark is
  // representable and honestly WASTEFUL — the replay pulls combine-out in
  const g1 = tHead(ac).match(/= ([\d.]+) GiB/)[1];
  ac.querySelector('button[data-mark="moe_add"]').click(); await T.tick(400);
  T.check('marking the routed+shared sum ↻ GROWS the stash (literal semantics)',
    +tHead(ac).match(/= ([\d.]+) GiB/)[1] > +g1, tHead(ac));
  ac.querySelector('button[data-mark="moe_add"]').click(); await T.tick(400);
  const g0 = tHead(ac).match(/= ([\d.]+) GiB/)[1];
  rq().click(); await T.tick(400);   // none \u2192 rope_q \u21bb: same bytes, now pre-RoPE
  T.check('flipping RoPE moves ZERO bytes (the stash total holds)',
    tHead(ac).includes(`= ${g0} GiB`), tHead(ac));
  T.check('\u2026but the unmetered line picks up the re-run',
    ribbons(ac).some(t => t.startsWith('not priced') && t.includes('RoPE')), '');
  btn(ac, 'recompute', 'dsv3').click(); await T.tick(400);
}
{ // dtype flip: the picket count pours through the tween (fixed unit)
  btn(f8, 'recipe', 'bf16').click(); await T.tick(60);
  const mid = fwdN(f8);
  await T.tick(400);
  T.check('picket count lerps through the dtype flip', mid > 92 && mid < 130, mid);
  T.check('and lands on the full count', fwdN(f8) === 134, fwdN(f8));
  btn(f8, 'recipe', 'dsv3-fp8').click(); await T.tick(400);
}
T.done();
