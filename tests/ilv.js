// @page studies/03-blackwell-memory.html
// the Megatron family: interleaved 1F1B with VP chunks on the wrap fold,
// layout strings, GB200/GB300 capacity — the schedule strip's drawn peak
// equals the law, the fold deals chunks round-robin, the sheet carries the
// per-kind in-flight cells, and the knobs the Blackwell post adds exist
const l = () => document.getElementById('local-diagram');
const knob = (k) => l().parentElement.querySelector(`.stp[data-knob="${k}"]`);
await T.tick(600);
T.check('sandbox opens on the GB300 anchor: PP2 · VP8 · EP32 · interleaved · wrap',
  l().pp === 2 && l().vpp === 8 && l().ep === 32 && l().sched === 'interleaved' && l().fold === 'wrap' && l().hw === 'gb300',
  JSON.stringify({ pp: l().pp, vpp: l().vpp, ep: l().ep, sched: l().sched, fold: l().fold, hw: l().hw }));
T.check('VP, rank stepper and GPU knobs render; PP offers 1/2/4/8', knob('vpp') && knob('hw') && knob('rank')?.querySelector('select')
  && [...knob('pp').querySelectorAll('option')].map((o) => o.value).join() === '1,2,4,8,16', '');
// the cells: chunk-weighted in-flight per kind + the rank law
const c = l()._cells();
// rank 0 of Et*4|(t*4|)*14tmL at PP2/VP8 with the a2a overlap: chunks hold
// 4,2,2,2,2,2,2,2 at the byte peak (18); its 29 MoE layers see (1×4 + 28×2)/29,
// its 3 dense layers (all on chunk 0) see 4
T.check('P6 (MoE) = 60/29 on rank 0; P6d (dense) = 4; P10 = 18 chunk-stashes (a2a overlap)',
  Math.abs(c.get('P6') - 60 / 29) < 1e-12 && c.get('P6d') === 4 && c.get('P10') === 18 && c.get('P8') === 8,
  `${c.get('P6')} ${c.get('P6d')} ${c.get('P10')}`);
T.check('Megatron conventions from the page attrs: bf16 grad buffer, MXFP8 params resident in both orientations',
  c.byId.get('G1').label.includes('bf16') && c.get('F1') === 1 && c.byId.get('W2').expr.includes('1/32') && l().a2a === true, c.byId.get('G1').label);
T.check('activation formulas wear the per-kind factor (L1 × P6) / (L2 × P6d), never a trailing × P6',
  c.cells.some((x) => /^A\d/.test(x.id) && /\(L1 × P6\)/.test(x.expr ?? '')) && !c.cells.some((x) => /× P7 × P6/.test(x.expr ?? '')), '');
const barTxt = () => [...l().querySelectorAll('.lv-bar text')].map((t) => t.textContent).join('|');
T.check('fit chart: 276 GiB (GB300) cap, activations ×2.07mb, bf16 gradients, ruler ×2 (PP) ×32 (EP) ×256 (GPUs)',
  barTxt().includes('276 GiB (GB300)') && barTxt().includes('activations ×2.07mb') && barTxt().includes('gradients (bf16)')
  && barTxt().includes('×32 (EP)') && barTxt().includes('×256 (GPUs)'), barTxt().slice(0, 200));
// the bound strip: Megatron's program, drawn peak = the law on the selected rank
const strip = document.querySelector('dsv3-pp-schedule[layer]');
T.check('strip draws the interleaved program (header) with 2.25 mb = 18 chunks in flight on rank 0',
  strip.querySelector('.hd').textContent.includes('interleaved 1F1B (Megatron)')
  && strip.querySelector('text[data-peak]')?.textContent.startsWith('2.25 mb in flight (peak) = 18 chunks')
  && !strip.querySelector('text[data-peak]')?.textContent.includes('the model charges'), strip.querySelector('text[data-peak]')?.textContent);
// walk to rank 1: 2.0 mb = 16 chunks, no dense layers, the head
knob('rank').querySelector('button:last-child').click(); await T.tick(500);
// rank 1: 16 chunk-stashes at the count peak, but its BYTE peak is [4,2,2,2,2,2,2,0]
// (the last chunk is one layer + MTP + loss): its 29 MoE layers see (4×4 + 24×2)/29
T.check('rank 1: 16 chunks in flight, P6 = 64/29, L2 = 0, head resident', l().stage === 1
  && strip.querySelector('text[data-peak]')?.textContent.startsWith('2 mb in flight (peak) = 16 chunks')
  && Math.abs(l()._cells().get('P6') - 64 / 29) < 1e-12 && l()._cells().get('L2') === 0 && l()._cells().get('L5') === 1, strip.querySelector('text[data-peak]')?.textContent);
// the fold: PP2 × VP8 layout, chunks deal round-robin, rank totals = whole model at EP32
const f = document.querySelector('dsv3-pp-fold');
T.check('fold offers the 2 ranks', f.querySelector('button.cyc').textContent.includes('fold onto the 2 ranks'), '');
f.querySelector('button.cyc').click(); await T.tick(700);
const rt = [...f.querySelectorAll('text[data-ranktotal]')].map((t) => +t.dataset.params);
const whole = [...f.querySelectorAll('g[data-chunk]')].reduce((s, g) => s + +g.dataset.params, 0);
T.check('folded: two rank totals that sum to the chunk total', rt.length === 2 && Math.abs(rt[0] + rt[1] - whole) < 1, `${rt} vs ${whole}`);
T.check('rank 0 = the even chunks (emb side), rank 1 = the odd chunks (mtp + head)', f._rankChunks(0).map((k) => k.c).join() === '0,2,4,6,8,10,12,14'
  && f._rankChunks(1).some((k) => k.head && k.mtp), '');
// hardware knob: GB200 relabels the cap
knob('hw').querySelector('button').click(); await T.tick(500);
T.check('GB200: the cap line relabels to 184 GiB', l().hw === 'gb200' && barTxt().includes('184 GiB (GB200)'), '');
T.check('no Haziza button on a Megatron-family sheet', !document.querySelector('dsv3-sheet .hzb'), '');
T.check('deck has its steps', document.querySelectorAll('#fitdeck').length === 1 && document.querySelector('#fitdeck .deck-step').textContent.includes('/ 7'), '');
// the story's headline numbers (pinned so the prose can quote them)
const deck = document.getElementById('fitdeck'), DL = deck.querySelector('dsv3-layer');
const tot = () => DL.querySelector('.lv-bar text[data-role="val:total"]')?.textContent ?? '';
deck.querySelector('.deck-last').click(); await T.tick(1200);
T.check('GB300 config on a GB200: 176.8 GiB against 184', tot().startsWith('176.8 GiB') && DL.hw === 'gb200', tot());
deck.querySelector('.deck-prev').click(); await T.tick(1200);
T.check('no-pipeline hypothetical: 236.4 GiB on GB300', tot().startsWith('236.4 GiB') && DL.pp === 1, tot());
deck.querySelector('.deck-prev').click(); await T.tick(1200);
T.check('step 5 (the reference config): 176.8 GiB, MXFP8 params on, bf16 grads', tot().startsWith('176.8 GiB') && DL.fp8Params && DL.gradB === 2 && DL.a2a, tot());
const mlperf = [...document.querySelectorAll('dsv3-layer[snapshot][to]')].at(-1);
T.check('MLPerf PP4/VP4 snapshot: 146.2 GiB, ×4.31mb on rank 0', mlperf.querySelector('.lv-bar text[data-role="val:total"]')?.textContent.startsWith('146.2 GiB')
  && mlperf.textContent.includes('activations ×4.31mb'), mlperf.querySelector('.lv-bar text[data-role="val:total"]')?.textContent);
T.done();
