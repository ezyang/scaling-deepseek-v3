// @page studies/02-hopper-memory.html
// snapshot story beats: from = saved baseline (ghosts), to = live bars with
// badges, solo picks the row, zero interactivity, sandbox link loads the
// scenario into the full widget
const snap = () => document.querySelector('dsv3-layer[sandbox][to]');   // the ZeRO demo (the fit beat has sandbox but no to)
// the opening tally beats: weights · optim · acts · grads · everything
const beats = () => [...document.querySelectorAll('dsv3-layer[snapshot]:not([sandbox])')]
  .filter(b => !b.closest('dsv3-beat-deck'));   // the deck drives its own layer
T.check('tally beats render bars (the parts beat is optional — author-curated)', beats().length >= 4
  && beats().every(b => b.querySelector('.lv-bar svg') && !b.querySelector('.lv-scroll')), beats().length);
T.check('beat 1: whole-model weights 1.22 TiB + breakdown', beats()[0].textContent.includes('weights1.22 TiB')
  && beats()[0].textContent.includes('· experts'), '');
// beats found by attribute, not position — the author reorders them freely
const beat = (sel) => beats().find(b => (b.getAttribute('solo') ?? b.getAttribute('comps')) === sel)
  ?? beats().find(b => b.hasAttribute('parts'));
T.check('optim beat: soloed at 8 B/param, accordion open', beat('optim').textContent.includes('optimizer states4.88 TiB')
  && beat('optim').textContent.includes('· experts') && !/weights\d/.test(beat('optim').textContent), '');
T.check('acts beat: soloed, per-op buckets open', beat('acts').textContent.includes('activations ×1mb106.4 GiB')
  && beat('acts').textContent.includes('· dispatched tokens') && beat('acts').textContent.includes('· swiglu out')
  && !beat('acts').textContent.includes('· experts'), '');
T.check('grads beat: soloed at fp32, accordion open', beat('grads').textContent.includes('gradients (fp32)2.44 TiB')
  && beat('grads').textContent.includes('· experts'), '');
const tally = beats().find(b => b.hasAttribute('parts'));
if (tally) T.check('tally beat: all four components, ALL param accordions open', ['weights', 'gradients', 'optimizer', 'activations', 'total']
  .every(t => tally.textContent.includes(t))
  && [...tally.querySelectorAll('text')].filter(t => t.textContent === '· experts').length === 3, '');
else T.log('no parts tally beat on the page (author choice)', '');
// snapshots are figures: the card shrink-wraps (no full-width right slack)
T.check('cards shrink-wrap their chart', beats().every(b =>
  b.querySelector('.lv').getBoundingClientRect().width < 900), '');
T.check('no-to beats have no ghosts',
  beats().filter(b => !b.hasAttribute('to')).every(b => !b.querySelector('.lv-bar')?.textContent.includes('saved:')), '');
// intro beats drop the total (nototal); the tally beat lands the full mass
T.check('intro beats have no total row', beats().filter(b => b.hasAttribute('nototal')).every(b => !b.textContent.includes('total')), '');
if (tally) T.check('the tally beat totals 8.65 TiB', tally.textContent.includes('total8.65 TiB'), '');
const bar = () => snap().querySelector('.lv-bar');
const txt = () => bar().textContent;

T.check('snapshot renders the fit chart only', !!bar().querySelector('svg')
  && !snap().querySelector('.lv-scroll'), '');
T.check('no knobs anywhere', !snap().querySelector('.stp') && !snap().querySelector('.savebox'), '');
// snapshots are config-static but MEASURABLE: the drag ruler works…
const dragRuler = async (host, fx0, fx1) => {
  const scrub = host.querySelector('.scrub');
  const r = scrub.getBoundingClientRect();
  const y = r.top + r.height / 2;
  scrub.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, clientX: r.left + r.width * fx0, clientY: y }));
  document.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: r.left + r.width * fx1, clientY: y }));
  document.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, clientX: r.left + r.width * fx1, clientY: y }));
  await T.tick(80);
};
await dragRuler(snap(), 0.2, 0.5);
const rul = snap().querySelector('.lv-ruler');
T.check('snapshot: drag ruler measures (read-only)', getComputedStyle(rul).display === 'block'
  && /×[\d.]+/.test(rul.textContent), rul.textContent);
// …a drag on ANOTHER chart dismisses this one's ruler (several per page)
await dragRuler(document.querySelector('dsv3-layer[snapshot]'), 0.3, 0.6);
T.check('another chart\'s drag dismisses the first ruler', getComputedStyle(rul).display === 'none', '');
// …but mutating clicks do nothing: the gutter legend is dead in a snapshot
{
  const b5 = [...document.querySelectorAll('dsv3-layer[snapshot]')].at(-1);
  const P5 = ['showWeights', 'showGrads', 'showOptim', 'showActs'];
  const before = P5.map((k) => b5[k]);
  b5.querySelector('.lv-bar [data-prop]').dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
  await T.tick(300);
  T.check('snapshot: solo click does nothing', P5.every((k, i) => b5[k] === before[i]), '');
}
T.check('baseline label names the from config', txt().includes('saved: EP64·PP16') && txt().includes('ZeRO-off'), '');
// zero-1 shards optimizer over DP=128: a bold ▼×128 badge on the optim row
T.check('optimizer shrink badge ▼×128', txt().includes('▼×128'), txt().slice(0, 160));
T.check('ghost bars drawn', bar().querySelectorAll('rect[stroke-dasharray]').length > 0, '');
// solo="optim": the off components' rows are GONE (a dimmed unclickable
// name is a dead affordance in a figure); total keeps the full mass
T.check('solo: no weights/grads/acts rows at all', !txt().includes('weights')
  && !txt().includes('gradients') && txt().includes('optimizer states'), '');
T.check('total row present', txt().includes('total'), '');
// snapshots keep no URL state
T.check('no snapshot URL state', !location.hash.includes('l:layer'), location.hash);
// the sandbox link loads the scenario into the full widget
const link = [...snap().querySelectorAll('a')].find(a => a.textContent.includes('full widget'));
T.check('sandbox link present', !!link, '');
const full = document.getElementById('local-diagram');
link.click(); await T.tick(600);
T.check('full widget took the scenario', full.zero === 1 && full.pp === 16 && full.ep === 64, `${full.zero}/${full.pp}/${full.ep}`);
T.check('full widget carries the baseline save (blended optim badge)', full._pinCfg?.state?.zero === 0
  && (full.querySelector('.lv-bar')?.textContent ?? '').includes('▼×4.6'), full._pinCfg?.state?.zero);
T.done();
