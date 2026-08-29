// @page studies/02-hopper-memory.html
// snapshot story beats: from = saved baseline (ghosts), to = live bars with
// badges, solo picks the row, zero interactivity, sandbox link loads the
// scenario into the full widget
const snap = () => document.querySelector('dsv3-layer[sandbox]');   // the ZeRO beat
// the opening tally beats: weights · optim · acts · grads · everything
const beats = () => [...document.querySelectorAll('dsv3-layer[snapshot]:not([sandbox])')];
T.check('five tally beats render bars', beats().length === 5
  && beats().every(b => b.querySelector('.lv-bar svg') && !b.querySelector('.lv-scroll')), beats().length);
T.check('beat 1: whole-model weights 1.22 TiB + breakdown', beats()[0].textContent.includes('weights1.22 TiB')
  && beats()[0].textContent.includes('· experts'), '');
T.check('beat 2: optimizer soloed at 8 B/param, accordion open', beats()[1].textContent.includes('optimizer states4.88 TiB')
  && beats()[1].textContent.includes('· experts') && !/weights\d/.test(beats()[1].textContent), '');
T.check('beat 3: activations soloed (no param accordion)', beats()[2].textContent.includes('activations ×1mb106.4 GiB')
  && !beats()[2].textContent.includes('· experts'), '');
T.check('beat 4: gradients soloed at fp32, accordion open', beats()[3].textContent.includes('gradients (fp32)2.44 TiB')
  && beats()[3].textContent.includes('· experts'), '');
T.check('beat 5: all four components, ALL param accordions open', ['weights', 'gradients', 'optimizer', 'activations', 'total']
  .every(t => beats()[4].textContent.includes(t))
  && [...beats()[4].querySelectorAll('text')].filter(t => t.textContent === '· experts').length === 3, '');
// snapshots are figures: the card shrink-wraps (no full-width right slack)
T.check('cards shrink-wrap their chart', beats().every(b =>
  b.querySelector('.lv').getBoundingClientRect().width < 900), '');
T.check('beats have no ghosts (no to= no baseline)',
  beats().every(b => !b.querySelector('.lv-bar')?.textContent.includes('saved:')), '');
// the total is ALWAYS the full consolidated mass (stacked grey + shown):
// context for what hasn't been introduced yet
T.check('every beat totals 8.65 TiB',
  beats().every(b => b.textContent.includes('total8.65 TiB')), '');
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
  const beat5 = document.querySelectorAll('dsv3-layer[snapshot]')[4];
  const gut = beat5.querySelector('.lv-bar [data-prop]');
  gut.dispatchEvent(new MouseEvent('mousedown', { bubbles: true })); await T.tick(300);
  T.check('snapshot: solo click does nothing', beat5.showWeights && beat5.showGrads && beat5.showOptim && beat5.showActs, '');
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
