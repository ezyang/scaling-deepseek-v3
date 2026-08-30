// @page studies/02-hopper-memory.html
// the page-wide audit overlay (hidden debug mode): Alt+A decorates EVERY fit
// chart with a verification chip, driven by the same auditFitCharts the
// battery runs; chips re-derive after changes settle, never mid-tween red
const altA = () => document.dispatchEvent(new KeyboardEvent('keydown',
  { code: 'KeyA', altKey: true, bubbles: true }));
const chips = () => [...document.querySelectorAll('.aud-chip')];

T.check('hidden by default', chips().length === 0, '');
altA(); await T.tick(300);
T.check('Alt+A decorates every chart on the page', chips().length >= 6, chips().length);
T.check('all chips verify (✓, none red)', chips().every((c) => c.textContent.startsWith('✓')
  && !c.classList.contains('bad')), chips().map((c) => c.textContent).join('|'));
T.check('mode persists in the URL hash', location.hash.includes('audit'), location.hash);
// click a chip → the report drops as an OVERLAY (no reflow anywhere)
const layer = document.getElementById('local-diagram');
const slot = () => layer.querySelector('.lv-bar');   // re-renders replace the node
const pageH = document.body.scrollHeight;
slot().querySelector('.aud-chip').click(); await T.tick(50);
const panel = slot().querySelector('.aud-panel');
T.check('report panel opens on the chart', panel.classList.contains('open')
  && panel.textContent.includes('implications verified'), '');
T.check('overlay never reflows the page', document.body.scrollHeight === pageH,
  `${pageH} -> ${document.body.scrollHeight}`);
// hover a line → its pattern stays lit, the rest dims
const ln = [...panel.querySelectorAll('.aud-ln')].find((l) => l.dataset.sel?.startsWith('[data-role="val:'));
ln.dispatchEvent(new MouseEvent('mouseover', { bubbles: true })); await T.tick(50);
T.check('hover dims everything but the verified pattern',
  [...slot().querySelectorAll('svg [data-role]')].filter((n) => n.style.opacity === '0.2').length > 5, '');
ln.dispatchEvent(new MouseEvent('mouseout', { bubbles: true })); await T.tick(50);
// save + knob change: the chart re-renders wholesale, ghosts and badges
// appear — the chip re-derives once the page settles (an open report
// survives) and stays green
[...layer.querySelector('.savebox').querySelectorAll('button')].find((b) => b.textContent === 'save').click();
await T.tick(500);
layer.parentElement.querySelector('.stp[data-knob="ep"]').querySelectorAll('button')[0].click();
await T.tick(1200);   // tween + the observer's quiet period
const chip2 = slot().querySelector('.aud-chip');
T.check('chip re-derives after the change, still green', chip2 && chip2.textContent.startsWith('✓')
  && !chip2.classList.contains('bad'), chip2?.textContent);
T.check('open report survives the re-audit', slot().querySelector('.aud-panel.open')
  && slot().querySelector('.aud-panel').textContent.includes('baseline'), '');
[...layer.querySelector('.savebox').querySelectorAll('button')].find((b) => b.textContent === 'reset all').click();
await T.tick(1200);
// the deck's chart is covered too
T.check('deck chart wears a chip', !!document.querySelector('#fitdeck .aud-chip'), '');
altA(); await T.tick(100);
T.check('Alt+A off removes everything and clears the hash', chips().length === 0
  && !location.hash.includes('audit'), location.hash);
T.done();
