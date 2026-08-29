// @page studies/scratch-bars.html
// visual audit after INTERACTIONS: save a baseline, turn knobs, then audit
// the badges/ghosts/geometry the interactive state produced
const { auditFitCharts } = await import('/src/audit.js');
const L = () => document.getElementById('bars-all');
const stp = (k) => L().querySelector(`.stp[data-knob="${k}"]`);

T.check('clean page audits clean', auditFitCharts(document).findings.length === 0,
  auditFitCharts(document).findings[0]);
// save, then change EP and PP: ghosts + badges everywhere
[...L().querySelectorAll('.savebox button')].find(b => b.textContent === 'save').click(); await T.tick(100);
stp('ep').querySelectorAll('button')[0].click(); await T.tick(500);
stp('pp').querySelectorAll('button')[1].click(); await T.tick(700);
const a = auditFitCharts(document);
T.check('pinned + mutated state audits clean', a.findings.length === 0, a.findings.slice(0, 3).join(' | '));
T.check('ghosts were actually audited', L().querySelectorAll('.lv-bar rect[data-ghost]').length > 0, '');
// solo: accordion parts audited too
L().querySelector('.lv-bar [data-prop]').dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
await T.tick(400);
const b = auditFitCharts(document);
T.check('soloed accordion audits clean', b.findings.length === 0, b.findings.slice(0, 3).join(' | '));
T.done();
