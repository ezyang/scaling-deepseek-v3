// @page studies/02-hopper-memory.html
// visual audit: every rendered fit-chart number re-derives from its exact
// value — rounding, badge factors, sums, and log-axis geometry
const { auditFitCharts } = await import('/src/audit.js');
const { charts, findings } = auditFitCharts(document);
T.log('charts audited', charts);
T.check('audits every chart on 02 (beats + zero beat + full sim)', charts >= 6, charts);
T.check('zero findings', findings.length === 0, findings.slice(0, 4).join(' | '));
// not vacuous: numbers and geometry were actually present
T.check('exact values present', document.querySelectorAll('.lv-bar text[data-true]').length > 30, '');
T.check('badges audited (the ZeRO beat has pins)',
  [...document.querySelectorAll('.lv-bar text[data-pin]')].some(t => t.dataset.pin !== ''), '');
// the audit CATCHES lies: corrupt an exact value (rounding + bar-edge fire),
// then a sub-row value (the decomposition-sum implication fires)
const v = document.querySelector('.lv-bar text[data-role^="val:"]');
const orig = v.dataset.true;
v.dataset.true = String(Math.round(+v.dataset.true * 3));
T.check('a corrupted value is caught', auditFitCharts(document).findings.length > 0, '');
v.dataset.true = orig;
const p2 = document.querySelector('.lv-bar text[data-role^="val:part:"]');
p2.dataset.true = String(+p2.dataset.true + 4096);
const f2 = auditFitCharts(document).findings;
T.check('a decomposition lie is caught', f2.some(m => m.includes('decomposition')), f2[0]);
p2.dataset.true = String(+p2.dataset.true - 4096);
// a distances-ruler tick claiming the wrong factor (drawn-at-×2, claims ×3)
const tk = document.querySelector('.lv-bar line[data-fac="2"]');
tk.dataset.fac = '3';
const f3 = auditFitCharts(document).findings;
T.check('a distances-ruler lie is caught', f3.some(m => m.includes('distances')), f3[0]);
tk.dataset.fac = '2';
T.done();
