// @page studies/02-hopper-memory.html
// visual audit: every rendered fit-chart number re-derives from its exact
// value — rounding, badge factors, sums, and log-axis geometry
const { auditFitCharts } = await import('/src/audit.js');
const { charts, findings } = auditFitCharts(document);
T.log('charts audited', charts);
T.check('audits every chart on 02 (5 beats + zero beat + full sim)', charts >= 7, charts);
T.check('zero findings', findings.length === 0, findings.slice(0, 4).join(' | '));
// not vacuous: numbers and geometry were actually present
T.check('exact values present', document.querySelectorAll('.lv-bar text[data-true]').length > 30, '');
T.check('badges audited (the ZeRO beat has pins)',
  [...document.querySelectorAll('.lv-bar text[data-pin]')].some(t => t.dataset.pin !== ''), '');
// the audit CATCHES a lie: corrupt one exact value and re-run
const v = document.querySelector('.lv-bar text[data-role^="val:"]');
v.dataset.true = String(Math.round(+v.dataset.true * 3));
T.check('a corrupted value is caught', auditFitCharts(document).findings.length > 0, '');
T.done();
