// Study helper: cross-check tables comparing our model's numbers against an
// external reference (a spreadsheet, a paper, a real trace). Studies live in
// studies/*.html — essay-style sections we workshop before wiring into the essay.

export function crossCheck(el, title, rows) {
  // rows: [{name, expected, actual, unit?, tolPct?=2, note?}]
  const fmt = (v) => typeof v === 'number'
    ? (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(3))
    : String(v);
  const html = [`<table class="xcheck"><tr><th>${title}</th><th>reference</th><th>ours</th><th>Δ</th><th></th></tr>`];
  let fails = 0;
  for (const r of rows) {
    const tol = (r.tolPct ?? 2) / 100;
    const delta = r.expected ? (r.actual - r.expected) / r.expected : 0;
    const ok = r.info || (r.lte ? r.actual <= r.expected : Math.abs(delta) <= tol);
    if (!ok) fails++;
    html.push(`<tr><td>${r.name}${r.note ? `<br><span class="note">${r.note}</span>` : ''}</td>` +
      `<td>${fmt(r.expected)}${r.unit ? ' ' + r.unit : ''}</td>` +
      `<td>${fmt(r.actual)}${r.unit ? ' ' + r.unit : ''}</td>` +
      `<td style="color:${r.info ? '#52514e' : ok ? '#006300' : '#d03b3b'}">${(delta * 100).toFixed(1)}%</td>` +
      `<td>${r.info ? 'ⓘ' : ok ? '✓' : '✗'}</td></tr>`);
  }
  html.push('</table>');
  el.innerHTML = html.join('');
  el.dataset.fails = fails;
  return fails;
}
