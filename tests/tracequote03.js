// @page studies/03-roofline.html
// prose numbers quoting the kernel table's trace column (data-trace spans) are
// filled from the table's data-us at load; the HTML's text is only the fallback
// and must match, so the prose can't go stale against the table
await T.tick(300);
const raw = new DOMParser().parseFromString(await (await fetch(location.pathname)).text(), 'text/html');
const live = [...document.querySelectorAll('main [data-trace]')], pub = [...raw.querySelectorAll('main [data-trace]')];
const stale = pub.flatMap((q, i) => q.textContent !== live[i].textContent ? [`${q.dataset.trace}/${q.dataset.col}: ${q.textContent} → ${live[i].textContent}`] : []);
T.check('four trace quotes, filled', live.length === 4 && live.every((q) => /^\d+\.\d ms$/.test(q.textContent)), live.map((q) => q.textContent).join(' | '));
T.check('the HTML\'s fallback text is what the table gives', stale.length === 0, stale.join(' | '));
T.done();
