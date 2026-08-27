// @page studies/02-hopper-memory.html
// sub-part filter: click 'experts' inside a solo — diagram filters, the
// soloed row stacks, ghosts render dotted
const layer = () => document.getElementById('local-diagram');
const chart = () => layer().querySelector('.lv-bar');
const md = (el2) => el2.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
const rowLabels = () => [...layer().querySelectorAll('.lv-bar g[data-prop]')];
const dims = () => layer().querySelector('g[data-op="ffn_gate_up"] text.dims:not([text-anchor])')?.textContent.trim();
const oproj = () => layer().querySelector('g[data-op="o_proj"] text.dims:not([text-anchor])')?.textContent.trim();
md(rowLabels()[0]); await T.tick(700);   // solo weights
const partRow = () => [...chart().querySelectorAll('g[data-part]')].find(g => g.textContent.includes('experts'));
T.check('part rows clickable', !!partRow(), '');
md(partRow()); await T.tick(700);        // select 'experts'
T.check('expert op keeps its number', (dims() ?? '').includes('GiB') || (dims() ?? '').includes('MiB'), dims());
T.check('non-expert op number gone', !oproj(), oproj());
T.check('soloed row stacks grey+color', !!chart().querySelector('rect[fill="#c3c2b7"]'), '');
const blue = [...layer().querySelectorAll('.lv-scroll rect[fill="#2a78d6"]')].length;
T.check('diagram squares only on expert ops', blue > 0 && blue < 8, blue);
// save -> ghost is a dashed rect, not a tick line
[...layer().querySelector('.savebox').querySelectorAll('button')].find(b => b.textContent === 'save').click(); await T.tick();
const stps = () => [...layer().parentElement.querySelectorAll('.stp')];
stps()[3].querySelectorAll('button')[0].click(); await T.tick(700);   // EP -
T.check('dashed ghost bars render', [...chart().querySelectorAll('rect[stroke-dasharray="2 2"]')].length >= 2, [...chart().querySelectorAll('rect[stroke-dasharray="2 2"]')].length);
T.check('no tick lines remain', ![...chart().querySelectorAll('line')].some(l2 => l2.getAttribute('stroke') === '#0b0b0b' && +l2.getAttribute('stroke-width') >= 1.4), '');
// part click again -> unfilter; unsolo; reset all
md(partRow()); await T.tick(700);
T.check('unfilter restores non-expert number', !!oproj(), oproj());
[...layer().querySelector('.savebox').querySelectorAll('button')].find(b => b.textContent === 'reset all').click(); await T.tick(400);
T.done();
