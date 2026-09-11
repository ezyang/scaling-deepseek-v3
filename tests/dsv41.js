// @page studies/dsv41-flash.html
// DeepSeek-V4.1-Flash study: exact parameter anchors (header-derived), the
// five CSA2 tabs, plan-driven kind flips, the three tally modes, sizes toggle.
await T.tick(400);

T.check('diagram mounted', !!T.el('#diagram svg'));
T.check('plan mounted', !!T.el('dsv41-anatomy-plan svg'));

const tal = 'dsv41-param-tally';
const bb = () => T.el(`${tal} tfoot tr.bb .pnum`)?.dataset.v;
const withEng = () => T.el(`${tal} tfoot tr.with .pnum`)?.dataset.v;
T.check('backbone exact', bb() === '551566180464', bb());
T.check('backbone shown as B', T.text(`${tal} tfoot tr.bb .pnum`) === '551.6B');
T.check('with Engram exact', withEng() === '748494684784', withEng());
T.click(`${tal} .mbtn[data-mode="decode"]`);
await T.tick(150);
T.check('decode active exact', bb() === '16130588784', bb());
T.check('decode flips diagram to active experts', (T.el('#params-diagram [data-op="gate_up"] .dims')?.textContent ?? '').includes('×6'));
T.click(`${tal} .mbtn[data-mode="prefill"]`);
await T.tick(150);
T.check('prefill active exact', bb() === '7895563064', bb());
T.click(`${tal} .mbtn[data-mode="total"]`);
await T.tick(150);
T.check('total restored', bb() === '551566180464', bb());

// tabs flip the attention kind and re-render the right machinery
const d = T.el('#diagram');
T.check('default kind full2', d.kind === 'full2' && !!T.el('#diagram [data-op="comp_kv"]') && !!T.el('#diagram [data-op="idx_k"]'));
T.click('#diagram [data-tab="reindex"]');
await T.tick(120);
T.check('reindex: indexer, no compressor', d.kind === 'reindex' && !!T.el('#diagram [data-op="idx_q"]') && !T.el('#diagram [data-op="comp_kv"]'));
T.click('#diagram [data-tab="reuse"]');
await T.tick(120);
T.check('reuse: no indexer', d.kind === 'reuse' && !T.el('#diagram [data-op="idx_q"]') && !!T.el('#diagram [data-op="attn"]'));
T.click('#diagram [data-tab="swa"]');
await T.tick(120);
T.check('swa: window only', d.kind === 'swa' && (T.el('#diagram [data-op="attn"] .dims')?.textContent ?? '').startsWith('window 128 · sink'));
T.click('#diagram [data-tab="full1"]');
await T.tick(120);
T.check('full1: plain projection, no gate', d.kind === 'full1' && (T.el('#diagram [data-op="comp_kv"] .name')?.textContent ?? '') === 'global kv proj');

// plan blocks drive the kind
T.click('dsv41-anatomy-plan [data-kind="full2"]');
await T.tick(120);
T.check('plan flips to full2', d.kind === 'full2' && (T.el('#diagram [data-op="comp_kv"] .name')?.textContent ?? '').startsWith('compress'));

// sizes toggle: factored ↔ multiplied-out dims on q up-proj
const qDims = () => T.el('#diagram [data-op="q_up"] .dims')?.textContent ?? '';
T.check('factored dims', qDims().includes('64×512'), qDims());
T.click('#diagram .kv-head button');
await T.tick(120);
T.check('flat dims', qDims().includes('32768'), qDims());
T.click('#diagram .kv-head button');
await T.tick(120);

// tally hover highlights the row's cells in the params diagram + plan
T.hover(`${tal} tbody tr[data-row="2"]`);
await T.tick(120);
T.check('hover fades diagram', T.el('#params-diagram svg')?.classList.contains('hlm'));
T.check('hover marks compressor', T.el('#params-diagram [data-op="comp_kv"]')?.classList.contains('hl'));
T.check('hover marks plan block',
  document.querySelector('dsv41-anatomy-plan[layer="params-diagram"] [data-op="block-full2"]')?.classList.contains('hl'));
T.unhover(`${tal} tbody tr[data-row="2"]`);
await T.tick(120);
T.check('unhover clears', !T.el('#params-diagram svg')?.classList.contains('hlm'));

// pinning a decoder row flips the params diagram's kind
T.click(`${tal} tbody tr[data-row="5"]`);
await T.tick(120);
T.check('pin flips params diagram to reindex', T.el('#params-diagram').kind === 'reindex');

T.done();
