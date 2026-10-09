// @page studies/kimi-k3.html
// Kimi K3 architecture diagram: tab and plan kind flips, sizes toggle (the
// exact 2.78T / 104.2B anchors are pinned by tests/kimisheet.js).
await T.tick(400);

T.check('diagram mounted', !!T.el('#diagram svg'));
T.check('plan mounted', !!T.el('kimi-anatomy-plan svg'));

// attention/FFN tabs flip independently and re-render
const d = T.el('#diagram');
T.click('#diagram [data-tab="a:mla"]');
await T.tick(120);
T.check('mla tab flip', d.kindA === 'mla' && !!T.el('#diagram [data-op="q_down"]'));
T.check('kda gone under mla', !T.el('#diagram [data-op="kda_scan"]'));
T.click('#diagram [data-tab="f:dense"]');
await T.tick(120);
T.check('dense tab flip', d.kindF === 'dense' && !!T.el('#diagram [data-op="ffn_gate_up_d"]'));
T.check('router gone under dense', !T.el('#diagram [data-op="router"]'));

// plan blocks drive the attention kind
T.click('kimi-anatomy-plan [data-kind="kda"]');
await T.tick(120);
T.check('plan flips to kda', d.kindA === 'kda' && !!T.el('#diagram [data-op="kda_scan"]'));
T.click('#diagram [data-tab="f:moe"]');
await T.tick(120);

// sizes toggle: factored ↔ multiplied-out dims on the fused qkv
const qkvDims = () => T.el('#diagram [data-op="qkv"] .dims')?.textContent ?? '';
T.check('factored dims', qkvDims().includes('3×96×128'), qkvDims());
T.click('#diagram .kv-head button');
await T.tick(120);
T.check('flat dims', qkvDims().includes('36864'), qkvDims());
T.click('#diagram .kv-head button');
await T.tick(120);

T.done();
