// @page studies/kimi-k3.html
// Kimi K3 study: exact parameter anchors (config-derived; reconciles with the
// advertised 2.78T / 104.2B), tab and plan kind flips, sizes toggle.
await T.tick(400);

T.check('diagram mounted', !!T.el('#diagram svg'));
T.check('plan mounted', !!T.el('kimi-anatomy-plan svg'));

// exact anchors — the tally's data-v carries the full integers
const tal = 'kimi-param-tally';
T.check('total exact', T.el(`${tal} tfoot .pnum`)?.dataset.v === '2779484476000',
  T.el(`${tal} tfoot .pnum`)?.dataset.v);
T.check('total shown as T', T.text(`${tal} tfoot .pnum`) === '2.78T');
T.click(`${tal} .mbtn[data-mode="active"]`);
await T.tick(150);
T.check('active exact', T.el(`${tal} tfoot .pnum`)?.dataset.v === '104189612640',
  T.el(`${tal} tfoot .pnum`)?.dataset.v);
T.click(`${tal} .mbtn[data-mode="total"]`);
await T.tick(150);

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

// tally hover highlights the row's cells in the params diagram + plan
T.hover(`${tal} tbody tr[data-row="2"]`);
await T.tick(120);
T.check('hover fades diagram', T.el('#params-diagram svg')?.classList.contains('hlm'));
T.check('hover marks experts', T.el('#params-diagram [data-op="gate_up"]')?.classList.contains('hl'));
T.check('hover marks plan block',
  document.querySelector('kimi-anatomy-plan[layer="params-diagram"] [data-op="block-kda"]')?.classList.contains('hl'));
T.unhover(`${tal} tbody tr[data-row="2"]`);
await T.tick(120);
T.check('unhover clears', !T.el('#params-diagram svg')?.classList.contains('hlm'));

T.done();
