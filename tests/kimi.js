// @page studies/kimi-k3.html
// Kimi K3 architecture diagram: tab and plan kind flips, sizes toggle (the
// exact 2.78T / 104.2B anchors are pinned by tests/kimisheet.js); the
// recompute-marks instance: the K3 preset reproduces the sheet's policy-A
// stash exactly, presets recognize themselves, clicks flip save ↔ ↻.
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


// ---- recompute marks (#ac-layer) ------------------------------------------
const m = await import('/src/kimi.js');
const { analyze } = await import('/src/blockgraph.js');
// the sheet's policy-A rows, restated (studies/kimi-k3.html stashA), + the
// decay bottleneck's rank-128 BF16 intermediate the sheet drops (KDA only)
const b8 = 1 + 1 / 32, dd = 7168, P = 12288, nh = 96, E = 896, k = 16, I = 3072, l = 3584, Id = 33792;
const sres = 4 * dd, sAk = 5 * P * b8 + nh * b8 + 2 * P + 256;
const sAm = (1536 + 512 + 64) * b8 + 2 * nh * (2 * (128 + 64) + 2 * 128) + 4 * nh + P * b8;
const sAe = 4 * E + l * b8 + k * 2 * I * b8 + l * b8 + 4 * k + 2 * I * 2 * b8, sAd = 2 * Id * b8;
const want = { 'kda/moe': sres + sAk + sAe, 'mla/moe': sres + sAm + sAe, 'kda/dense': sres + sAk + sAd };
for (const [key, v] of Object.entries(want)) {
  const [a, f] = key.split('/');
  const got = analyze(m.kimiGraph(a, f), m.K3_RECOMPUTE.k3).savedBytes;
  T.check(`K3 preset = sheet policy A (${key})`, Math.abs(got - v) < 1e-6, `${got} vs ${v}`);
}
T.check('full recompute keeps only x0', analyze(m.kimiGraph('mla', 'moe'), m.K3_RECOMPUTE.full).savedBytes === 2 * dd);
const ac = T.el('#ac-layer');
T.check('marks instance mounted on K3', ac?.policy() === 'k3' && !!T.el('#ac-layer [data-pol="k3"].on'));
T.check('conv recomputes, qkv stashed',
  T.el('#ac-layer [data-op="conv"]')?.classList.contains('rpl') && !T.el('#ac-layer [data-op="qkv"]')?.classList.contains('rpl'));
T.check('ledger total', T.text('#ac-layer .kv-led b').startsWith('stashed for backward: 236.4 KiB'), T.text('#ac-layer .kv-led b'));
// flip conv to saved: its output (the kernel's q, k, v) now stashes, custom
T.click('#ac-layer [data-op="conv"]');
await T.tick(100);
T.check('click flips to saved', ac.marks.conv === true && ac.policy() === null && T.text('#ac-layer .cust').startsWith('custom'));
T.check('saved conv stashes its output', ac._ana.neededSaved.has('conv'));
T.click('#ac-layer [data-op="conv"]');
await T.tick(100);
T.check('flip back recognizes K3', ac.policy() === 'k3');
T.click('#ac-layer [data-pol="none"]');
await T.tick(100);
T.check('none preset: nothing replays', ac.policy() === 'none' && ac._ana.replayed.size === 0);
T.click('#ac-layer [data-pol="k3"]');
await T.tick(100);

T.done();
