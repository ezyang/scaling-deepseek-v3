// @page studies/kimi-k3.html
// Kimi K3 architecture diagram: tab and plan kind flips, sizes toggle (the
// exact 2.78T / 104.2B anchors are pinned by tests/kimisheet.js); the
// recompute-marks instance: the K3 preset reproduces the sheet's policy-A
// stash exactly, presets recognize themselves, clicks flip save ↔ ↻; the
// AttnRes pipeline figure: transition widths (arXiv 2603.15031 §4.1), knobs,
// and the sheet's r_res default = the figure's busiest rank.
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

// ---- the figure drives the sheet's policy A ---------------------------------
const cellQ = (nm) => [...document.querySelectorAll('.cellsheet tr')].find((tr) => tr.querySelector('td.nm')?.innerHTML === nm);
const cellV = (nm) => Number(cellQ(nm)?.cells[2].textContent.replace(/,/g, ''));
const head6V = 6 * 163840;
T.check('published σᴬ = the graph under K3 (+ the head)', cellV('σ<sup>A</sup>') === m.k3Stash(m.K3_RECOMPUTE.k3) + head6V, cellV('σ<sup>A</sup>'));
for (const p of ['none', 'full']) {
  T.click(`#ac-layer [data-pol="${p}"]`);
  await T.tick(250);
  T.check(`${p}: σᴬ follows the figure exactly`, cellV('σ<sup>A</sup>') === m.k3Stash(m.K3_RECOMPUTE[p]) + head6V, `${cellV('σ<sup>A</sup>')} vs ${m.k3Stash(m.K3_RECOMPUTE[p]) + head6V}`);
}
T.check('full: both all-to-alls replay, the GEMM replays priced', cellV('δ<sub>comb</sub>') === 1 && cellV('c<sup>A</sup><sub>rp8</sub>') > 2e11
  && cellQ('T<sup>A</sup><sub>c</sub>')?.classList.contains('off'));
T.click('#ac-layer [data-pol="k3"]');
await T.tick(250);
T.check('back to K3: no sheet cell off its published value', !document.querySelector('.cellsheet tr.off'), document.querySelector('.cellsheet tr.off td.nm')?.innerHTML);

// ---- AttnRes across the pipeline (#pipe) -----------------------------------
const chain6 = m.attnresPipe(6, 'chain');
T.check('1F1B PP 6: widths 3 4 5 7 8 (blocks + the partial sum)', chain6.tr.map((t) => t.w).join(' ') === '3 4 5 7 8', chain6.tr.map((t) => t.w).join(' '));
const v6 = m.attnresPipe(6, 'v');
T.check('V PP 6: the turn sends nothing, the cache trims the return', v6.tr[5].w === 0 && v6.tr[6].w === 2 && v6.tr[6].naive === 6, JSON.stringify(v6.tr[6]));
T.check('V PP 6: busiest rank 19 d, every rank holds ≤ 8 blocks', v6.rres === 19 && Math.max(...v6.held) === 8 && v6.held[0] === 8);
T.check('no pipeline, no traffic', m.attnresPipe(1, 'v').rres === 0 && m.attnresPipe(1, 'chain').tr.length === 0);
const rresCell = [...document.querySelectorAll('.cellsheet tr')].find((tr) => tr.querySelector('td.nm')?.innerHTML === 'r<sub>res</sub>');
T.check('sheet r_res default = the figure (V, PP 6)', rresCell?.cells[2].textContent.trim() === String(v6.rres), rresCell?.cells[2].textContent);
const pipe = T.el('#pipe');
T.check('figure: busiest rank in the ledger', T.text('#pipe .kpp-led b').startsWith('busiest rank (1): 19 d'), T.text('#pipe .kpp-led b'));
const h0 = pipe.getBoundingClientRect().height;
T.click('#pipe [data-s="chain"]');
await T.tick(80);
T.click('#pipe [data-pp="24"]');
await T.tick(80);
T.check('knobs: 1F1B PP 24 = 18 d, height unchanged', pipe.sched === 'chain' && pipe.pp === 24 && T.text('#pipe .kpp-led b').includes('18 d')
  && Math.abs(pipe.getBoundingClientRect().height - h0) < 1, `${T.text('#pipe .kpp-led b')} ${h0} ${pipe.getBoundingClientRect().height}`);
T.click('#pipe [data-s="v"]');
T.click('#pipe [data-pp="6"]');
await T.tick(80);

T.done();
