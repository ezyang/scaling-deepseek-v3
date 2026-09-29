// @page studies/03-roofline.html
// <dsv3-anatomy experttint>: the routed-expert GEMMs wear the olive expert
// tint, every other parameter-carrying box the slate one (param-less boxes —
// attention, RoPE, the a2a — stay plain); the dense tab has no experts; the
// tally's split rows are N_X and N_O exactly (N_O = the FSDP sheet's row)
const a = document.querySelector('dsv3-anatomy[experttint]');
const q = (s) => a.querySelector(s), qa = (s) => [...a.querySelectorAll(s)];
const cls = (c) => qa(`.lv g[data-op] > rect.${c}`).map((r) => r.parentElement.dataset.op);
T.check('MoE: the two routed-expert GEMMs are the expert boxes', cls('xpt').sort().join() === 'ffn_down,ffn_gate_up', cls('xpt').join());
const nx = new Set(cls('nxp'));
T.check('the key names both classes', /routed experts · non-expert parameters/.test(q('[data-xt-legend]')?.textContent ?? ''), q('[data-xt-legend]')?.textContent);
T.check('MoE: attention, norms, router + bias, shared expert are non-expert', ['qkv_down', 'q_up', 'kv_up', 'o_proj', 'q_norm', 'kv_norm', 'norm1', 'norm2', 'router', 'router_bias', 'shared'].every((id) => nx.has(id)), [...nx].join());
T.check('param-less boxes stay plain', !nx.has('attn') && !qa('.lv g[data-op=attn] > rect.xpt').length, '');
T.check('plan: embedding, dense block, final norm, lm head non-expert; MoE block not', ['embed', 'final_norm', 'lm_head', 'block-dense'].every((id) => q(`dsv3-anatomy-plan g[data-op="${id}"] > rect.nxp`)) && !q('dsv3-anatomy-plan g[data-op="block-moe"] > rect.nxp'), '');
const sv = (c) => +q(`tr[data-split=${c}] .pnum`).dataset.v, tot = +q('tfoot tr:first-child .pnum').dataset.v;
T.check('tally split: routed experts = N_X exactly', sv('xpt') === 653908770816, sv('xpt'));
T.check('tally split: everything else = the sheet\'s N_O', sv('nxp') === 17117648384 && sv('xpt') + sv('nxp') === tot, sv('nxp'));
q('dsv3-anatomy-plan g[data-kind=dense]').dispatchEvent(new MouseEvent('click', { bubbles: true }));
await T.tick(400);
T.check('dense tab: no expert boxes, its FFN is non-expert', !cls('xpt').length && cls('nxp').includes('ffn_gate_up') && cls('nxp').includes('ffn_down'), cls('xpt').join());
T.check('dense tab: the selected dense block and the mixed MoE block stay untinted', !q('dsv3-anatomy-plan g[data-op="block-moe"] > rect.nxp') && !q('dsv3-anatomy-plan g[data-op="block-dense"] > rect.nxp'), '');
T.done();
