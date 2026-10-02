// @page studies/03-roofline.html
// <dsv3-layer dtypetint>: every GEMM box wears its compute dtype — under
// dsv3-fp8 the head, router and attention core BF16, everything else E4M3
// (attn out-proj stashes E5M6 but computes E4M3); each box's readout matches
await T.tick(100);
const a = document.querySelector('#dtype-block');
const cls = (c) => [...a.querySelectorAll(`.lv g[data-op] > rect.${c}`)].map((r) => r.parentElement.dataset.op).sort().join(' ');
T.check('bf16 boxes: attention core, lm head, router', cls('dt-bf16') === 'attn lm_head router', cls('dt-bf16'));
T.check('e4m3 boxes: the MLA projections, attn out-proj, expert GEMMs', cls('dt-e4m3') === 'ffn_down ffn_gate_up kv_up o_proj q_up qkv_down qkv_down', cls('dt-e4m3'));
const tags = [...a.querySelectorAll('.lv g[data-op] > rect[class*="dt-"]')].map((r) => {
  const g = r.parentElement, t = [...g.parentElement.querySelectorAll('text')].find((x) => x.getAttribute('text-anchor') === 'end' && Math.abs(+x.getAttribute('y') - (+r.getAttribute("y") + 13)) <= 1 && +x.getAttribute('x') > +r.getAttribute('x') && +x.getAttribute('x') <= +r.getAttribute('x') + +r.getAttribute('width'));
  return t?.textContent === r.getAttribute('class').split('dt-')[1];
});
T.check('every tinted box carries a matching dtype readout', tags.length === 10 && tags.every(Boolean), tags.join(' '));
T.check('no dtype buttons (static tier)', !a.querySelector('button[data-dt]'), '');
T.done();
