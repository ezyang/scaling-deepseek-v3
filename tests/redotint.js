// @page studies/03-roofline.html
// <dsv3-layer redotint controls="static">: the save-everything drawing tints
// the instance's recompute policy, GEMMs only — under dsv3 exactly the two
// MLA up-projections (the norms, RoPE and SwiGLU replay too, but aren't GEMMs).
// Any box color code drops the a2a boxes' comm purple.
await T.tick(100);
const r = document.querySelector('#redo-block');
const redo = [...r.querySelectorAll('.lv g[data-op] > rect.redo')].map((x) => x.parentElement.dataset.op).sort().join(' ');
T.check('redo tint: q up-proj + kv up-proj only', redo === 'kv_up q_up', redo);
T.check('static tier tints in the amber variant', r.querySelectorAll('.lv rect.redo.y').length === 2, '');
T.check('static tier carries the replay key', /GEMM replayed in backward/.test(r.querySelector('.lv-head')?.textContent ?? ''), '');
const d = document.querySelector('#dtype-block');
T.check('dtypetint: dispatch/combine drawn as plain boxes', !d.querySelector('rect.comm')
  && ['dispatch', 'combine'].every((id) => d.querySelector(`g[data-op="${id}"] > rect.box`)), '');
T.done();
