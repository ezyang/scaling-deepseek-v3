// @page studies/ac-pareto.html
// the AC Pareto study: every save/recompute marking of 02's MoE layer (both
// a2a outputs saved), the frontier, its nested convex hull as numbered steps
// + the film of mini schematics, and the two-way link to 02's AC diagram
// (whose ↻ boxes wear the redotint teal)
const w = document.querySelector('dsv3-acpareto');
for (let i = 0; i < 100 && !('ready' in w.dataset); i++) await T.tick(100);
T.check('enumeration finished', 'ready' in w.dataset, '');
const IDS = ['norm1', 'qkv_down', 'q_norm', 'kv_norm', 'q_up', 'kv_up', 'rope_q', 'rope_kv',
  'attn', 'o_proj', 'x1', 'norm2', 'router', 'dispatch', 'gate_up', 'swiglu', 'ffn_down', 'combine', 'moe_add'];
const redo = (p) => IDS.filter((id) => p.marks[id] !== true);
T.check('hull is nested: every step recomputes a superset of the previous',
  w.hull.every((p, i) => !i || redo(w.hull[i - 1]).every((id) => redo(p).includes(id))), w.hull.map((p) => redo(p).join('+')).join(' | '));
const ds = w.presets.find((q) => q.key === 'dsv3');
const hi = w.hull.findIndex((p) => Math.abs(p.f - ds.f) < 1e-12 && Math.abs(p.b - ds.b) < 1e-6);
T.check('DeepSeek-V3 policy is a hull step, with the exact preset marks', hi > 0 && redo(w.hull[hi]).join() === redo(ds).join(), hi);
T.check('hull starts at save-everything (no ↻ at all)', redo(w.hull[0]).length === 0, redo(w.hull[0]));
T.check('attn-replay is dominated', !w.front.some((p) => same(p, w.presets.find((q) => q.key === 'attn-replay'))), '');
function same(p, q) { return Math.abs(p.f - q.f) < 1e-12 && Math.abs(p.b - q.b) < 1e-6; }
T.check('no a2a replay on the chart: presets that replay comm (moe, full) are left out',
  !w.presets.some((q) => q.key === 'moe' || q.key === 'full') && w.hull.every((p) => p.marks.dispatch && p.marks.combine), w.presets.map((q) => q.key).join());
const cur = () => w.querySelector('circle[data-cur]');
const hx = (i) => +w.querySelector(`circle[data-hullpt="${i}"]`).getAttribute('cx');
T.check('ring starts on the DeepSeek step (diagram preset dsv3)', cur() && Math.abs(+cur().getAttribute('cx') - hx(hi)) < 0.2, cur()?.getAttribute('cx'));
T.check('readout names the step', w.querySelector('.ro').textContent.includes(`hull step ${hi} of ${w.hull.length - 1}`), w.querySelector('.ro').textContent);
// the film: one frame per step; its solid cells are exactly what the step adds
const frames = [...w.querySelectorAll('.fr')];
T.check('film: one frame per hull step, the diagram\'s step outlined', frames.length === w.hull.length
  && frames.findIndex((f) => f.classList.contains('cur')) === hi, frames.length);
const SOLID = '#0a98a0';
const solid = (el) => [...el.querySelectorAll('svg rect')].filter((r) => r.getAttribute('fill') === SOLID).length;
T.check('frame 1 lights one cell (SwiGLU ×2: routed + shared), step 0 none', solid(frames[0]) === 0 && solid(frames[1]) === 2, `${solid(frames[0])} ${solid(frames[1])}`);
T.log('film captions', frames.map((f) => f.textContent.replace(/\s+/g, ' ').trim()).join(' | '));
// hover a frame: its vertex grows, the tip carries the labeled schematic
const fr = frames[hi].getBoundingClientRect();
frames[hi].dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: fr.left + 20, clientY: fr.top + 20 }));
await T.tick(50);
const tip = w.querySelector('.dsv3-tip');
T.check('frame hover: vertex lit, tip = step head + schematic', frames[hi].classList.contains('hot')
  && w.querySelector(`circle[data-hullpt="${hi}"]`).getAttribute('r') === '5.5'
  && /hull step \d+ — adds q up-proj, RoPE/.test(tip.textContent) && tip.querySelectorAll('svg text').length === 23, tip.textContent.slice(0, 80));
// the diagram tints exactly its ↻ boxes
const tinted = () => [...document.querySelectorAll('dsv3-anatomy rect.redo')];
T.check('diagram: ↻ boxes wear the teal tint', tinted().length > 0 && tinted().every((r) => {
  const op = r.closest('g[data-op]')?.dataset.op;
  return !op || !IDS.includes(op) || document.getElementById('ac-layer').marks[op] !== true;
}), tinted().length);
// click frame 0 → the diagram saves everything, nothing tinted
frames[0].dispatchEvent(new MouseEvent('click', { bubbles: true }));
await T.tick(600);
T.check('frame click loads save-everything: no tint, ring on step 0', tinted().length === 0 && Math.abs(+cur().getAttribute('cx') - hx(0)) < 0.2, tinted().length);
// click the last hull vertex → the diagram takes its marks
const last = w.hull.length - 1, lc = w.querySelector(`circle[data-hullpt="${last}"]`);
const svg = w.querySelector('svg.plot'), r = svg.getBoundingClientRect();
svg.dispatchEvent(new MouseEvent('click', { bubbles: true, clientX: r.left + +lc.getAttribute('cx'), clientY: r.top + +lc.getAttribute('cy') }));
await T.tick(600);
const layer = document.getElementById('ac-layer');
T.check('chart click loads the vertex into the diagram', redo({ marks: layer.marks }).join() === redo(w.hull[last]).join(), Object.keys(layer.marks).join());
T.check('ring and frame outline follow', Math.abs(+cur().getAttribute('cx') - hx(last)) < 0.2
  && w.querySelectorAll('.fr')[last].classList.contains('cur'), '');
// replaying the dispatch leaves the charted space: no ring, the readout says so
w._load({ ...layer.marks, dispatch: undefined });
await T.tick(600);
T.check('a2a replay in the diagram: ring hidden, readout explains', !cur() && /all-to-all/.test(w.querySelector('.ro').textContent), w.querySelector('.ro').textContent);
T.log('widget height', w.getBoundingClientRect().height);
T.done();
