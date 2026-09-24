// @page studies/02-hopper-memory.html
const ac = document.querySelector('dsv3-anatomy[controls="marks"]');
const tly = ac.querySelector('.lv > svg:last-of-type') ?? [...ac.querySelectorAll('svg')].pop();
const texts = [...tly.querySelectorAll('text')];
const tail = texts.find(t => t.textContent.includes('ms per mb·layer'));
T.check('time ruler tail present', !!tail, '');
const bb = tail.getBBox();
const vbW = tly.viewBox.baseVal.width || +tly.getAttribute('width');
T.log('tail end vs svg width', `${(bb.x + bb.width).toFixed(0)} / ${vbW}`);
T.check('tail fits inside the svg', bb.x + bb.width <= vbW, `${(bb.x + bb.width).toFixed(0)} vs ${vbW}`);
// tick labels are small integers (ms), not MFLOP hundreds
const near = texts.filter(t => Math.abs(t.getBBox().y - bb.y) < 8 && t !== tail).map(t => t.textContent);
T.log('tick labels', near.join(','));
T.check('tick labels count in ms (1,2,3…)', near.length > 4 && near.every(t => +t <= 20 && +t > 0), near.join(','));
// the router is a BF16 tensor-core GEMM (DeepSeek's trace: a bf16 cuBLAS
// kernel, fp32 logits out) — 3.7 MFLOP/token, well under one picket, so its
// box carries only the sub-picket trace, no solid pickets
const g = ac.querySelector('g[data-op="router"]');
const box = g.querySelector('rect.box').getBBox();
const inBox = (r) => { const b = r.getBBox(); return b.x > box.x && b.x < box.x + box.width && b.y > box.y && b.y < box.y + box.height; };
const solid = [...ac.querySelectorAll('.lv-scroll rect[height="5"]')].filter(r => inBox(r) && r.getAttribute('fill') !== 'none');
const trace = [...ac.querySelectorAll('.lv-scroll rect[width="1.4"]')].filter(inBox);
T.log('router pickets', `${solid.length} solid, ${trace.length} trace`);
T.check('router is sub-picket (bf16 GEMM): trace only, no solid pickets', solid.length === 0 && trace.length === 1, `${solid.length}/${trace.length}`);
// the group label leads with TIME
T.check('group label says time at H100 peak', texts.some(t => t.textContent.includes('per-layer compute as TIME at H100 peak')), '');
T.done();
