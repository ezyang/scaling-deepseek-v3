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
// fp32 router pickets: brick-colored height-5 rects present in its box
const g = ac.querySelector('g[data-op="router"]');
const box = g.querySelector('rect.box').getBBox();
const brick = [...ac.querySelectorAll('.lv-scroll rect[height="5"]')].filter(r => {
  const b = r.getBBox();
  return b.x > box.x && b.x < box.x + box.width && b.y > box.y && b.y < box.y + box.height
    && r.getAttribute('fill') !== 'none';
});
T.log('router pickets', `${brick.length} fill=${brick[0]?.getAttribute('fill')}`);
T.check('router shows ~5 fp32 pickets at the CUDA-core rate', brick.length >= 5 && brick.length <= 7, brick.length);
// the group label leads with TIME
T.check('group label says time at H100 peak', texts.some(t => t.textContent.includes('per-layer compute as TIME at H100 peak')), '');
T.done();
