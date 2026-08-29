// @page studies/scratch-globals.html
const l = document.getElementById('all-diagram');
const plan = l.closest('.anat-grid').querySelector('dsv3-anatomy-plan');
const embed = () => plan.querySelector('g[data-op="embed"]');
const stripsOf = () => {
  const rects = [...embed().querySelectorAll('rect')].slice(1);   // first is the pill
  return rects.map(r => `${r.getAttribute('fill') === 'none' ? 'hollow' : 'fill'}:${r.getAttribute('fill') === 'none' ? r.getAttribute('stroke') : r.getAttribute('fill')}`);
};
T.log('per-block embed strips', JSON.stringify(stripsOf()));
const cumBtn = [...l.parentElement.querySelectorAll('button')].find(b => b.textContent === 'per block' || b.textContent.includes('blocks'));
cumBtn.click(); await T.tick(800);
T.log('x58 embed strips', JSON.stringify(stripsOf()));
T.check('embedding unchanged by x58', JSON.stringify(stripsOf()).includes('fill'), JSON.stringify(stripsOf()));
T.done();
