// @page studies/02-hopper-memory.html
// solo -> all: no four-color flash on the total row (grey + dissolving tip only)
const layer = () => document.getElementById('local-diagram');
const md = (el2) => el2.dispatchEvent(new MouseEvent('mousedown', { bubbles: true }));
const rowLabels = () => [...layer().querySelectorAll('.lv-bar g[data-prop]')];
const totalRowColors = () => {
  // rects in the total row band (y of the 5th row)
  const rects = [...layer().querySelectorAll('.lv-bar rect[height="8"]')];
  const ys = [...new Set(rects.map(r => +r.getAttribute('y')))].sort((a, b) => a - b);
  const totY = ys[ys.length - 1];
  return rects.filter(r => +r.getAttribute('y') === totY).map(r => r.getAttribute('fill'));
};
md(rowLabels()[0]); await T.tick(700);   // solo weights
T.check('solo: grey + blue only', JSON.stringify([...new Set(totalRowColors())].sort()) === JSON.stringify(['#2a78d6', '#c3c2b7']), totalRowColors().join('|'));
md(rowLabels()[0]); await T.tick(60);    // returning to all — probe MID-tween
const mid = [...new Set(totalRowColors())];
T.log('mid-tween total colors', mid.join('|'));
T.check('no orange/green/amber mid-tween', !mid.includes('#eb6834') && !mid.includes('#1baf7a') && !mid.includes('#eda100'), mid.join('|'));
await T.tick(700);
T.check('ends plain grey', JSON.stringify([...new Set(totalRowColors())]) === JSON.stringify(['#52514e']), totalRowColors().join('|'));
T.done();
