// @page studies/02-hopper-memory.html
// the shared tooltip (src/tip.js) across widgets: the raw-bytes lens pins
// too; a click while ANY tip is pinned only closes (never re-pins); the
// formula sheet's refs jump on click, so they show a tip but never pin one
await T.tick(200);
const at = (el2, type) => {
  const r = el2.getBoundingClientRect();
  el2.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, clientX: r.left + 3, clientY: r.top + 3 }));
};
const A = document.getElementById('local-diagram'), B = document.getElementById('ac-layer');
const tipA = A.querySelector('.lv-tip'), tipB = B.querySelector('.lv-tip');
const rawA = A.querySelector('[data-raw]'), rawB = B.querySelector('.lv-scroll text[data-tip]');
rawA.scrollIntoView({ block: 'center' }); await T.tick(30);
at(rawA, 'mousemove'); at(rawA, 'click'); await T.tick(30);
T.check('raw-bytes lens pins', tipA.classList.contains('pinned') && / B$/.test(tipA.textContent), tipA.textContent);
rawB.scrollIntoView({ block: 'center' }); await T.tick(30);
at(rawB, 'click'); await T.tick(30);
T.check('a click on another widget closes the pinned tip…', tipA.style.display === 'none' && !tipA.classList.contains('pinned'), '');
T.check('…without pinning its own', !tipB.classList.contains('pinned'), tipB.className);
at(rawB, 'click'); await T.tick(30);
T.check('the next click pins there', tipB.classList.contains('pinned') && tipB.style.display === 'block', tipB.textContent);
document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
T.check('Escape closes', tipB.style.display === 'none', '');
const sheet = document.querySelector('dsv3-sheet'), stip = sheet.querySelector(':scope > .dsv3-tip');
const ref = sheet.querySelector('.cellref');
ref.scrollIntoView({ block: 'center' }); await T.tick(30);
at(ref, 'mousemove');
T.check('sheet ref hover shows the cell', stip.style.display === 'block' && / · /.test(stip.textContent), stip.textContent.slice(0, 50));
at(ref, 'click'); await T.tick(30);
T.check('sheet ref click jumps, never pins', !stip.classList.contains('pinned'), stip.className);
T.done();
