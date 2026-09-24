// @page studies/01-deepseek-diagram.html
// the shared tooltip (src/tip.js) on the param tally: hover shows, click
// pins (selectable) along with the row, Escape closes (cross-widget rules:
// tests/tip02.js)
await T.tick(200);
const tal = document.querySelector('dsv3-param-tally');
const tip = tal.querySelector(':scope > .dsv3-tip');
const at = (el2, type) => {
  const r = el2.getBoundingClientRect();
  el2.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, clientX: r.left + 3, clientY: r.top + 3 }));
};
const num = tal.querySelector('tbody tr .pnum');
num.scrollIntoView({ block: 'center' }); await T.tick(30);
at(num, 'mousemove');
T.check('tally hover: exact count', tip.style.display === 'block' && tip.textContent === Number(num.dataset.v).toLocaleString('en-US'), tip.textContent);
const tr = num.closest('tr');
at(num, 'click'); await T.tick(30);
T.check('click pins the tip (selectable) and the row', tip.classList.contains('pinned') && getComputedStyle(tip).pointerEvents === 'auto' && tr.classList.contains('sel'), tip.className);
at(tal.querySelectorAll('tbody tr .pnum')[2], 'mousemove');
T.check('pinned: hover elsewhere leaves it', tip.textContent === Number(num.dataset.v).toLocaleString('en-US'), tip.textContent);
document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape' }));
T.check('Escape closes', tip.style.display === 'none' && !tip.classList.contains('pinned'), '');
T.done();
