// @page studies/02-hopper-memory.html
// floating section rail (toc.js): real sections only, scrollspy, hover
// overlay, and NO hash writes (the hash is widget-state territory)
await T.tick(200);
const nav = document.querySelector('nav.toc');
T.check('rail present on the long post', !!nav, '');
const labs = [...nav.querySelectorAll('.lab')].map(l => l.textContent);
T.check('one item per REAL section (scrap stays out)', labs.length >= 4 && !labs.includes('SCRAP HEAP'), labs.join('|'));
T.check('first item is the first section', labs[0] === document.querySelector('main h2').textContent, labs[0]);
// scrollspy: jump to the third section, its tick lights
const h3 = [...document.querySelectorAll('main h2')][2];
h3.scrollIntoView(); await T.tick(150);
window.dispatchEvent(new Event('scroll')); document.dispatchEvent(new Event('scroll')); await T.tick(50);
T.check('scrollspy lights the section under the reading line', nav.children[2].classList.contains('on'), '');
// clicks scroll but never write the hash
const hash0 = location.hash;
nav.children[0].dispatchEvent(new MouseEvent('click', { bubbles: true })); await T.tick(300);
T.check('a section click never touches the hash', location.hash === hash0, location.hash);
T.check('rail is viewport-fixed at the left edge', getComputedStyle(nav).position === 'fixed'
  && nav.getBoundingClientRect().left === 0, '');
T.check('collapsed: labels hidden (max-width 0)', getComputedStyle(nav.querySelector('.lab')).maxWidth === '0px', '');
T.done();
