// @page studies/01-deepseek-diagram.html
// @args --width 430
// Narrow viewport: diagrams NEVER scale down — natural size + horizontal scroll.
const l = document.getElementById('diagram');
const svg = l.querySelector('.lv-scroll svg') ?? l.querySelector('.lv svg');
const natural = +svg.getAttribute('width');
T.log('viewport', window.innerWidth);
T.check('svg renders at natural width (no scaling)',
  Math.abs(svg.getBoundingClientRect().width - natural) < 1,
  `${Math.round(svg.getBoundingClientRect().width)} vs attr ${natural}`);
const scroller = svg.closest('.lv-scroll');
T.check('scroll container present and overflowing',
  scroller && getComputedStyle(scroller).overflowX === 'auto'
  && scroller.scrollWidth > scroller.clientWidth, scroller && `${scroller.scrollWidth} > ${scroller.clientWidth}`);
const grid = l.closest('.anat-grid');
T.check('anatomy grid stacked (single column)',
  grid && getComputedStyle(grid).gridTemplateColumns.split(' ').length === 1,
  grid && getComputedStyle(grid).gridTemplateColumns);
T.done();
