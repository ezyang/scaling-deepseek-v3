// @page studies/02-hopper-memory.html
// Night mode: the ◐ toggle flips :root.dark (CSS variables) AND re-renders
// widgets so their C()-emitted SVG colors follow (fills stay concrete hexes).
await T.tick(300);
T.check('◐ toggle present', !!document.querySelector('.themeb'), '');
T.check('defaults to light (headless prefers light)', !document.documentElement.classList.contains('dark'), '');
const bar = () => document.querySelector('dsv3-layer[comps="weights"] svg rect[data-bar]');
T.check('light: weights bar wears the light blue', bar()?.getAttribute('fill') === '#2a78d6', bar()?.getAttribute('fill'));

T.click('.themeb');
await T.tick(200);
T.check('dark: root class + page surface flips',
  document.documentElement.classList.contains('dark')
  && getComputedStyle(document.body).backgroundColor === 'rgb(22, 22, 20)', getComputedStyle(document.body).backgroundColor);
T.check('dark: widgets re-rendered — weights bar wears the DARK blue',
  bar()?.getAttribute('fill') === '#5c9ae6', bar()?.getAttribute('fill'));
T.check('dark: choice persisted', localStorage.getItem('dsv3-theme') === 'dark', '');

T.click('.themeb');
await T.tick(200);
T.check('back to light, bars repainted', !document.documentElement.classList.contains('dark')
  && bar()?.getAttribute('fill') === '#2a78d6', bar()?.getAttribute('fill'));
T.done();
