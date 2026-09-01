// @page studies/02-hopper-memory.html
// @args --width 430
// 02 mobile: margin notes become numbered two-way footnotes (their desktop
// gutter margin-top shunts cleared), every wide widget previews, and focus
// mode works from a tap on the widget body itself.
await T.tick(300);
const notes = [...document.querySelectorAll('.mnotes li')];
T.check('all 6 margin notes became footnotes', notes.length === 6, notes.length);
T.check('refs are numbered links', document.querySelector('#mnref-1')?.getAttribute('href') === '#mnote-1'
  && T.text('#mnref-1') === '1', T.text('#mnref-1'));
T.check('backlinks point home', notes[0]?.querySelector('.mback')?.getAttribute('href') === '#mnref-1', '');
T.check('gutter shunts cleared on relocated notes',
  notes.every((li) => (li.querySelector('.mn')?.style.marginTop ?? 'x') === ''), '');
T.check('Notes section sits before the series nav', !!document.querySelector('.mnotes-sec + .series-nav'), '');

const prevs = [...document.querySelectorAll('.mprev')];
T.log('previews', prevs.map((p) => p.tagName.replace('DSV3-', '')).join(','));
T.check('every top-level widget previews (13 on 02)', prevs.length === 13, prevs.length);
T.check('no horizontal page scroll', document.documentElement.scrollWidth <= innerWidth + 1,
  `${document.documentElement.scrollWidth} vs ${innerWidth}`);

const sheet = document.querySelector('dsv3-sheet');
sheet.dispatchEvent(new MouseEvent('click', { bubbles: true }));
await T.tick();
T.check('tap on widget body focuses it', document.body.classList.contains('mfocus') && sheet.style.transform === '', '');
T.click('.mclose');
await T.tick();
T.check('closed clean', !document.body.classList.contains('mfocus') && sheet.classList.contains('mprev'), '');
T.done();
