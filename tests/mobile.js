// @page studies/01-deepseek-diagram.html
// @args --width 430
// Mobile framing (studies/mobile.js): widgets become inert scaled previews;
// tapping (widget or its explore button) enters focus mode where the PAGE
// pans the natural-width widget. 01 has no margin notes, so no Notes section
// appears (its hand-written .fnotes stays).
await T.tick(300);
const l = document.querySelector('dsv3-anatomy[layer="diagram"]');
T.check('preview: widget scaled + classed', l.classList.contains('mprev') && /^scale\(0\./.test(l.style.transform), l.style.transform);
T.check('preview: children inert (taps land on the widget)',
  getComputedStyle(l.querySelector('svg')).pointerEvents === 'none', '');
T.check('preview: no horizontal page scroll',
  document.documentElement.scrollWidth <= innerWidth + 1, `${document.documentElement.scrollWidth} vs ${innerWidth}`);
const btns = [...document.querySelectorAll('.mopen')];
T.check('explore buttons under both anatomies', btns.length === 2, btns.length);
T.check('no Notes section; manual .fnotes intact',
  !document.querySelector('.mnotes-sec') && !!document.querySelector('.fnotes'), '');

T.click('.mopen');
await T.tick();
T.check('focus: body flagged, widget at natural size',
  document.body.classList.contains('mfocus') && l.style.transform === '', '');
T.check('focus: prose hidden', getComputedStyle(document.querySelector('main > p')).display === 'none', '');
T.check('focus: page pans horizontally (natural width overflows)',
  document.documentElement.scrollWidth > innerWidth, document.documentElement.scrollWidth);
T.check('focus: children interactive again',
  getComputedStyle(l.querySelector('svg')).pointerEvents !== 'none', '');

T.click('.mclose');
await T.tick();
T.check('close: preview restored, scroll sane',
  !document.body.classList.contains('mfocus') && l.classList.contains('mprev')
  && /^scale\(0\./.test(l.style.transform) && document.documentElement.scrollWidth <= innerWidth + 1, '');
T.done();
