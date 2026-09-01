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

// footnote hops scroll WITHOUT putting their anchors in the hash (that's
// widget-state territory — a real anchor jump clobbers widget params and
// corrupts history entries; widgets may legitimately write state meanwhile)
T.click('#mnref-1');
await T.tick(400);
T.check('ref hop scrolls to the note, no anchor in the hash',
  !location.hash.includes('mnote') && scrollY > 0
  && document.querySelector('#mnote-1').getBoundingClientRect().top >= 0
  && document.querySelector('#mnote-1').getBoundingClientRect().top < innerHeight,
  `y=${scrollY} hash='${location.hash}'`);
T.click('#mnote-1 .mback');
await T.tick(400);
const refTop = document.querySelector('#mnref-1').getBoundingClientRect().top;
T.check('backlink returns to the ref, no anchor in the hash',
  !location.hash.includes('mnref') && scrollY < 200 && refTop >= 0 && refTop < innerHeight,
  `top=${Math.round(refTop)} y=${scrollY}`);
scrollTo(0, 0); await T.tick(50);

const prevs = [...document.querySelectorAll('.mprev')];
T.log('previews', prevs.map((p) => p.tagName.replace('DSV3-', '')).join(','));
T.check('every top-level widget previews (13 on 02)', prevs.length === 13, prevs.length);
T.check('no horizontal page scroll', document.documentElement.scrollWidth <= innerWidth + 1,
  `${document.documentElement.scrollWidth} vs ${innerWidth}`);

const sheet = document.querySelector('dsv3-sheet');
sheet.dispatchEvent(new MouseEvent('click', { bubbles: true }));
await T.tick();
T.check('tap on widget body focuses it', document.body.classList.contains('mfocus') && sheet.style.transform === '', '');
const meta = document.querySelector('meta[name="viewport"]');
T.check('focus locks the viewport scale (no native re-fit / pinch)',
  meta.getAttribute('content').includes('maximum-scale=1')
  && getComputedStyle(document.body).touchAction === 'pan-x pan-y', meta.getAttribute('content'));
// double-tap on a non-control spot: fit-width; again: back to 1x centered there
sheet.dispatchEvent(new MouseEvent('dblclick', { bubbles: true, clientX: 5, clientY: 5 }));
await T.tick();
T.check('double-tap zooms to fit-width', /^scale\(0\./.test(sheet.style.transform), sheet.style.transform);
sheet.dispatchEvent(new MouseEvent('dblclick', { bubbles: true, clientX: 5, clientY: 5 }));
await T.tick();
T.check('double-tap again returns to 1x', sheet.style.transform === '', sheet.style.transform);
T.click('.mclose');
await T.tick();
T.check('closed clean, viewport lock lifted',
  !document.body.classList.contains('mfocus') && sheet.classList.contains('mprev')
  && !meta.getAttribute('content').includes('maximum-scale')
  && getComputedStyle(document.body).touchAction !== 'pan-x pan-y', meta.getAttribute('content'));

// focused schedule strip: its overscroll-behavior-x:none guard must not
// trap touch pans — the chain to the page scroller is freed while focused
const pps = document.querySelector('dsv3-pp-schedule[layer="local-diagram"]');
pps.dispatchEvent(new MouseEvent('click', { bubbles: true }));
await T.tick();
const scr = pps.querySelector('.scroll');
T.check('focus frees scroll chaining on inner scrollers',
  document.body.classList.contains('mfocus') && getComputedStyle(scr).overscrollBehaviorX === 'auto',
  getComputedStyle(scr).overscrollBehaviorX);
T.check('focus freezes inner scrollers (no gesture latching)',
  getComputedStyle(scr).overflowX === 'hidden' && scr.scrollLeft === 0, getComputedStyle(scr).overflowX);
T.click('.mclose');
await T.tick();
T.check('strip guard + scroller restored after close',
  getComputedStyle(scr).overscrollBehaviorX === 'none' && getComputedStyle(scr).overflowX === 'auto', '');
T.done();
