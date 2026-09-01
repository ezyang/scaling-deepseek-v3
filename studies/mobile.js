// Mobile treatment (≤860px), applied by JS so wider layouts are untouched.
//  · Every top-level widget becomes an inert, scaled-to-fit PREVIEW in the
//    flow (children lose pointer events; tapping it or its "explore" button
//    enters focus mode). Focus mode hides everything else and lets the PAGE
//    scroll the widget at its natural desktop width — native two-axis pan
//    and pinch zoom, full-size knob targets. The desktop implementation IS
//    the mobile implementation, only re-framed: widgets are never moved in
//    the DOM (connectedCallback does not tolerate a re-mount) and never
//    re-laid-out below their natural width.
//  · Margin notes become real end-of-post footnotes (two-way links); their
//    inline margin-top shunts, which position them in the desktop gutter,
//    are cleared (study.css also neutralizes them for the mid-width
//    note-boxes).
// One-way: runs once when the viewport first matches; widening afterwards
// keeps the mobile framing (a reload restores the desktop layout).
const MQ = matchMedia('(max-width: 860px)');
const W = 'dsv3-anatomy, dsv3-layer, dsv3-pp-schedule, dsv3-pp-fold, dsv3-beat-deck, dsv3-sheet';

function footnotes(main) {
  const refs = [...main.querySelectorAll('.mn-ref')].filter((r) => r.nextElementSibling?.classList.contains('mn'));
  if (!refs.length) return;
  const ol = document.createElement('ol');
  ol.className = 'mnotes';
  refs.forEach((ref, i) => {
    const n = i + 1, note = ref.nextElementSibling;
    ref.classList.add('live');
    ref.innerHTML = `<sup><a id="mnref-${n}" href="#mnote-${n}">${n}</a></sup>`;
    note.style.marginTop = '';   // the desktop gutter shunts (-9em …) mean nothing down here
    const li = document.createElement('li');
    li.id = `mnote-${n}`;
    const back = document.createElement('a');
    back.href = `#mnref-${n}`; back.textContent = '↩'; back.className = 'mback';
    li.append(note, ' ', back);
    ol.append(li);
  });
  const sec = document.createElement('section');
  sec.className = 'fnotes mnotes-sec';
  sec.innerHTML = '<h2>Notes</h2>';
  sec.append(ol);
  (main.querySelector('.series-nav') ?? { before: (x) => main.append(x) }).before(sec);
}

function focus(el, saved, pt) {
  if (document.body.classList.contains('mfocus')) return;
  const at = { x: scrollX, y: scrollY };
  let top = el;
  while (top.parentElement && top.parentElement.tagName !== 'MAIN') top = top.parentElement;
  document.body.classList.add('mfocus');
  // native pinch zoom is a trap here: no API can reset it on close, and a
  // pinched-in view loses the ✕ (fixed elements pin to the LAYOUT viewport).
  // So focus mode locks it — the meta rescale-lock for Android/emulation
  // (which otherwise re-fits the page when the document turns wide),
  // touch-action for iOS (which ignores user-scalable=no) — and offers
  // double-tap 1× ↔ fit-width instead.
  document.documentElement.classList.add('mta');
  const vp = document.querySelector('meta[name="viewport"]');
  const vp0 = vp?.getAttribute('content');
  vp?.setAttribute('content', 'width=device-width, initial-scale=1, minimum-scale=1, maximum-scale=1, user-scalable=no');
  top.classList.add('mfull-top');
  el.classList.remove('mprev');
  el.style.transform = '';
  el.style.marginBottom = '';
  el.style.marginLeft = saved.over + 'px';
  const x = document.createElement('button');
  x.type = 'button'; x.className = 'mclose'; x.textContent = '✕';
  document.body.append(x);
  // belt-and-braces: if a browser zooms anyway (or was already pinched when
  // focus opened), glue the ✕ to the VISUAL viewport at constant size
  const vv = window.visualViewport;
  const glue = vv ? () => {
    x.style.transform = `translate(${vv.offsetLeft}px, ${vv.offsetTop}px) scale(${1 / vv.scale})`;
  } : null;
  if (glue) { vv.addEventListener('resize', glue); vv.addEventListener('scroll', glue); glue(); }
  // double-tap (anywhere that isn't a control): toggle 1× ↔ fit-width,
  // keeping the tapped spot centered
  let z = 1;
  const fitK = Math.min(1, (innerWidth - 24) / (el.offsetWidth + saved.over));
  const dbl = (e) => {
    if (e.target.closest('button, a, input, select, textarea, label')) return;
    const r0 = el.getBoundingClientRect();
    const p = { x: (e.clientX - r0.left) / z, y: (e.clientY - r0.top) / z };   // content coords
    z = z === 1 ? fitK : 1;
    el.style.transform = z === 1 ? '' : `scale(${z})`;
    const r1 = el.getBoundingClientRect();
    scrollTo(Math.max(0, r1.left + scrollX + p.x * z - innerWidth / 2),
      Math.max(0, r1.top + scrollY + p.y * z - innerHeight / 3));
    e.preventDefault();
  };
  el.addEventListener('dblclick', dbl);
  // start zoomed in ON the tapped spot (photo-viewer style): map the tap
  // through the preview scale and center it; the explore button (no spot)
  // opens at the widget's top
  const r = el.getBoundingClientRect();
  if (pt) scrollTo(Math.max(0, r.left + scrollX + pt.x - innerWidth / 2),
    Math.max(0, r.top + scrollY + pt.y - innerHeight / 3));
  else scrollTo(0, 0);
  const esc = (e) => { if (e.key === 'Escape') close(); };
  const close = () => {
    x.remove();
    removeEventListener('keydown', esc);
    el.removeEventListener('dblclick', dbl);
    if (glue) { vv.removeEventListener('resize', glue); vv.removeEventListener('scroll', glue); }
    document.body.classList.remove('mfocus');
    document.documentElement.classList.remove('mta');
    if (vp && vp0 != null) vp.setAttribute('content', vp0);
    top.classList.remove('mfull-top');
    el.classList.add('mprev');
    el.style.transform = saved.transform;
    el.style.marginBottom = saved.mb;
    el.style.marginLeft = saved.ml;
    scrollTo(at.x, at.y);
  };
  x.onclick = close;
  addEventListener('keydown', esc);
}

// content extent via the union of leaf-ish descendant RECTS: rects ignore
// ancestor clipping (inner scroll boxes, the tokens rail overhang), work
// identically across engines (Safari's scrollWidth omits visible overflow —
// on-device the responsive bar charts measured as "fits" and never got
// previews), and skipping deep wrappers ignores stretch-to-fill containers.
function extent(el) {
  const r0 = el.getBoundingClientRect();
  let L = r0.left, R = r0.left + 1;
  for (const d of el.querySelectorAll('*')) {
    if (d.firstElementChild?.firstElementChild) continue;   // wrappers stretch; leaves hug the content
    const q = d.getBoundingClientRect();
    if (!q.width || q.left < r0.left - 600) continue;
    L = Math.min(L, q.left); R = Math.max(R, q.right);
  }
  return { over: Math.max(0, Math.ceil(r0.left - L)), w: Math.ceil(R - r0.left) };
}

function previews(main) {
  const avail = main.clientWidth - 40;   // main's 20px side paddings
  const PROBE = 940;   // desktop main content width: measure the DESKTOP layout, not the squished one
  for (const el of main.querySelectorAll(W)) {
    if (el.parentElement.closest(W)) continue;   // nested widgets ride inside their host's preview
    // Probe: render wide, measure, adopt the measured width, re-measure
    // (knob rows unwrap and content re-seats). Renderers bake pixel widths
    // at render time, so each width change needs an explicit re-render.
    el.classList.add('mwide');   // free viewport-capped internals (see study.css)
    const w0 = el.style.width;
    let cw = PROBE, u;
    for (let pass = 0; pass < 2; pass++) {
      el.style.width = cw + 'px';
      for (const w of [el, ...el.querySelectorAll(W)]) w.render?.();
      u = extent(el);
      if (pass === 0 && u.w <= avail + 8) break;
      cw = u.w;
    }
    if (u.w <= avail + 8) {                      // genuinely fits: restore the live in-flow layout
      el.classList.remove('mwide');
      el.style.width = w0;
      for (const w of [el, ...el.querySelectorAll(W)]) w.render?.();
      continue;
    }
    const over = u.over;
    cw = u.w;
    el.style.width = cw + 'px';
    // the visual span can exceed the set width: card padding/borders (the
    // beat-deck) widen the border box beyond the content width
    const k = avail / (Math.max(el.offsetWidth, cw) + over);
    el.classList.add('mprev');
    el.style.minHeight = '0';                    // desktop placeholder heights don't apply at this width
    el.style.transformOrigin = '0 0';
    el.style.transform = `scale(${k})`;
    el.style.marginLeft = Math.ceil(over * k) + 'px';
    const h = el.offsetHeight;                   // layout height (transform-independent)
    el.style.marginBottom = `${Math.round(26 - h * (1 - k))}px`;
    const saved = { transform: el.style.transform, mb: el.style.marginBottom, ml: el.style.marginLeft, over };
    const btn = document.createElement('button');
    btn.type = 'button'; btn.className = 'mopen';
    btn.textContent = '⤢ tap to explore';
    el.after(btn);
    btn.onclick = () => focus(el, saved, null);
    el.addEventListener('click', (e) => {        // children are pointer-inert, so taps land here
      const r = el.getBoundingClientRect();      // preview box (scaled): map the tap to natural coords
      focus(el, saved, { x: (e.clientX - r.left) / k, y: (e.clientY - r.top) / k });
    });
  }
}

function setup() {
  const main = document.querySelector('main');
  if (!MQ.matches || !main || main.dataset.mobile) return;
  main.dataset.mobile = '1';
  footnotes(main);
  previews(main);
}

// widgets that FOLLOW another widget (the sheet, the schedule strip) render
// one event-turn after load — measure only once everything has settled
const go = () => setTimeout(setup, 250);
if (document.readyState === 'complete') go();
else addEventListener('load', go, { once: true });
MQ.addEventListener('change', (e) => { if (e.matches) go(); });
