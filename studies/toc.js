// Floating section rail for long posts. The left gutter is NOT free space —
// the anatomy plan column extrudes into it — so the rail never reserves
// width there: it collapses to ticks pinned at the viewport's left edge and
// expands as an OVERLAY (white panel, above the diagrams) on hover.
// Inert on short pages (the gate below): post 01 imports this via series.js
// but has a single section, so its DOM is untouched.
// Widget state lives in location.hash (URLSearchParams), so section links
// must never write the hash — clicks scroll, nothing else.
const main = document.querySelector('main');
const hs = [];
if (main) {
  for (const node of main.childNodes) {
    // authored scrap stays out of the nav: collection stops at the SLOP marker
    if (node.nodeType === Node.TEXT_NODE && /SLOP/.test(node.textContent)) break;
    if (node.tagName === 'H2') hs.push(node);
  }
}
if (hs.length >= 4) {
  const nav = document.createElement('nav');
  nav.className = 'toc';
  nav.setAttribute('aria-label', 'sections');
  for (const h of hs) {
    const a = document.createElement('a');
    a.innerHTML = '<span class="tick"></span><span class="lab"></span>';
    a.querySelector('.lab').textContent = h.textContent;
    a.onclick = (e) => { e.preventDefault(); h.scrollIntoView({ behavior: 'smooth', block: 'start' }); };
    nav.append(a);
  }
  document.body.append(nav);
  // scrollspy: the current section = the last heading above the reading line
  const spy = () => {
    let cur = -1;
    for (let j = 0; j < hs.length; j++) if (hs[j].getBoundingClientRect().top <= 130) cur = j;
    [...nav.children].forEach((a, j) => a.classList.toggle('on', j === cur));
  };
  document.addEventListener('scroll', spy, { passive: true });
  spy();
}
