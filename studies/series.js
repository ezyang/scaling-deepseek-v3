// The post series: one manifest, shared prev/next navigation. Each post page
// includes this module; it fills the series strip above the <h1> and injects
// prev/next cards at the end of <main>. Listed posts ship the strip as static
// markup (a strip injected at module load shoves the whole page down a line,
// a visible jump on a scrolled refresh); this rewrites it from the manifest.
// Ordering = manifest order; file slugs match the visible ordinal (post N of M).
import './toc.js';      // floating section rail — inert on short pages (gate inside)
import './mobile.js';   // ≤860px framing: widget previews + focus mode, margin notes → footnotes (gate inside)
export const SERIES = [
  { href: '01-deepseek-diagram.html', title: 'An infra-oriented diagram of the DeepSeek-V3 architecture' },
  { href: '02-hopper-memory.html', title: 'Memory: a Hopper case study' },
  // published incrementally — uncomment as posts go live (keep in step with index.html's list)
];

const i = SERIES.findIndex(p => location.pathname.endsWith('/' + p.href));
const main = document.querySelector('main');
if (i >= 0 && main) {
  const strip = main.querySelector('.series-strip') ?? document.createElement('nav');
  strip.className = 'series-strip';
  strip.innerHTML = `<a href="../index.html">DeepSeek-V3: from roofline to reality</a>`
    + (SERIES.length > 1 ? ` · post ${i + 1} of ${SERIES.length}` : '');
  if (!strip.isConnected) (main.querySelector('h1') ?? main.firstElementChild).insertAdjacentElement('beforebegin', strip);

  const card = (p, dir) => p
    ? `<a class="card ${dir}" href="./${p.href}"><small>${dir === 'prev' ? '← previous' : 'next →'}</small><b>${p.title}</b></a>`
    : '<span class="card empty"></span>';
  const nav = document.createElement('nav');
  nav.className = 'series-nav';
  nav.innerHTML = card(SERIES[i - 1], 'prev')
    + '<a class="card up" href="../index.html"><small>series</small><b>all posts</b></a>'
    + card(SERIES[i + 1], 'next');
  main.append(nav);
}
