// The post series: one manifest, shared prev/next navigation. Each post page
// includes this module; it injects a series strip under the <h1> and
// prev/next cards at the end of <main>. Ordering = manifest order; file slugs
// match the visible ordinal (post N of M).
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
  const strip = document.createElement('nav');
  strip.className = 'series-strip';
  strip.innerHTML = `<a href="../index.html">DeepSeek-V3: from roofline to reality</a>`
    + (SERIES.length > 1 ? ` · post ${i + 1} of ${SERIES.length}` : '');
  const h1 = main.querySelector('h1');
  (h1 ?? main.firstElementChild).insertAdjacentElement('beforebegin', strip);

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
