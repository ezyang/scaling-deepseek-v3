// The post series: one manifest, shared prev/next navigation. Each post page
// includes this module; it injects a series strip under the <h1> and
// prev/next cards at the end of <main>. Ordering = manifest order; filenames
// keep their NN- slugs but the visible numbering is ordinal.
export const SERIES = [
  { href: '00-deepseek-diagram.html', title: 'The DSv3 block diagram: what it shows and why' },
  { href: '01-hopper-memory.html', title: 'A DSv3-family MoE on 2048 H100s: memory' },
  { href: '02-nvidia-mlperf-gb300.html', title: 'NVIDIA’s MLPerf 6.0 submitted configuration (256 × GB300)' },
];

const i = SERIES.findIndex(p => location.pathname.endsWith('/' + p.href));
const main = document.querySelector('main');
if (i >= 0 && main) {
  const strip = document.createElement('nav');
  strip.className = 'series-strip';
  strip.innerHTML = `<a href="../index.html">DeepSeek-V3 in detail</a> · post ${i + 1} of ${SERIES.length}`;
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
