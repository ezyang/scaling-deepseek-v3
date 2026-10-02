// Shared widget-screenshot machinery for ogshot.mjs (og:image cards) and
// pixelgold.mjs (pixel goldens): serve the repo, inject an isolation script
// that hides everything but the target widget, and screenshot a fixed window.
//
// Isolation: visibility sweep (an overlay cover loses to ancestor stacking
// contexts), unclip inner overflow boxes so wide diagrams paint fully, then
// fixed-position + scale the element so its PAINTED union (which can
// overflow the element box on both sides) lands where asked. Coordinates are
// delta-corrected by re-measuring, since a transformed ancestor makes
// position:fixed resolve against it rather than the viewport.
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { launch, serve, root } from './cdp.mjs';

export { root };
// one server + one browser per device scale factor, launched on first use.
// DSF is a launch flag, not Emulation.setDeviceMetricsOverride: emulated
// DSF paints the fractionally-positioned card ~1 device px off.
let server = null;
const browsers = new Map();

// opts.card = {W,H,PAD}: scale to fit the card, centered (og mode).
// opts.origin = N: scale 1, painted union anchored at (N, N) (pixel mode).
const ISOLATE = (sel, o) => `
  await new Promise((r) => setTimeout(r, 700));   // let widgets mount (virtual time)
  if (${!!o.dark}) {   // flip through the real theme path: class + C() re-renders
    const { setTheme } = await import(${JSON.stringify(o.themePath ?? '../src/theme.js')});
    setTheme(true);
    await new Promise((r) => setTimeout(r, 150));
  }
  const el = document.querySelector(${JSON.stringify(sel)});
  const subtree = [el, ...el.querySelectorAll('*')];
  const union = () => {
    let r = el.getBoundingClientRect(), L = r.left, T = r.top, R = r.right, B = r.bottom;
    for (const n of subtree) {
      const q = n.getBoundingClientRect();
      if (!q.width || !q.height || q.left < r.left - 600 || q.top < r.top - 600) continue;
      const s = getComputedStyle(n);
      if (s.visibility === 'hidden' || s.opacity === '0') continue;
      L = Math.min(L, q.left); T = Math.min(T, q.top); R = Math.max(R, q.right); B = Math.max(B, q.bottom);
    }
    return { left: L, top: T, w: R - L, h: B - T };
  };
  if (${!!o.unclip}) for (const n of subtree)   // og cards unclip; pixel goldens keep the as-rendered clipping
    if (n.scrollWidth > n.clientWidth + 1 || n.scrollHeight > n.clientHeight + 1) n.style.overflow = 'visible';
  const prior = subtree.map((n) => n.style.visibility);
  for (const n of document.querySelectorAll('body *')) n.style.visibility = 'hidden';
  subtree.forEach((n, i) => { n.style.visibility = prior[i]; });
  for (let a = el; a && a !== document.body; a = a.parentElement) a.style.visibility = 'visible';
  const u = union();
  const k = ${o.card ? `Math.min((${o.card.W} - 2 * ${o.card.PAD}) / u.w, (${o.card.H} - 2 * ${o.card.PAD}) / u.h, 1)` : '1'};
  el.style.width = el.getBoundingClientRect().width + 'px';
  el.style.position = 'fixed';
  el.style.left = '0px'; el.style.top = '0px';
  el.style.transform = 'scale(' + k + ')';
  el.style.transformOrigin = 'top left';
  for (let i = 0; i < 2; i++) {   // translation-delta correction, twice for safety
    const q = union();
    const [tx, ty] = ${o.card ? `[(${o.card.W} - q.w) / 2, (${o.card.H} - q.h) / 2]` : `[${o.origin}, ${o.origin}]`};
    el.style.left = parseFloat(el.style.left) + tx - q.left + 'px';
    el.style.top = parseFloat(el.style.top) + ty - q.top + 'px';
  }
`;

// Resolves with the PNG (also written to opts.out when given). Captured when
// the 40 s virtual-time budget runs out — the page is idle by then.
export async function shoot(page, sel, opts) {
  const srv = await (server ??= serve());
  if (!browsers.has(opts.dsf)) browsers.set(opts.dsf, launch([`--force-device-scale-factor=${opts.dsf}`]));
  const browser = await browsers.get(opts.dsf);
  const pageHtml = await readFile(join(root, page), 'utf8');
  const job = srv.inject(page, pageHtml.replace('</body>',
    `<script type="module">(async () => {${ISOLATE(sel, opts)}})();</script></body>`));
  const p = await browser.open(job.url, { width: opts.w, height: opts.h });
  try {
    await p.until();
    const png = await p.screenshot({ fast: !opts.out });   // files ship (og images); buffers just get decoded
    if (opts.out) await writeFile(opts.out, png);
    return png;
  } finally { await p.close(); job.done(); }
}

export async function closeShots() {
  for (const b of browsers.values()) (await b).close();
  (await server)?.close();
}
