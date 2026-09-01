// og:image prerenders: screenshot each post's signature widget onto a
// 1200×630 card (rendered at 2× → 2400×1260 PNG) for URL previews, so
// scrapers pick up the interactive diagram instead of the first <img> on the
// page (01's is the paper's own architecture figure). The widget is scaled
// to fit the card and centered on the page background. Checked-in output;
// regenerate after visual changes to these widgets:
//
//   node scripts/ogshot.mjs
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { execFile } from 'node:child_process';
import { extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromePath } from './chromepath.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.png': 'image/png' };
const CHROME = chromePath();
const W = 1200, H = 630, PAD = 24;

const SHOTS = [
  { page: 'studies/01-deepseek-diagram.html', sel: '#diagram', out: 'assets/og-01.png' },
  { page: 'studies/02-hopper-memory.html', sel: 'dsv3-anatomy[layer="local-diagram"]', out: 'assets/og-02.png' },
];

// Isolate the target: hide everything else (visibility sweep — an overlay
// cover loses to ancestor stacking contexts), unclip inner overflow boxes so
// wide diagrams paint fully, then fixed-position + scale the element so its
// PAINTED union (which can overflow the element box on both sides) fits the
// card. Coordinates are delta-corrected by re-measuring, since a transformed
// ancestor makes position:fixed resolve against it rather than the viewport.
const ISOLATE = (sel) => `
  await new Promise((r) => setTimeout(r, 700));   // let widgets mount (virtual time)
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
  for (const n of subtree)
    if (n.scrollWidth > n.clientWidth + 1 || n.scrollHeight > n.clientHeight + 1) n.style.overflow = 'visible';
  const prior = subtree.map((n) => n.style.visibility);
  for (const n of document.querySelectorAll('body *')) n.style.visibility = 'hidden';
  subtree.forEach((n, i) => { n.style.visibility = prior[i]; });
  for (let a = el; a && a !== document.body; a = a.parentElement) a.style.visibility = 'visible';
  const u = union();
  const k = Math.min((${W} - 2 * ${PAD}) / u.w, (${H} - 2 * ${PAD}) / u.h, 1);
  el.style.width = el.getBoundingClientRect().width + 'px';
  el.style.position = 'fixed';
  el.style.left = '0px'; el.style.top = '0px';
  el.style.transform = 'scale(' + k + ')';
  el.style.transformOrigin = 'top left';
  for (let i = 0; i < 2; i++) {   // translation-delta correction, twice for safety
    const q = union();
    el.style.left = parseFloat(el.style.left) + (${W} - q.w) / 2 - q.left + 'px';
    el.style.top = parseFloat(el.style.top) + (${H} - q.h) / 2 - q.top + 'px';
  }
`;

for (const { page, sel, out } of SHOTS) {
  const pageHtml = await readFile(join(root, page), 'utf8');
  const injected = pageHtml.replace('</body>',
    `<script type="module">(async () => {${ISOLATE(sel)}})();</script></body>`);
  const srv = createServer(async (req, res) => {
    try {
      const path = decodeURIComponent(new URL(req.url, 'http://x').pathname);
      if (path === '/' + page) { res.writeHead(200, { 'content-type': 'text/html' }); res.end(injected); return; }
      const body = await readFile(join(root, path));
      res.writeHead(200, { 'content-type': MIME[extname(path)] ?? 'application/octet-stream' });
      res.end(body);
    } catch { res.writeHead(404); res.end(); }
  });
  await new Promise((r) => srv.listen(0, r));
  const url = `http://localhost:${srv.address().port}/${page}`;
  await new Promise((res2, rej) => execFile(CHROME,
    ['--headless', '--disable-gpu', '--hide-scrollbars', '--virtual-time-budget=40000',
      `--window-size=${W},${H}`, '--force-device-scale-factor=2', `--screenshot=${join(root, out)}`, url],
    { maxBuffer: 64 * 1024 * 1024 }, (err) => err ? rej(err) : res2()));
  srv.close();
  console.log(`ogshot: ${out} (${W * 2}×${H * 2} from ${page} ${sel})`);
}
