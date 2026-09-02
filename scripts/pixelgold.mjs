// Pixel goldens: each distinct widget renderer, screenshotted at its resting
// state on the published pages and pinned in tests/pixel/*.png. Catches
// unintended formatting drift the numeric goldens can't see (colors, fonts,
// overlaps, spacing). Changing the look is allowed — invisibly changing it
// is not: any pixel drift fails, `--update` re-baselines, and the git diff
// plus the A/B report is the review artifact.
//
// Goldens are machine-tied (this Mac's fonts + Playwright's pinned
// chrome-headless-shell rasterizer): after a Chrome/OS update, expect a
// wholesale re-baseline — review the report, then --update.
//
//   node scripts/pixelgold.mjs             # compare (battery job)
//   node scripts/pixelgold.mjs --update    # re-baseline (writes the report first)
//   node scripts/pixelgold.mjs 02-final    # filter by name substring
//   node scripts/pixelgold.mjs --vs HEAD~1  # A/B the checked-in goldens against
//                                          # a git revision (no shooting) — the
//                                          # review tool for golden-changing commits
// On drift an A/B viewer is written: open /tmp/pixelgold-report.html
// (hold mouse or space = flip golden/new, d = red diff mask).
import { readFileSync, writeFileSync, existsSync, mkdirSync, readdirSync } from 'node:fs';
import { execSync } from 'node:child_process';
import { join } from 'node:path';
import { shoot, root } from './shotlib.mjs';
import { decode, encode } from './pngio.mjs';

const W = 1500, H = 2600;   // shot window; a widget touching its edge = enlarge it
const PAGE01 = 'studies/01-deepseek-diagram.html', PAGE02 = 'studies/02-hopper-memory.html';
const SHOTS = [
  { name: '01-diagram', page: PAGE01, sel: 'dsv3-anatomy[layer="diagram"]' },
  { name: '01-params', page: PAGE01, sel: 'dsv3-anatomy[layer="params-diagram"]' },
  { name: '02-snapshot', page: PAGE02, sel: 'dsv3-layer[comps="weights"]' },
  { name: '02-parts', page: PAGE02, sel: 'dsv3-layer[parts]' },
  { name: '02-ppsched', page: PAGE02, sel: 'dsv3-pp-schedule[sched="one"]' },
  { name: '02-ppsched-full', page: PAGE02, sel: 'dsv3-pp-schedule[layer="local-diagram"]' },
  { name: '02-ppfold', page: PAGE02, sel: 'dsv3-pp-fold' },
  { name: '02-ac', page: PAGE02, sel: 'dsv3-anatomy[layer="ac-layer"]' },
  { name: '02-fp8', page: PAGE02, sel: 'dsv3-anatomy[layer="fp8-layer"]' },
  { name: '02-final', page: PAGE02, sel: 'dsv3-anatomy[layer="local-diagram"]' },
  { name: '02-sheet', page: PAGE02, sel: 'dsv3-sheet' },
];

const update = process.argv.includes('--update');
const vsIdx = process.argv.indexOf('--vs');
const VS = vsIdx > -1 ? (process.argv[vsIdx + 1] ?? 'HEAD~1') : null;
const filters = process.argv.slice(2).filter((a, i) => !a.startsWith('--') && process.argv[2 + i - 1] !== '--vs');
// every shot has a night twin: same widget, theme flipped through setTheme()
const ALL = SHOTS.flatMap((s) => [s, { ...s, name: s.name + '-dark', dark: true }]);
const picked = ALL.filter((s) => !filters.length || filters.some((f) => s.name.includes(f)));
const GOLD = join(root, 'tests/pixel');
mkdirSync(GOLD, { recursive: true });

// trim the shot to its content (background = the corner pixel) + 8px apron
function trim(img) {
  const { w, h, data } = img;
  const bg = data.readUInt32BE(0);
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      if (data.readUInt32BE((y * w + x) * 4) !== bg) {
        if (x < x0) x0 = x; if (x > x1) x1 = x;
        if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
  if (x1 < 0) throw new Error('blank shot');
  if (x1 >= w - 1 || y1 >= h - 1) throw new Error(`content touches the ${W}x${H} window edge — enlarge W/H`);
  x0 = Math.max(0, x0 - 8); y0 = Math.max(0, y0 - 8); x1 = Math.min(w - 1, x1 + 8); y1 = Math.min(h - 1, y1 + 8);
  const tw = x1 - x0 + 1, th = y1 - y0 + 1;
  const out = Buffer.alloc(tw * th * 4);
  for (let y = 0; y < th; y++) data.copy(out, y * tw * 4, ((y + y0) * w + x0) * 4, ((y + y0) * w + x1 + 1) * 4);
  return { w: tw, h: th, data: out };
}

const results = [];
if (VS) {
  // review mode: current baselines vs a git revision — old plays 'golden',
  // current plays 'NEW' in the report
  const revNames = execSync(`git ls-tree -r --name-only ${VS} -- tests/pixel`, { cwd: root })
    .toString().trim().split('\n').filter(Boolean).map((p) => p.split('/').pop().slice(0, -4));
  const names = [...new Set([...readdirSync(GOLD).filter((f) => f.endsWith('.png')).map((f) => f.slice(0, -4)), ...revNames])]
    .filter((n) => !filters.length || filters.some((f) => n.includes(f)));
  for (const name of names.sort()) {
    const curPath = join(GOLD, name + '.png');
    const cur = existsSync(curPath) ? decode(readFileSync(curPath)) : null;
    let old = null;
    try { old = decode(execSync(`git show ${VS}:tests/pixel/${name}.png`, { cwd: root, maxBuffer: 64 * 1024 * 1024 })); } catch { /* absent at rev */ }
    if (!cur && !old) continue;
    let verdict = 'match', ndiff = 0, diffImg = null;
    if (!old) verdict = `new since ${VS}`;
    else if (!cur) verdict = `removed since ${VS}`;
    else if (old.w !== cur.w || old.h !== cur.h) verdict = `size ${old.w}x${old.h} → ${cur.w}x${cur.h}`;
    else if (!old.data.equals(cur.data)) {
      const d = Buffer.alloc(cur.w * cur.h * 4);
      for (let i = 0; i < d.length; i += 4)
        if (cur.data.readUInt32BE(i) !== old.data.readUInt32BE(i)) { ndiff++; d[i] = 255; d[i + 3] = 255; }
      verdict = `${ndiff} px differ`;
      diffImg = { w: cur.w, h: cur.h, data: d };
    }
    results.push({ name, img: cur ?? old, gold: old, verdict, diffImg });
  }
}
let next = 0;
if (!VS) await Promise.all(Array.from({ length: 4 }, async () => {
  while (next < picked.length) {
    const s = picked[next++];
    const tmp = join('/tmp', `pixelgold-${s.name}.png`);
    await shoot(s.page, s.sel, { origin: 24, dark: s.dark, w: W, h: H, dsf: 1, out: tmp });
    const img = trim(decode(readFileSync(tmp)));
    const goldPath = join(GOLD, s.name + '.png');
    const gold = existsSync(goldPath) ? decode(readFileSync(goldPath)) : null;
    let verdict = 'match', ndiff = 0, diffImg = null;
    if (!gold) verdict = 'new';
    else if (gold.w !== img.w || gold.h !== img.h) { verdict = `size ${gold.w}x${gold.h} → ${img.w}x${img.h}`; }
    else if (!gold.data.equals(img.data)) {
      const d = Buffer.alloc(img.w * img.h * 4);
      for (let i = 0; i < d.length; i += 4)
        if (img.data.readUInt32BE(i) !== gold.data.readUInt32BE(i)) { ndiff++; d[i] = 255; d[i + 3] = 255; }
      verdict = `${ndiff} px differ`;
      diffImg = { w: img.w, h: img.h, data: d };
    }
    results.push({ ...s, img, gold, verdict, diffImg, goldPath });
  }
}));

results.sort((a, b) => a.name.localeCompare(b.name));
const bad = results.filter((r) => r.verdict !== 'match');
for (const r of results) console.log(`${r.verdict === 'match' ? 'PASS' : update ? 'BASE' : 'FAIL'}  ${r.name}  (${r.img.w}x${r.img.h}${r.verdict === 'match' ? '' : ' — ' + r.verdict})`);


// ---- aligned A/B composition -----------------------------------------------
// Document screenshots REFLOW: one inserted row shifts everything below it,
// so a naive flip is unreadable past the first insertion. The fix is a text
// diff on pixels: hash each scanline, anchor on hashes UNIQUE in both images
// (patience-style, LIS for monotonic order), then compose both images at a
// COMMON height — gaps padded with tinted spacers so matched content sits at
// the same y in both panes. Gutter: green added · red removed · amber changed.
const GUTW = 8;
const GUTC = { eq: null, add: [46, 160, 67], del: [218, 54, 51], chg: [212, 153, 0] };
function alignPair(A, B) {
  const W2 = Math.max(A.w, B.w);
  const bg = [A.data[0], A.data[1], A.data[2]];
  const rowHash = (img, y) => {
    let h = 0x811c9dc5;
    const off = y * img.w * 4, end = off + img.w * 4;
    for (let i = off; i < end; i++) { h ^= img.data[i]; h = Math.imul(h, 0x01000193); }
    return h >>> 0;
  };
  const hashes = (img) => Array.from({ length: img.h }, (_, y) => rowHash(img, y));
  const ha = hashes(A), hb = hashes(B);
  const count = (arr) => { const m = new Map(); for (const h of arr) m.set(h, (m.get(h) ?? 0) + 1); return m; };
  const ca = count(ha), cb = count(hb);
  const posB = new Map();
  hb.forEach((h, i) => { if (cb.get(h) === 1) posB.set(h, i); });
  const cand = [];
  ha.forEach((h, i) => { if (ca.get(h) === 1 && posB.has(h)) cand.push([i, posB.get(h)]); });
  // LIS by b (cand is increasing in a) — O(n log n) tails
  const tails = [], back = new Array(cand.length), tidx = [];
  cand.forEach(([, b], i) => {
    let lo = 0, hi = tails.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (tails[mid] < b) lo = mid + 1; else hi = mid; }
    tails[lo] = b; tidx[lo] = i; back[i] = lo > 0 ? tidx[lo - 1] : -1;
  });
  const anchors = [];
  for (let i = tidx[tails.length - 1]; i != null && i >= 0; i = back[i]) anchors.unshift(cand[i]);
  if (anchors.length < 2) return null;   // nothing to anchor on — fall back to raw
  // walk anchors; between them, trim hash-equal prefix/suffix, pair the rest
  const rows = [];   // { a: y|null, b: y|null, g: 'eq'|'add'|'del'|'chg' }
  let pa = 0, pb = 0;
  const seg = (aEnd, bEnd) => {
    while (pa < aEnd && pb < bEnd && ha[pa] === hb[pb]) rows.push({ a: pa++, b: pb++, g: 'eq' });
    let sa = aEnd, sb = bEnd;
    while (sa > pa && sb > pb && ha[sa - 1] === hb[sb - 1]) { sa--; sb--; }
    const na = sa - pa, nb = sb - pb, n = Math.max(na, nb);
    for (let i = 0; i < n; i++) rows.push({
      a: i < na ? pa + i : null, b: i < nb ? pb + i : null,
      g: na && nb ? 'chg' : nb ? 'add' : 'del',
    });
    for (let y = sa; y < aEnd; y++) rows.push({ a: y, b: pb + (y - sa) + (sb - pb), g: 'eq' });
    pa = aEnd; pb = bEnd;
  };
  for (const [ai, bi] of anchors) {
    if (ai > pa || bi > pb) seg(ai, bi);
    else { pa = Math.max(pa, ai); pb = Math.max(pb, bi); }
    if (ai >= pa && bi >= pb) { rows.push({ a: ai, b: bi, g: 'eq' }); pa = ai + 1; pb = bi + 1; }
  }
  seg(A.h, B.h);
  // render both composites (gutter + padded rows); spacer = bg tinted 12%
  const H2 = rows.length, OW = GUTW + W2;
  const mk = () => Buffer.alloc(OW * H2 * 4);
  const outA = mk(), outB = mk();
  const put = (out, x, y, r, g2, b2) => { const o = (y * OW + x) * 4; out[o] = r; out[o + 1] = g2; out[o + 2] = b2; out[o + 3] = 255; };
  rows.forEach((r, y) => {
    const gc = GUTC[r.g];
    for (const [out, src, sy] of [[outA, A, r.a], [outB, B, r.b]]) {
      const tint = sy == null && gc;
      for (let x = 0; x < GUTW; x++) {
        const c = gc ?? bg;
        put(out, x, y, c[0], c[1], c[2]);
      }
      for (let x = 0; x < W2; x++) {
        if (sy != null && x < src.w) {
          const o = (sy * src.w + x) * 4;
          put(out, GUTW + x, y, src.data[o], src.data[o + 1], src.data[o + 2]);
        } else if (tint) put(out, GUTW + x, y, (bg[0] * 7 + gc[0]) >> 3, (bg[1] * 7 + gc[1]) >> 3, (bg[2] * 7 + gc[2]) >> 3);
        else put(out, GUTW + x, y, bg[0], bg[1], bg[2]);
      }
    }
  });
  // sparse diff mask on the ALIGNED pair (content columns only)
  const d = Buffer.alloc(OW * H2 * 4);
  let ndiff = 0;
  rows.forEach((r, y) => {
    if (r.a == null || r.b == null) return;
    for (let x = 0; x < W2; x++) {
      const oa = (r.a * A.w + x) * 4, ob = (r.b * B.w + x) * 4;
      const da = x < A.w ? A.data.readUInt32BE(oa) : 0, db = x < B.w ? B.data.readUInt32BE(ob) : 0;
      if (da !== db) { ndiff++; const o = (y * OW + GUTW + x) * 4; d[o] = 255; d[o + 3] = 255; }
    }
  });
  const added = rows.filter((r) => r.a == null).length, removed = rows.filter((r) => r.b == null).length;
  return { a: { w: OW, h: H2, data: outA }, b: { w: OW, h: H2, data: outB },
    diff: { w: OW, h: H2, data: d }, added, removed, ndiff };
}

if (bad.length) {   // the A/B viewer: written before any --update overwrites the baseline
  const b64 = (i) => 'data:image/png;base64,' + encode(i.w, i.h, i.data).toString('base64');
  const html = `<!DOCTYPE html><meta charset="utf-8"><title>pixelgold A/B</title>
<style>body{font:14px system-ui;margin:20px;background:#f4f4f2}
.hint{position:sticky;top:0;background:#fff;border:1px solid #ccc;border-radius:6px;padding:8px 12px;z-index:9}
.ab{position:relative;display:inline-block;border:1px solid #bbb;background:#fff}
.ab img{display:block}.ab img.B,.ab img.D{position:absolute;left:0;top:0;display:none}
body.flip .ab img.B{display:block}body.diff .ab img.D{display:block}
.ab::after{content:'golden';position:absolute;right:4px;top:4px;background:#0008;color:#fff;padding:1px 7px;border-radius:4px;font-size:12px}
body.flip .ab::after{content:'NEW';background:#c40}</style>
<div class="hint"><b>space / n</b> = LOCK on new (scroll &amp; eyeball) · <b>hold mouse</b> = momentary flip · <b>d</b> = red diff mask · gutter: <span style="color:#2ea043">■ added</span> <span style="color:#da3633">■ removed</span> <span style="color:#d49900">■ changed</span> (panes are row-aligned: spacer bands hold reflowed content in place)</div>` +
    bad.map((r) => {
      // reflow-aware panes: align on unique scanlines so the flip stays
      // locked below insertions; fall back to the raw pair when unanchorable
      const al = r.gold && r.img && r.gold !== r.img ? alignPair(r.gold, r.img) : null;
      const [pa, pb, pd] = al ? [al.a, al.b, al.diff] : [r.gold ?? r.img, r.img, r.diffImg];
      const note = al ? ` · aligned: +${al.added} / −${al.removed} rows, ${al.ndiff} px differ` : '';
      return `<h2>${r.name} — ${r.verdict}${note}</h2><div class="ab"><img src="${b64(pa)}"><img class="B" src="${b64(pb)}">${pd ? `<img class="D" src="${b64(pd)}">` : ''}</div>`;
    }).join('\n') +
    `<script>let lock=false,hold=false;
const sync=()=>document.body.classList.toggle('flip',lock!==hold);
addEventListener('keydown',(e)=>{if(e.key==='d')document.body.classList.toggle('diff');
if(e.key===' '||e.key==='n'){lock=!lock;sync();e.preventDefault()}});
addEventListener('mousedown',()=>{hold=true;sync()});
addEventListener('mouseup',()=>{hold=false;sync()});</script>`;
  writeFileSync('/tmp/pixelgold-report.html', html);
  console.log(`\nA/B viewer: open /tmp/pixelgold-report.html`);
}
if (VS) {
  console.log(`pixelgold: ${bad.length ? bad.length + ' golden(s) differ from ' + VS : 'identical to ' + VS} (review mode — not a gate)`);
} else if (update) {
  for (const r of bad) writeFileSync(r.goldPath, encode(r.img.w, r.img.h, r.img.data));
  console.log(`pixelgold: baselined ${bad.length ? bad.map((r) => r.name).join(', ') : 'nothing (all matched)'}`);
} else if (bad.length) {
  console.log(`pixelgold: ${bad.length} drift(s) — intentional? review the A/B report, then \`node scripts/pixelgold.mjs --update\``);
  process.exit(1);
} else console.log(`pixelgold: ${results.length} widgets match`);
