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
// On drift an A/B viewer is written: open /tmp/pixelgold-report.html
// (hold mouse or space = flip golden/new, d = red diff mask).
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
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
const filters = process.argv.slice(2).filter((a) => !a.startsWith('--'));
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
let next = 0;
await Promise.all(Array.from({ length: 4 }, async () => {
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
<div class="hint"><b>hold mouse / space</b> = flip golden ↔ new · <b>d</b> = red diff mask</div>` +
    bad.map((r) => `<h2>${r.name} — ${r.verdict}</h2><div class="ab"><img src="${r.gold ? b64(r.gold) : b64(r.img)}"><img class="B" src="${b64(r.img)}">${r.diffImg ? `<img class="D" src="${b64(r.diffImg)}">` : ''}</div>`).join('\n') +
    `<script>addEventListener('keydown',e=>{if(e.key==='d')document.body.classList.toggle('diff');if(e.key===' '){document.body.classList.toggle('flip');e.preventDefault()}});
addEventListener('mousedown',()=>document.body.classList.add('flip'));
addEventListener('mouseup',()=>document.body.classList.remove('flip'));</script>`;
  writeFileSync('/tmp/pixelgold-report.html', html);
  console.log(`\nA/B viewer: open /tmp/pixelgold-report.html`);
}
if (update) {
  for (const r of bad) writeFileSync(r.goldPath, encode(r.img.w, r.img.h, r.img.data));
  console.log(`pixelgold: baselined ${bad.length ? bad.map((r) => r.name).join(', ') : 'nothing (all matched)'}`);
} else if (bad.length) {
  console.log(`pixelgold: ${bad.length} drift(s) — intentional? review the A/B report, then \`node scripts/pixelgold.mjs --update\``);
  process.exit(1);
} else console.log(`pixelgold: ${results.length} widgets match`);
