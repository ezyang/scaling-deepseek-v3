// Cache busting for GitHub Pages, which serves everything with a fixed
// Cache-Control: max-age=600 and no way to customize headers: for up to ten
// minutes after a deploy, a plain reload revalidates the HTML but reuses
// cached JS/CSS — fresh prose driving stale widgets. Fix: every module gets
// a content-hash query (?v=xxxxxxxx) via an import map in each public page.
// Import maps remap nested `import` specifiers too (static and dynamic), so
// one map in the always-revalidated HTML busts the whole module graph, and
// unchanged files keep their hash and stay cached. Like the min-height
// placeholders, the stamps are generated values pasted into the HTML; the
// battery's `stamp` job pins their freshness.
//
//   node scripts/stamp.mjs           # restamp index.html + studies/NN-*.html
//   node scripts/stamp.mjs --check   # exit 1 if any stamp is stale
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { join, relative, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('..', import.meta.url));
const check = process.argv.includes('--check');
const hash = (path) => createHash('sha1').update(readFileSync(join(root, path))).digest('hex').slice(0, 8);

const modules = ['src', 'studies'].flatMap((dir) =>
  readdirSync(join(root, dir)).filter((f) => f.endsWith('.js')).map((f) => dir + '/' + f));
const pages = ['index.html',
  ...readdirSync(join(root, 'studies')).filter((f) => /^\d\d-.*\.html$/.test(f)).map((f) => 'studies/' + f)];

const stale = [];
for (const page of pages) {
  const dir = dirname(page);
  const rel = (p) => { const r = relative(dir, p); return r.startsWith('.') ? r : './' + r; };
  const imports = Object.fromEntries(modules.map((p) => [rel(p), `${rel(p)}?v=${hash(p)}`]));
  const map = `<script type="importmap">\n  ${JSON.stringify({ imports }, null, 2).replace(/\n/g, '\n  ')}\n  </script>`;

  let html = readFileSync(join(root, page), 'utf8');
  const orig = html;
  html = html.replace(/href="([^"?]+\.css)(?:\?v=[0-9a-f]+)?"/g, (_, href) =>
    `href="${href}?v=${hash(join(dir, href))}"`);
  if (html.includes('<script type="importmap">')) {
    html = html.replace(/<script type="importmap">[\s\S]*?<\/script>/, map);
  } else {
    html = html.replace(/\n(\s*)<script type="module">/,
      `\n$1<!-- cache stamps: regenerate with \`node scripts/stamp.mjs\` -->\n$1${map}\n$1<script type="module">`);
  }
  if (html !== orig) {
    stale.push(page);
    if (!check) writeFileSync(join(root, page), html);
  }
}
if (check && stale.length) {
  console.error(`stamp: STALE — run \`node scripts/stamp.mjs\` (${stale.join(', ')})`);
  process.exit(1);
}
console.log(`stamp: ${stale.length ? (check ? 'stale: ' : 'updated ') + stale.join(', ') : `fresh (${pages.length} pages)`}`);
