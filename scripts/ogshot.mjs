// og:image prerenders: screenshot each post's signature widget onto a
// 1200×630 card (rendered at 2× → 2400×1260 PNG) for URL previews, so
// scrapers pick up the interactive diagram instead of the first <img> on the
// page (01's is the paper's own architecture figure). The widget is scaled
// to fit the card and centered on the page background. Checked-in output;
// regenerate after visual changes to these widgets:
//
//   node scripts/ogshot.mjs
import { join } from 'node:path';
import { shoot, root } from './shotlib.mjs';

const W = 1200, H = 630, PAD = 24;
const SHOTS = [
  { page: 'studies/01-deepseek-diagram.html', sel: '#diagram', out: 'assets/og-01.png' },
  { page: 'studies/02-hopper-memory.html', sel: 'dsv3-anatomy[layer="local-diagram"]', out: 'assets/og-02.png' },
];

for (const { page, sel, out } of SHOTS) {
  await shoot(page, sel, { card: { W, H, PAD }, unclip: true, w: W, h: H, dsf: 2, out: join(root, out) });
  console.log(`ogshot: ${out} (${W * 2}×${H * 2} from ${page} ${sel})`);
}
