// Resolve the headless browser for the test drivers. Prefer Playwright's
// chrome-headless-shell (starts in ~0.2 s vs ~2 s, exits cleanly, never
// touches the user's Chrome profile or Dock); fall back to the Google Chrome
// app. $CHROME overrides everything.
import { readdirSync, existsSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';

export function chromePath() {
  if (process.env.CHROME) return process.env.CHROME;
  for (const cache of [join(homedir(), 'Library/Caches/ms-playwright'), join(homedir(), '.cache/ms-playwright')]) {
    try {
      const latest = readdirSync(cache).filter(d => d.startsWith('chromium_headless_shell-')).sort().at(-1);
      if (!latest) continue;
      const shell = readdirSync(join(cache, latest)).map(d => join(cache, latest, d, 'chrome-headless-shell')).find(existsSync);
      if (shell) return shell;
    } catch {}
  }
  return '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
}
