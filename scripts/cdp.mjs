// Persistent headless Chrome for the test drivers, over the DevTools
// protocol on --remote-debugging-pipe: ONE browser per driver process, a
// fresh browser context (own renderer, own localStorage) per page load.
// Launching a browser per test cost ~0.2 CPU-s and four execs each — a
// fifth of the battery — plus the occasional multi-second cold start.
//
// Virtual time mirrors the old `--virtual-time-budget` CLI path: policy
// pauseIfNetworkFetchesPending, so timers fast-forward and T.tick(700)
// costs nothing when the page is idle.
//
//   const b = await launch();
//   const p = await b.open(url, { width, height, budget, bindings: ['__done'] });
//   const r = await p.until('__done');   // binding payload, or null if the budget ran out
//   await p.close(); b.close();
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { extname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromePath } from './chromepath.mjs';

export const root = fileURLToPath(new URL('..', import.meta.url));
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.png': 'image/png' };

// Static server for the repo. inject(path, html) serves `html` in place of
// the page at its own path (so relative imports still resolve), keyed by a
// ?__job query so concurrent jobs on one page don't collide. Bound to
// 127.0.0.1, never the wildcard: a local daemon holding ::1 on an
// ephemeral port would otherwise answer `localhost` and wedge the page.
export async function serve() {
  const jobs = new Map();
  let n = 0;
  const srv = createServer(async (req, res) => {
    try {
      const u = new URL(req.url, 'http://x');
      const job = jobs.get(u.searchParams.get('__job'));
      if (job) { res.writeHead(200, { 'content-type': 'text/html' }); res.end(job); return; }
      const path = decodeURIComponent(u.pathname);
      const body = await readFile(join(root, path));
      res.writeHead(200, { 'content-type': MIME[extname(path)] ?? 'application/octet-stream' });
      res.end(body);
    } catch { res.writeHead(404); res.end(); }
  });
  await new Promise((r) => srv.listen(0, '127.0.0.1', r));
  const origin = `http://127.0.0.1:${srv.address().port}`;
  return {
    url: (path) => `${origin}/${path}`,
    inject(path, html) {
      const id = String(++n);
      jobs.set(id, html);
      return { url: `${origin}/${path}?__job=${id}`, done: () => jobs.delete(id) };
    },
    close: () => srv.close(),
  };
}

export async function launch(flags = []) {
  const ch = spawn(chromePath(), ['--headless', '--disable-gpu', '--hide-scrollbars', '--remote-debugging-pipe', ...flags],
    { stdio: ['ignore', 'ignore', 'ignore', 'pipe', 'pipe'] });
  const pending = new Map(), handlers = new Set();
  let id = 0, chunks = [];
  ch.stdio[4].on('data', (d) => {
    let i;
    while ((i = d.indexOf(0)) >= 0) {   // messages are NUL-terminated JSON
      chunks.push(d.subarray(0, i));
      const msg = JSON.parse(Buffer.concat(chunks).toString());
      chunks = []; d = d.subarray(i + 1);
      if (msg.id) {
        const p = pending.get(msg.id); pending.delete(msg.id);
        msg.error ? p.rej(new Error(`${p.method}: ${msg.error.message}`)) : p.res(msg.result);
      } else for (const h of handlers) h(msg);
    }
    if (d.length) chunks.push(d);
  });
  let dead = null;
  ch.on('exit', (code) => {
    dead = new Error(`chrome exited (${code})`);
    for (const p of pending.values()) p.rej(dead);
    for (const h of handlers) h({ dead });
  });
  const send = (method, params = {}, sessionId) => dead ? Promise.reject(dead) : new Promise((res, rej) => {
    pending.set(++id, { res, rej, method });
    ch.stdio[3].write(JSON.stringify({ id, method, params, sessionId }) + '\0');
  });

  async function open(url, { width = 1500, height = 4000, budget = 40000, bindings = [] } = {}) {
    const { browserContextId } = await send('Target.createBrowserContext');
    const { targetId } = await send('Target.createTarget', { url: 'about:blank', width, height, browserContextId });
    const { sessionId: s } = await send('Target.attachToTarget', { targetId, flatten: true });
    // events for this page, buffered until someone awaits them
    const seen = [], waiters = [];
    const h = (m) => {
      if (!m.dead && m.sessionId !== s) return;
      seen.push(m);
      for (const w of waiters.splice(0)) w();
    };
    handlers.add(h);
    if (bindings.length) await send('Runtime.enable', {}, s);
    for (const name of bindings) await send('Runtime.addBinding', { name }, s);
    // budget starts once the navigation commits: on about:blank nothing is
    // pending, so a budget set earlier would expire before the fetch begins
    await send('Emulation.setVirtualTimePolicy', { policy: 'pause' }, s);
    await send('Page.navigate', { url }, s);
    await send('Emulation.setVirtualTimePolicy', { policy: 'pauseIfNetworkFetchesPending', budget }, s);
    return {
      send: (method, params) => send(method, params, s),
      // resolve with the named binding's payload, or null once the virtual
      // time budget expires (with no binding name: just wait out the budget).
      // The real-time cap catches a wedged page (virtual time frozen on a
      // fetch that never completes) — fail loudly, never hang the battery.
      async until(binding, ms = 60_000) {
        const t = setTimeout(() => { seen.push({ timeout: true }); for (const w of waiters.splice(0)) w(); }, ms);
        try {
          for (let i = 0; ; ) {
            for (; i < seen.length; i++) {
              const m = seen[i];
              if (m.dead) throw m.dead;
              if (m.timeout) throw new Error(`page still running after ${ms / 1000} s real time: ${url}`);
              if (m.method === 'Runtime.bindingCalled' && m.params.name === binding) return m.params.payload;
              if (m.method === 'Emulation.virtualTimeBudgetExpired') return null;
            }
            await new Promise((r) => waiters.push(r));
          }
        } finally { clearTimeout(t); }
      },
      async eval(expression) {
        const r = await send('Runtime.evaluate', { expression, returnByValue: true }, s);
        return r.result.value;
      },
      // fast: bigger file, ~4x cheaper to encode and decode (same pixels)
      async screenshot({ fast = false } = {}) {
        return Buffer.from((await send('Page.captureScreenshot', { format: 'png', optimizeForSpeed: fast }, s)).data, 'base64');
      },
      async close() {
        handlers.delete(h);
        await send('Target.disposeBrowserContext', { browserContextId }).catch(() => {});
      },
    };
  }
  return { open, close: () => ch.kill() };
}
