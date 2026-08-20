// Chrome Trace Event Format export (loads in ui.perfetto.dev / chrome://tracing),
// plus browser helpers to download or hand the trace to a real Perfetto tab.

export function toChromeTrace(trace, title = 'dsv3-sim') {
  const events = [];
  trace.ranks.forEach((rank, pid) => {
    events.push({ ph: 'M', pid, tid: 0, name: 'process_name', args: { name: rank.label } });
    events.push({ ph: 'M', pid, tid: 0, name: 'process_sort_index', args: { sort_index: pid } });
    rank.tracks.forEach((track, tid) => {
      events.push({ ph: 'M', pid, tid, name: 'thread_name', args: { name: track.name } });
      events.push({ ph: 'M', pid, tid, name: 'thread_sort_index', args: { sort_index: tid } });
      for (const s of track.slices) {
        events.push({ ph: 'X', pid, tid, ts: s.ts, dur: s.dur, name: s.name, cat: s.cat, args: s.args });
      }
    });
  });
  return {
    traceEvents: events,
    displayTimeUnit: 'ms',
    metadata: { title, sim: trace.meta ? { stats: trace.meta.stats, cfg: strippedCfg(trace.meta.cfg) } : undefined },
  };
}

function strippedCfg({ refinements, ...cfg }) { return cfg; }

export function downloadTrace(trace, title = 'dsv3-sim') {
  const blob = new Blob([JSON.stringify(toChromeTrace(trace, title))], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${title}.json`;
  a.click();
  URL.revokeObjectURL(a.href);
}

// Perfetto UI deep-link: open the tab, ping until it answers PONG, post the buffer.
export function openInPerfetto(trace, title = 'dsv3-sim') {
  const buffer = new TextEncoder().encode(JSON.stringify(toChromeTrace(trace, title))).buffer;
  const win = window.open('https://ui.perfetto.dev');
  if (!win) return;
  const onMsg = (e) => {
    if (e.data !== 'PONG' || e.source !== win) return;
    clearInterval(timer);
    window.removeEventListener('message', onMsg);
    win.postMessage({ perfetto: { buffer, title, fileName: `${title}.json` } }, 'https://ui.perfetto.dev');
  };
  window.addEventListener('message', onMsg);
  const timer = setInterval(() => { try { win.postMessage('PING', 'https://ui.perfetto.dev'); } catch { } }, 300);
  setTimeout(() => { clearInterval(timer); window.removeEventListener('message', onMsg); }, 30000);
}
