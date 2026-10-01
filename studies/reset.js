// Floating reset (top right): one click returns every widget on the page to
// its defaults. Widgets keep their state in location.hash and write only
// departures from their defaults (an untouched page has no hash), so
// "modified" is just a non-empty hash — shown as a dot on the button.
// Widgets don't tolerate re-mounting, so reset clears the hash and reloads
// (the browser restores the scroll position).
// Hovering the button lists what's changed, one line per edit: click a line
// to go there, ↺ to undo just that one. A module that can name its edits
// registers describe(hashKey, () => [{ html, el, revert }]); any other hash
// key is one line for its widget (the element whose id follows the key's
// "x:" prefix), undone by dropping the key and reloading.
const b = document.createElement('button');
b.className = 'resetb'; b.type = 'button'; b.textContent = '↺';
b.setAttribute('aria-label', 'reset every widget to its default');
const p = document.createElement('div');
p.className = 'resetp';
const describers = new Map();
export const describe = (key, fn) => { describers.set(key, fn); sync(); };

const heading = (el) => [...document.querySelectorAll('main h2, main h3')]
  .filter((h) => h.compareDocumentPosition(el) & Node.DOCUMENT_POSITION_FOLLOWING).pop();
function widgetLine(key) {
  const el = document.getElementById(key.slice(key.indexOf(':') + 1));
  const h = el && heading(el);
  // figures in the same section: say which (by order)
  const peers = h ? [...document.querySelectorAll('main [id]')].filter((w) => w.tagName.startsWith('DSV3-') && heading(w) === h) : [];
  const n = peers.indexOf(el) + 1, th = n % 100 > 10 && n % 100 < 14 ? 'th' : ['th', 'st', 'nd', 'rd'][n % 10] ?? 'th';
  return {
    html: h ? `${peers.length > 1 ? `${n}${th} ` : ''}figure in <i>${h.textContent}</i>` : `figure <i>${key}</i>`,
    el,
    revert: () => {
      const q = new URLSearchParams(location.hash.slice(1)); q.delete(key);
      history.replaceState(null, '', q.size ? '#' + q : location.pathname + location.search);
      location.reload();
    },
  };
}
function sync() {
  const keys = [...new URLSearchParams(location.hash.slice(1)).keys()];
  b.classList.toggle('mod', keys.length > 0); b.disabled = !keys.length;
  const lines = keys.flatMap((k) => describers.get(k)?.() ?? [widgetLine(k)]);
  const card = document.createElement('div');
  card.className = 'resetc';
  card.innerHTML = '<div class="hd">changed on this page · ↺ resets all</div>';
  for (const ln of lines) {
    const row = document.createElement('div');
    row.className = 'ln';
    row.innerHTML = `<span class="go">${ln.html}</span><button type="button" class="un" title="undo this one">↺</button>`;
    row.querySelector('.go').onclick = () => ln.el?.scrollIntoView({ block: 'center' });
    row.querySelector('.un').onclick = () => ln.revert();
    card.append(row);
  }
  p.replaceChildren(card);
}
b.onclick = () => {
  history.replaceState(null, '', location.pathname + location.search);
  location.reload();
};
// every widget writes through history.replaceState
const replace = history.replaceState.bind(history);
history.replaceState = (...a) => { replace(...a); sync(); };
addEventListener('hashchange', sync);
document.body.append(b, p);
sync();
