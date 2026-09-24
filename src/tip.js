// The site's one data tooltip (docs/widgets.md § Tooltips). Hover a tipped
// element → a light card follows the pointer; click → it PINS (amber
// border, selectable; clicks inside it never close it); the next click
// anywhere else, or Escape, closes it (and only closes: never re-pins).
// Clicks that already mean something (buttons, links, form controls, or
// whatever `pinnable` refuses) don't pin.
// Widgets supply content, never behavior:
//   attachTip(host, content, { parent = host, cls, pinnable }) → { tip, pinned, set, unpin }
// content(ev, pinning) → null | string | Node for the element under ev; it is
// called on hover, and again with pinning = true (tip already .pinned) on
// the click that pins. The card lives in `parent` (keep it outside anything
// the widget re-renders) and is placed from the pointer, whatever its
// containing block.
const CSS = `
.dsv3-tip { position: absolute; display: none; pointer-events: none; z-index: 7; max-width: 360px;
  background: var(--c-ffffff); color: var(--c-1c1c1a); padding: 6px 9px; border: 1px solid var(--c-c3c2b7);
  border-radius: 5px; box-shadow: 0 2px 10px rgba(11,11,11,0.12); text-align: left; white-space: pre-line;
  font: 11.5px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; font-variant-numeric: tabular-nums; }
.dsv3-tip.pinned { pointer-events: auto; border-color: var(--c-eda100); box-shadow: 0 2px 10px rgba(237,161,0,0.3); }
`;
const NO_PIN = 'button, select, input, label, a';
let styled = false;
let open = 0;   // pinned tips page-wide: while any is pinned, a click only closes

export function attachTip(host, content, { parent = host, cls = '', pinnable = () => true } = {}) {
  if (!styled) {
    styled = true;
    const s = document.createElement('style'); s.textContent = CSS;
    document.head.append(s);
  }
  const tip = parent.appendChild(document.createElement('div'));
  tip.className = cls ? `dsv3-tip ${cls}` : 'dsv3-tip';
  let pinned = false;
  const set = (c) => { if (typeof c === 'string') tip.textContent = c; else tip.replaceChildren(c); };
  const show = (ev, c) => {
    if (c == null || c === '') { tip.style.display = 'none'; return; }
    set(c);
    tip.style.display = 'block'; tip.style.left = '0px';   // measure unsqueezed
    const p = tip.offsetParent;
    const r = p && p !== document.body ? p.getBoundingClientRect() : { left: -scrollX, top: -scrollY };
    const [bl, bt] = p && p !== document.body ? [p.clientLeft, p.clientTop] : [0, 0];
    const x = Math.min(ev.clientX + 14, document.documentElement.clientWidth - tip.offsetWidth - 8);
    tip.style.left = x - r.left - bl + 'px';
    tip.style.top = ev.clientY + 14 - r.top - bt + 'px';
  };
  const unpin = () => { if (pinned) open--; pinned = false; tip.classList.remove('pinned'); tip.style.display = 'none'; };
  host.addEventListener('mousemove', (ev) => { if (!pinned) show(ev, content(ev, false)); });
  host.addEventListener('mouseleave', () => { if (!pinned) tip.style.display = 'none'; });
  // capture: runs before the widget's own click handlers (which may re-render
  // the target) and can't be swallowed by their stopPropagation
  document.addEventListener('click', (ev) => {
    if (tip.contains(ev.target)) return;
    if (pinned) { unpin(); ev.dsv3TipClosed = true; return; }
    if (open || ev.dsv3TipClosed) return;
    if (!host.contains(ev.target) || ev.target.closest?.(NO_PIN) || !pinnable(ev)) return;
    if (!getSelection().isCollapsed) return;   // the click that ends a text selection
    pinned = true; open++; tip.classList.add('pinned');
    const c = content(ev, true);
    if (c == null || c === '') unpin(); else show(ev, c);
  }, true);
  document.addEventListener('keydown', (ev) => { if (ev.key === 'Escape' && pinned) unpin(); });
  return { tip, get pinned() { return pinned; }, set, unpin };
}
