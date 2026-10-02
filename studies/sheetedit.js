// 03's static cell sheets, made live. The formula column IS the program:
// every value on screen is computed from it, the HTML's numbers are only the
// pre-module fallback (tests/sheetedit.js fails while one is stale). Each
// formula's HTML compiles to exact rational arithmetic, so editing a real
// knob (PAGES below names them per page: 03's B, S, GPUs, EP, NVL, n_c and the
// rates the model runs at; the spec compute peaks are fixed)
// recomputes every dependent row, wherever it's repeated. Rows that depart
// from the untouched sheet turn amber. Edits live in the hash (departures
// only), so the floating reset (reset.js) lists and undoes them. Prose numbers
// that quote a cell (data-cellref = the row's name) follow along; each render
// fires 'dsv3-cells' so page scripts can follow too (current(), setKnob()).
// The page's .sheet-dl button (static HTML, at the bottom) downloads every
// sheet as one .xlsx with live formulas (names → C-column addresses), at the
// current knob values.
import { describe } from './reset.js';
import { downloadXlsx, xesc } from '../src/xlsx.js';

// per page (<body data-sheet>): which rows are knobs (name → hash key), which
// must be whole numbers, the cross-row rules an edit must keep, the .xlsx name
const PAGES = {
  '03': {
    knobs: { B: 'B', S: 'S', GPUs: 'GPUs', 'π<sup>sol</sup><sub>bf16</sub>': 'sol', 'π<sup>sol</sup><sub>fp8</sub>': 'sol8', EP: 'EP', NVL: 'NVL', 'β<sub>IB</sub>': 'IB', 'n<sub>c</sub>': 'SMc' },
    counts: ['B', 'S', 'GPUs', 'EP', 'NVL', 'SMc'],
    xlsx: ['dsv3-roofline-sheet.xlsx', 'DSv3 roofline'],
    rules(v, edits, pub) {
      const [G, E, N] = ['GPUs', 'EP', 'NVL'].map((k) => v(k)[0]);
      if (G % E) return `EP must divide GPUs (${G.toLocaleString('en-US')})`;
      if (E % N) return `NVL must divide EP (${E.toLocaleString('en-US')})`;
      if (E < 4n * N) return 'EP must span at least 4 nodes (EP ≥ 4 · NVL): M = 4 assumes a token can reach 4 nodes';
      const sms = pub.get('n<sub>SM</sub>')?.[0];
      if (edits.SMc?.[0] > sms) return `an H800 has ${sms} SMs`;
      return null;
    },
  },
  k3: {
    knobs: { B: 'B', S: 'S', GPUs: 'GPUs', 'π<sup>sol</sup><sub>bf16</sub>': 'sol', 'π<sup>sol</sup><sub>fp8</sub>': 'sol8', 'β<sub>NV</sub>': 'NV', 'β<sub>IB</sub>': 'IB', NVL: 'NVL', PP: 'PP', EP: 'EP', 'V<sub>pp</sub>': 'Vpp', 'r<sub>res</sub>': 'rres', ovl: 'ovl', 'μ': 'mu', 'g<sub>B</sub>': 'gB', 'o<sub>B</sub>': 'oB', 'β<sub>host</sub>': 'host' },
    counts: ['B', 'S', 'GPUs', 'NVL', 'PP', 'EP', 'Vpp', 'rres', 'ovl', 'gB', 'oB'],
    zero: ['ovl', 'mu'],   // knobs that may be 0 (a 0/1 switch, a share)
    xlsx: ['kimi-k3-roofline-sheet.xlsx', 'Kimi K3 roofline'],
    rules(v, edits, pub) {
      const [G, P, E, N] = ['GPUs', 'PP', 'EP', 'NVL'].map((k) => v(k)[0]);
      if (N % E) return `EP must divide NVL (${N}): EP lives inside one NVLink domain`;
      if (G % (P * E)) return `PP · EP must divide GPUs (${G.toLocaleString('en-US')})`;
      const L = pub.get('L')?.[0];
      if (P > L) return `the model has ${L} layers`;
      return null;
    },
  },
};
const PAGE = PAGES[document.body.dataset.sheet ?? '03'];
const KNOBS = PAGE.knobs;
const COUNTS = new Set(PAGE.counts);
const KEY = 'c:cells';

// ---- exact rationals: [num, den] BigInts, den > 0, reduced ----------------
const gcd = (a, b) => { a = a < 0n ? -a : a; while (b) [a, b] = [b, a % b]; return a; };
const Q = (n, d = 1n) => { if (d < 0n) { n = -n; d = -d; } const g = gcd(n, d) || 1n; return [n / g, d / g]; };
const OPS = {
  '+': ([a, b], [c, d]) => Q(a * d + c * b, b * d),
  '−': ([a, b], [c, d]) => Q(a * d - c * b, b * d),
  '·': ([a, b], [c, d]) => Q(a * c, b * d),
  '/': ([a, b], [c, d]) => Q(a * d, b * c),
};
const ceil = ([n, d]) => Q(n >= 0n ? (n + d - 1n) / d : n / d);
const eq = (x, y) => x[0] === y[0] && x[1] === y[1];
const dec = (s) => { const [i, f = ''] = s.replace(/,/g, '').split('.'); return Q(BigInt(i + f), 10n ** BigInt(f.length)); };
const num = ([n, d]) => Number(n) / Number(d);

// the formula cell's HTML → (get) => Q. <i>…</i> is commentary; <b>X</b> is a
// reference to the row named X; 10<sup>k</sup> is a literal; ⌈…⌉ and max(…, …)
// are the only functions (both have Excel spellings for the .xlsx)
function compile(html) {
  const s = html.replace(/<i>[\s\S]*?<\/i>/g, '').replace(/10<sup>(\d+)<\/sup>/g, (_, e) => '1' + '0'.repeat(+e)).trim();
  const re = /\s*(?:<b>([\s\S]*?)<\/b>|(\d(?:[\d,]*\d)?(?:\.\d+)?)|(max\(|[·/+−(),⌈⌉]))/y;
  const toks = [];
  while (re.lastIndex < s.length) {
    const at = re.lastIndex, m = re.exec(s);
    if (!m) throw new Error(`sheetedit: can't read "${s.slice(at)}"`);
    toks.push(m[1] != null ? { ref: m[1] } : m[2] != null ? { q: dec(m[2]) } : { op: m[3] });
  }
  let i = 0;
  const peek = (op) => toks[i]?.op === op;
  const want = (op) => { if (!peek(op)) throw new Error(`sheetedit: expected ${op} in "${s}"`); i++; };
  const atom = () => {
    const t = toks[i++];
    if (t?.q) return () => t.q;
    if (t?.ref != null) return (get) => get(t.ref);
    if (t?.op === '(') { const e = expr(); want(')'); return e; }
    if (t?.op === '⌈') { const e = expr(); want('⌉'); return (get) => ceil(e(get)); }
    if (t?.op === 'max(') { const x = expr(); want(','); const y = expr(); want(')'); return (get) => { const a = x(get), b = y(get); return a[0] * b[1] >= b[0] * a[1] ? a : b; }; }
    throw new Error(`sheetedit: unexpected token in "${s}"`);
  };
  const chain = (sub, ops) => () => {
    let a = sub();
    while (ops.some(peek)) { const op = toks[i++].op, x = a, y = sub(); a = (get) => OPS[op](x(get), y(get)); }
    return a;
  };
  const term = chain(atom, ['·', '/']);
  const expr = chain(term, ['+', '−']);
  const f = expr();
  if (i !== toks.length) throw new Error(`sheetedit: trailing tokens in "${s}"`);
  return f;
}

// ---- the sheets' own number formats ----------------------------------------
// exact column: the decimal when it terminates within 9 places, else blank —
// a rounded figure isn't exact (the ≈ columns carry it). Typed knobs always
// terminate, so they print in full (places = Infinity)
export function fmtExact([n, d], places = 9) {
  const neg = n < 0n; if (neg) n = -n;
  let k = 0;
  while ((n * 10n ** BigInt(k)) % d) if (++k > places) return '';
  const m = 10n ** BigInt(k), r = n * m / d;
  return (neg ? '−' : '') + (r / m).toLocaleString('en-US') + (k ? '.' + (r % m).toString().padStart(k, '0') : '');
}
const fmtKnob = (q) => fmtExact(q, Infinity);
const fmtSci = (x) => { const [m, e] = x.toExponential(2).split('e'); return `${m} × 10<sup>${(+e).toString().replace('-', '−')}</sup>`; };
const PFX = ['', 'K', 'M', 'G', 'T', 'P', 'E'];
// → [scaled value, prefix, base unit, power of ten applied] (the .xlsx's SI column reuses it)
function siScale(x, unit) {
  const base = unit === 'ms' || unit === 'µs' ? 's' : unit === '%' || unit === 's' || !PFX.includes(unit[0]) ? unit : unit.slice(1);
  let v = base === '%' ? x * 100 : x, p = 0, pf = '', e = base === '%' ? 2 : 0;
  if (unit === 'µs') { v *= 1e6; pf = 'µ'; e = 6; }   // per-kernel rows stay in µs, the trace's unit
  else if (base === 's') { if (Math.abs(v) < 1) { v *= 1000; pf = 'm'; e = 3; } }
  else if (base !== '%') { while (p < 6 && Math.abs(v) >= 1000) { v /= 1000; p++; } pf = PFX[p]; e = -3 * p; }
  return [v, pf, base, e];
}
// decimals in three significant figures (none for integers and from 1,000 up)
const sigDec = (v) => Number.isInteger(v) || Math.abs(v) >= 1000 ? 0 : Math.max(0, 2 - Math.floor(Math.log10(Math.abs(v))));
function fmtSI(x, unit) {
  const [v, pf, base] = siScale(x, unit);
  const t = Number.isInteger(v) ? String(v) : Math.abs(v) >= 1000 ? v.toLocaleString('en-US', { maximumSignificantDigits: 3 }) : v.toPrecision(3);
  const [ip, fp] = t.split('.');
  return `${ip}<span class="d">${fp ? '.' + fp : ''}</span><span class="u"> ${pf}${base}</span>`;
}

// ---- the sheet model --------------------------------------------------------
const rows = new Map();   // name (innerHTML) → { trs, label, leaf, f?, orig, pub }
for (const tr of document.querySelectorAll('.cellsheet tr')) {
  const [nm, lb, vl, sc, si, fx] = tr.cells;
  if (nm?.className !== 'nm') continue;
  const name = nm.innerHTML;
  tr._sci = !!sc.innerHTML;
  tr._unit = si.querySelector('.u')?.textContent.trim();
  if (rows.has(name)) { rows.get(name).trs.push(tr); continue; }
  const leaf = !fx.innerHTML.replace(/<i>[\s\S]*?<\/i>/g, '').trim();
  rows.set(name, { trs: [tr], label: lb.textContent, leaf, f: leaf ? null : compile(fx.innerHTML), lit: leaf ? dec(vl.textContent) : null });
}
const evalAll = (edits) => {
  const memo = new Map();
  const get = (name) => {
    if (memo.has(name)) return memo.get(name);
    const r = rows.get(name);
    if (!r) throw new Error(`sheetedit: no row named ${name}`);
    const v = r.leaf ? edits[KNOBS[name]] ?? r.lit : r.f(get);
    memo.set(name, v);
    return v;
  };
  for (const name of rows.keys()) get(name);
  return memo;
};
const PUB = evalAll({});
export { PUB as published, evalAll, set as setKnob };

// cross-row rules a knob edit must keep (else a leaf like M = 4 goes false)
function invalid(edits) {
  const v = (k) => edits[k] ?? PUB.get(Object.keys(KNOBS).find((n) => KNOBS[n] === k));
  for (const k of Object.keys(edits)) {
    if (edits[k][0] < 0n || (edits[k][0] === 0n && !PAGE.zero?.includes(k))) return `${k} must be positive`;
    if (COUNTS.has(k) && edits[k][1] !== 1n) return `${k} must be a whole number`;
  }
  return PAGE.rules(v, edits, PUB);
}

// ---- state ------------------------------------------------------------------
let edits = {};
try {
  const raw = JSON.parse(new URLSearchParams(location.hash.slice(1)).get(KEY) ?? '{}');
  for (const [k, s] of Object.entries(raw)) if (Object.values(KNOBS).includes(k)) edits[k] = dec(String(s));
  if (invalid(edits)) edits = {};
} catch { edits = {}; }
function save() {
  const p = new URLSearchParams(location.hash.slice(1)), o = {};
  for (const [k, q] of Object.entries(edits)) { const s = fmtKnob(q).replace(/,/g, ''); o[k] = Number.isSafeInteger(+s) ? +s : s; }
  if (Object.keys(o).length) p.set(KEY, JSON.stringify(o)); else p.delete(KEY);
  history.replaceState(null, '', p.size ? '#' + p : location.pathname + location.search);
}
export const current = () => evalAll(edits);
export const withKnobs = (over) => evalAll({ ...edits, ...over });   // the sheet at other knob values (page sweeps), on top of the user's edits
function render() {
  const now = current();
  for (const [name, r] of rows) {
    const v = now.get(name), off = !eq(v, PUB.get(name));
    for (const tr of r.trs) {
      const [, , vl, sc, si] = tr.cells;
      if (vl.querySelector('input')) continue;
      vl.textContent = KNOBS[name] ? fmtKnob(v) : fmtExact(v);
      if (tr._sci) sc.innerHTML = fmtSci(num(v));
      if (tr._unit) si.innerHTML = fmtSI(num(v), tr._unit);
      tr.classList.toggle('off', off);
    }
  }
  // prose numbers quoting a cell (<span data-cellref="name">published SI</span>)
  for (const q of document.querySelectorAll('[data-cellref]')) {
    const name = q.dataset.cellref, v = now.get(name), off = !eq(v, PUB.get(name));
    q.innerHTML = fmtSI(num(v), rows.get(name).trs[0]._unit);
    q.classList.toggle('off', off);
  }
  document.dispatchEvent(new CustomEvent('dsv3-cells', { detail: now }));
}
function set(k, q) {
  if (q && !eq(q, PUB.get(Object.keys(KNOBS).find((n) => KNOBS[n] === k)))) edits[k] = q; else delete edits[k];
  save(); render();
}

// ---- editing: click an editable value, type, Enter (Esc cancels) ------------
// typed numbers accept commas, e-notation, an SI prefix and the row's unit,
// loosely spelled (989T, 800 TFLOP/s, 800 tflops, 50 GB/s, 50GBps; Gb/s is bits)
const UNITS = { 'FLOP/s': /^flop(?:s|\/s(?:ec)?)?$/i, 'B/s': /^(?:B|bytes?)(?:\/s(?:ec)?|ps)?$/ };
const parse = (s, unit) => {
  const m = /^\s*(\d[\d,]*(?:\.\d*)?|\.\d+)(?:e([+-]?\d+))?\s*([kmgtpe]?)\s*([a-z/]*)\s*$/i.exec(s);
  if (!m) return 'not a number';
  if (m[4] && !UNITS[unit]?.test(m[4])) return unit ? `unit should be ${unit}` : 'just a number here, no unit';
  let q = dec(m[1].replace(/\.$/, ''));
  const e = +(m[2] ?? 0) + 3 * PFX.indexOf(m[3].toUpperCase());
  return OPS['·'](q, e >= 0 ? Q(10n ** BigInt(e)) : Q(1n, 10n ** BigInt(-e)));
};
const err = Object.assign(document.createElement('div'), { className: 'cs-err' });
document.body.append(err);
function open(td, name) {
  if (td.querySelector('input')) return;
  const k = KNOBS[name], inp = document.createElement('input'), u = rows.get(name).trs[0]._unit;
  inp.className = 'cs-in'; inp.value = td.textContent; inp.spellcheck = false;
  td.replaceChildren(inp); inp.focus(); inp.select();
  let done = false;
  const close = () => { done = true; err.style.display = 'none'; td.replaceChildren(); render(); };
  const commit = (soft) => {
    const q = parse(inp.value, u?.slice(1));   // TFLOP/s → FLOP/s, GB/s → B/s
    const why = typeof q === 'string' ? q : invalid({ ...edits, [k]: q });
    if (!why) { close(); set(k, q); if (!soft) td.focus(); return; }
    if (soft) { close(); return; }   // blur with a bad value: drop it
    inp.classList.add('bad');
    const r = td.getBoundingClientRect();
    err.textContent = why; err.style.display = 'block';
    err.style.left = r.left + scrollX + 'px'; err.style.top = r.bottom + scrollY + 3 + 'px';
  };
  inp.addEventListener('keydown', (ev) => {
    if (ev.key === 'Enter') commit(false);
    else if (ev.key === 'Escape') { ev.stopPropagation(); close(); td.focus(); }
    else { inp.classList.remove('bad'); err.style.display = 'none'; }
  });
  inp.addEventListener('blur', () => { if (!done) commit(true); });
}
for (const [name, r] of rows) {
  if (!KNOBS[name]) continue;
  for (const tr of r.trs) {
    const td = tr.cells[2];
    td.classList.add('edv'); td.tabIndex = 0; td.title = 'click to edit';
    td.addEventListener('click', (ev) => { ev.stopPropagation(); open(td, name); });
    td.addEventListener('keydown', (ev) => { if (ev.key === 'Enter' && ev.target === td) open(td, name); });
  }
}
render();

// ---- the .xlsx download: each row once (where it first appears), under its
// section's h2, with the page's three value columns (exact · scientific · SI)
// as number formats over live formulas; xfs collects the cell styles it uses
const plain = (h) => h.replace(/<sup>(.*?)<\/sup>/g, '^$1').replace(/<sub>(.*?)<\/sub>/g, '_$1').replace(/<[^>]+>/g, '').replace(/\s+/g, ' ').trim();
export function sheetXml(xfs = []) {
  const now = current(), at = new Map(), out = [];
  let sec = null;
  for (const el of document.querySelectorAll('h2, .cellsheet tr')) {
    if (el.tagName === 'H2') { sec = el.textContent; continue; }
    const name = el.cells[0]?.className === 'nm' && el.cells[0].innerHTML;
    if (!name || rows.get(name).trs[0] !== el) continue;
    if (sec) { out.push({ sec }); sec = null; }
    out.push({ name, tr: el });
    at.set(name, out.length + 1);
  }
  const sid = (x) => { const k = JSON.stringify(x), i = xfs.findIndex((y) => JSON.stringify(y) === k); return 6 + (i < 0 ? xfs.push(x) - 1 : i); };
  const str = (ref, t, x) => `<c r="${ref}" t="inlineStr" s="${sid(x)}"><is><t xml:space="preserve">${xesc(t)}</t></is></c>`;
  const xf = (fx) => fx.replace(/<i>[\s\S]*?<\/i>/g, '').replace(/<b[^>]*>([\s\S]*?)<\/b>/g, (_, n) => `C${at.get(n)}`)
    .replace(/10<sup>(\d+)<\/sup>/g, '1E+$1').replace(/(\d),(?=\d{3})/g, '$1')
    .replace(/·/g, '*').replace(/−/g, '-').replace(/⌈/g, 'CEILING(').replace(/⌉/g, ',1)').replace(/max\(/g, 'MAX(').replace(/\s+/g, '');
  const dec = (n) => n ? '.' + '0'.repeat(n) : '';
  const body = out.map((e, i) => {
    const r = i + 2;
    if (e.sec) return `<row r="${r}" ht="24" customHeight="1">${str('A' + r, e.sec, { font: 3 })}</row>`;
    const { name, tr } = e, fx = tr.cells[5].innerHTML, v = now.get(name), x = num(v), leaf = rows.get(name).leaf;
    const hl = tr.classList.contains('hl');   // the page's tinted bold rows
    const st = (o, font = 0) => (hl ? { ...o, font: [1, 1, 2, 3, 5, 5][font], fill: 2 } : font ? { ...o, font } : o);
    const vc = (col, f, fmt, font) => `<c r="${col}${r}" s="${sid(st({ fmt }, font))}">${f ? `<f>${xesc(f)}</f>` : ''}<v>${x}</v></c>`;
    const exact = leaf ? `<c r="C${r}" s="${sid(st({ fmt: Number.isInteger(x) ? '#,##0' : '#,##0.0########' }))}"><v>${fmtKnob(v).replace(/,/g, '')}</v></c>`
      : vc('C', xf(fx), Number.isInteger(x) ? '#,##0' : '#,##0.0########');
    const unit = tr._unit;
    let si = str('E' + r, '', st({}));
    if (unit) {
      const [sv, pf, base, p] = siScale(x, unit);
      si = vc('E', base === '%' || !p ? `C${r}` : `C${r}${p > 0 ? '*' : '/'}1E${Math.abs(p)}`,
        base === '%' ? `#,##0${dec(sigDec(sv))}%` : `#,##0${dec(sigDec(sv))}${pf + base ? `" ${pf}${base}"` : ''}`, 2);
    }
    const note = [...fx.matchAll(/<i>([\s\S]*?)<\/i>/g)].map((m) => plain(m[1])).join(' ');
    return `<row r="${r}">${str('A' + r, plain(name), st({}, 4))}${str('B' + r, tr.cells[1].textContent, st({}))}${exact}`
      + vc('D', `C${r}`, '0.00E+00', 2) + si
      + str('F' + r, leaf ? '' : plain(fx.replace(/<i>[\s\S]*?<\/i>/g, '')), st({})) + str('G' + r, note, st({}, 2)) + '</row>';
  });
  const head = `<row r="1">${['cell', 'quantity', 'value (exact)', 'scientific', 'SI', 'formula', 'note'].map((h, i) => str('ABCDEFG'[i] + '1', h, { font: 1, fill: 3, border: 1 })).join('')}</row>`;
  return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
    + '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
    + '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
    + '<cols>' + [10, 44, 28, 11, 16, 44, 60].map((w, i) => `<col min="${i + 1}" max="${i + 1}" width="${w}" customWidth="1"/>`).join('') + '</cols>'
    + `<sheetData>${head}${body.join('')}</sheetData></worksheet>`;
}
for (const b of document.querySelectorAll('.sheet-dl')) b.addEventListener('click', () => { const xfs = []; downloadXlsx(...PAGE.xlsx, sheetXml(xfs), xfs); });

describe(KEY, () => Object.entries(KNOBS).filter(([, k]) => edits[k]).map(([name, k]) => ({
  html: `<b>${name}</b> ${fmtKnob(PUB.get(name))} → ${fmtKnob(edits[k])}`,
  el: rows.get(name).trs[0],
  revert: () => set(k, null),
})));
