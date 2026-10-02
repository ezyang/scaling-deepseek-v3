// 03's static cell sheets, made live. The published values stay in the HTML
// (the page reads right before, and without, this module); the formula
// column IS the program: each formula's HTML compiles to exact rational
// arithmetic, so editing a real knob (B, S, GPUs, EP, NVL, n_c and the rates the model runs at: π^sol_bf16, π^sol_fp8, β_IB;
// the spec compute peaks are fixed)
// recomputes every dependent row, wherever it's repeated. Rows that drift
// from the published value turn amber. Edits live in the hash (departures
// only), so the floating reset (reset.js) lists and undoes them. Prose numbers
// that quote a cell (data-cellref = the row's name) follow along; each render
// fires 'dsv3-cells' so page scripts can follow too (current(), setKnob()).
import { describe } from './reset.js';

const KNOBS = { B: 'B', S: 'S', GPUs: 'GPUs', 'π<sup>sol</sup><sub>bf16</sub>': 'sol', 'π<sup>sol</sup><sub>fp8</sub>': 'sol8', EP: 'EP', NVL: 'NVL', 'β<sub>IB</sub>': 'IB', 'n<sub>c</sub>': 'SMc' };
const COUNTS = new Set(['B', 'S', 'GPUs', 'EP', 'NVL', 'SMc']);
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
// reference to the row named X; 10<sup>k</sup> is a literal
function compile(html) {
  const s = html.replace(/<i>[\s\S]*?<\/i>/g, '').replace(/10<sup>(\d+)<\/sup>/g, (_, e) => '1' + '0'.repeat(+e)).trim();
  const re = /\s*(?:<b>([\s\S]*?)<\/b>|(\d[\d,]*(?:\.\d+)?)|([·/+−()⌈⌉]))/y;
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
function fmtSI(x, unit) {
  const base = unit === 'ms' || unit === 'µs' ? 's' : unit === '%' || unit === 's' || !PFX.includes(unit[0]) ? unit : unit.slice(1);
  let v = base === '%' ? x * 100 : x, p = 0, pf = '';
  if (unit === 'µs') { v *= 1e6; pf = 'µ'; }   // per-kernel rows stay in µs, the trace's unit
  else if (base === 's') { if (Math.abs(v) < 1) { v *= 1000; pf = 'm'; } }
  else if (base !== '%') { while (p < 6 && Math.abs(v) >= 1000) { v /= 1000; p++; } pf = PFX[p]; }
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
  tr._pub = [vl.innerHTML, sc.innerHTML, si.innerHTML];
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
    if (edits[k][0] <= 0n) return `${k} must be positive`;
    if (COUNTS.has(k) && edits[k][1] !== 1n) return `${k} must be a whole number`;
  }
  const [G, E, N] = ['GPUs', 'EP', 'NVL'].map((k) => v(k)[0]);
  if (G % E) return `EP must divide GPUs (${G.toLocaleString('en-US')})`;
  if (E % N) return `NVL must divide EP (${E.toLocaleString('en-US')})`;
  if (E < 4n * N) return 'EP must span at least 4 nodes (EP ≥ 4 · NVL): M = 4 assumes a token can reach 4 nodes';
  const sms = PUB.get('n<sub>SM</sub>')?.[0];
  if (edits.SMc?.[0] > sms) return `an H800 has ${sms} SMs`;
  return null;
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
function render() {
  const now = current();
  for (const [name, r] of rows) {
    const v = now.get(name), off = !eq(v, PUB.get(name));
    for (const tr of r.trs) {
      const [, , vl, sc, si] = tr.cells;
      if (vl.querySelector('input')) continue;
      if (!off) [vl.innerHTML, sc.innerHTML, si.innerHTML] = tr._pub;
      else {
        vl.textContent = KNOBS[name] ? fmtKnob(v) : fmtExact(v);
        if (tr._pub[1]) sc.innerHTML = fmtSci(num(v));
        if (tr._unit) si.innerHTML = fmtSI(num(v), tr._unit);
      }
      tr.classList.toggle('off', off);
    }
  }
  // prose numbers quoting a cell (<span data-cellref="name">published SI</span>)
  for (const q of document.querySelectorAll('[data-cellref]')) {
    const name = q.dataset.cellref, v = now.get(name), off = !eq(v, PUB.get(name));
    q._pub ??= q.innerHTML;
    q.innerHTML = off ? fmtSI(num(v), rows.get(name).trs[0]._unit) : q._pub;
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

describe(KEY, () => Object.entries(KNOBS).filter(([, k]) => edits[k]).map(([name, k]) => ({
  html: `<b>${name}</b> ${fmtKnob(PUB.get(name))} → ${fmtKnob(edits[k])}`,
  el: rows.get(name).trs[0],
  revert: () => set(k, null),
})));
