// <dsv3-mesh>: DeepSeek-V3's 2,048 GPUs as 32 rows (EP groups) × 8 nodes ×
// 8 GPUs, and who our GPU exchanges parameters with under FSDP-without-PP
// (studies/03-roofline.html). Three mesh axes: EP (the MoE all-to-all, our
// row: NVLink to our node-mates, IB to our local rank on the group's 7
// other nodes, which forward over their NVLink), FSDP of the non-expert parameters
// (sharded over all 2,048: NVLink to our 7 node-mates, IB to the 255 GPUs
// with our local rank, 1/8 of the bytes each; every other GPU's shard
// arrives through one of those) and EFSDP of our expert slice (its 32
// copies are our column, one per EP group, all on other nodes: IB only).
import { C } from './theme.js';
import { knobCss } from './ui.js';

export const ROWS = 32, NODES = 8, GPUS = 8;       // EP groups, nodes per EP group, GPUs per node
export const OURS = { row: 12, node: 5, rank: 3 };  // our GPU: EP group 12, its node 5, local rank 3
const VIEWS = ['fsdp', 'efsdp', 'ep'];
// our GPU's relation to GPU (row, node, rank) in a view: ours · ib · nv · grp (holds a shard, reached
// through a peer) · out (not in the group)
export function role(view, row, node, rank) {
  const mine = node === OURS.node && rank === OURS.rank;
  if (mine && row === OURS.row) return 'ours';
  if (view === 'efsdp') return mine ? 'ib' : 'out';
  if (row === OURS.row && node === OURS.node) return 'nv';
  if (view === 'ep') return row !== OURS.row ? 'out' : rank === OURS.rank ? 'ib' : 'grp';
  return rank === OURS.rank ? 'ib' : 'grp';
}
// the hops between GPU at and our GPU in a view, as [from, to, link] with GPUs as [row, node, rank]:
// gathers flow to us, EP dispatch flows from us. A hierarchical all-gather crosses IB first (each GPU
// gathers from its local rank on every other node: an eighth of the blob), then NVLink swaps the eighths
export function route(view, row, node, rank) {
  const ro = role(view, row, node, rank), at = [row, node, rank], me = [OURS.row, OURS.node, OURS.rank];
  if (ro === 'ib' || ro === 'nv') return view === 'ep' ? [[me, at, ro]] : [[at, me, ro]];
  if (ro !== 'grp') return [];
  if (view === 'ep') { const fw = [row, node, OURS.rank]; return [[me, fw, 'ib'], [fw, at, 'nv']]; }
  const mate = [OURS.row, OURS.node, rank];
  return [[at, mate, 'ib'], [mate, me, 'nv']];
}
// dark = in the group, graded by how the bytes reach us: direct IB peer ·
// direct NVLink peer · one hop further (through a peer); white = not in it
const FILL = { ours: '#2a78d6', ib: '#1c1c1a', nv: '#52514e', grp: '#8f8d86', out: '#ffffff' };
const READOUT = {
  ep: 'our EP64 group (our row): dispatch + combine each MoE layer · NVLink: our 7 node-mates · IB: 7 same-rank peers, who forward to 49 more',
  fsdp: 'non-expert shards on all 2,048 GPUs · NVLink: our 7 node-mates · IB: the 255 GPUs with our local rank, 1/8 of the bytes (4.26 GB a move)',
  efsdp: 'our expert slice\'s 32 copies: one per EP group, on 32 nodes · NVLink: none, our node-mates hold other slices · IB: all of it (19.8 GB a move)',
};

// layout: a label gutter, then per row 8 node boxes of 8 GPU cells
const W = 738, LX = 100, PX = 9, CS = 8, NG = 7, PY = 14, TOP = 34;   // PY − box height = the gap between nodes
const NP = GPUS * PX + NG;                          // node pitch
const cx = (node, rank) => LX + node * NP + rank * PX, cy = (row) => TOP + row * PY;
const GX1 = cx(NODES - 1, GPUS - 1) + CS;           // grid's right edge
const H = cy(ROWS) + 44;
const hex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));
const mix = (a, b, t) => '#' + hex(a).map((v, i) => Math.round(v + (hex(b)[i] - v) * t).toString(16).padStart(2, '0')).join('');
const ease = (p) => 1 - (1 - p) ** 3;
// the route's hops as arrows from rim to rim, each with a background halo. IB: an amber curve. NVLink stays inside one node box, where the cells
// are a pitch apart: a green bracket that leaves the source's top (or bottom) edge, runs along the
// gap between rows and drops into the target — on the side the IB arc doesn't arrive from
const ctr = ([r, n, k]) => [cx(n, k) + CS / 2, cy(r) + CS / 2];
const f1 = (v) => v.toFixed(1);
const HALO = (d) => `<path d="${d}" fill="none" stroke="${C('#fcfcfb')}" stroke-width="3.5"/>`;
const head = (tx, ty, ux, uy, c) => `<path d="M${f1(tx)},${f1(ty)} L${f1(tx - ux * 4 - uy * 2.3)},${f1(ty - uy * 4 + ux * 2.3)} L${f1(tx - ux * 4 + uy * 2.3)},${f1(ty - uy * 4 - ux * 2.3)}Z" fill="${c}" stroke="${C('#fcfcfb')}" stroke-width="0.8"/>`;
function ibHop([x0, y0], [x1, y1]) {   // leaves and enters vertically, crossing only the gaps between rows
  const dx = x1 - x0, dy = y1 - y0, sg = Math.sign(dy) || 1, h = dy ? Math.abs(dy) * 0.45 : 6 + Math.min(40, Math.abs(dx) / 4);
  // down our column (EFSDP): leave and enter sideways instead, off the other copies
  const [ax, ay, bx, by] = !dx ? [x0 + 24, y0, x1 + 24, y1] : dy ? [x0, y0 + sg * h, x1, y1 - sg * h] : [x0, y0 - h, x1, y1 - h];
  const u = (x, y, tx, ty) => { const l = Math.hypot(tx - x, ty - y); return [(tx - x) / l, (ty - y) / l]; };
  const [sx, sy] = u(x0, y0, ax, ay), [ex, ey] = u(bx, by, x1, y1);
  const tx = x1 - ex * 4.5, ty = y1 - ey * 4.5, c = C('#eda100');   // both ends at the cell rim
  const line = `M${f1(x0 + sx * 4.5)},${f1(y0 + sy * 4.5)} C${f1(ax)},${f1(ay)} ${f1(bx)},${f1(by)} ${f1(tx - ex * 3)},${f1(ty - ey * 3)}`;
  return HALO(line) + `<path data-hop="ib" d="${line}" fill="none" stroke="${c}" stroke-width="1.6"/>` + head(tx, ty, ex, ey, c);
}
function nvHop(a, b, side) {   // side: -1 above the row, +1 below
  const x0 = ctr(a)[0], x1 = ctr(b)[0], edge = cy(a[0]) + (side > 0 ? CS : 0), yr = edge + side * 6.5, c = C('#1baf7a');
  const line = `M${f1(x0)},${f1(edge)} V${f1(yr)} H${f1(x1)} V${f1(edge + side * 3)}`;
  return HALO(line) + `<path data-hop="nv" d="${line}" fill="none" stroke="${c}" stroke-width="1.5" stroke-linejoin="miter"/>` + head(x1, edge, 0, -side, c);
}

const CSS = `
.msh { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-0b0b0b);
  border: 1px solid var(--c-e1e0d9); border-radius: 6px; background: var(--c-fcfcfb); padding: 8px 10px;
  width: ${W + 22}px; max-width: 100%; box-sizing: border-box; position: relative; }
.msh .top { display: flex; flex-wrap: wrap; align-items: stretch; gap: 8px 10px; padding-bottom: 6px; }
${knobCss('.msh .top')}
.msh svg { display: block; }
.msh text { font: 11px system-ui, -apple-system, "Segoe UI", sans-serif; fill: var(--c-52514e); }
.msh .dims { font-size: 9.5px; fill: var(--c-898781); }
.msh .hd { font-weight: 600; fill: var(--c-1c1c1a); }
`;

class Dsv3Mesh extends (typeof HTMLElement === 'undefined' ? class {} : HTMLElement) {
  connectedCallback() {
    // views="fsdp" / views="efsdp,ep": one mesh per instance (absent = all); the first is the default
    this.views = this.getAttribute('views')?.split(',').filter((v) => VIEWS.includes(v)) ?? VIEWS;
    const st = this.id ? readState('m:' + this.id) : null;
    this.view = this.views.includes(st?.v) ? st.v : this.views[0];
    const style = document.createElement('style'); style.textContent = CSS;
    this._root = el('div', 'msh');
    this._top = el('div', 'top');
    this._chart = el('div');
    this._root.append(this._top, this._chart);
    this.append(style, this._root);
    // the two device meshes, one knob group each (the buttons act as one radio):
    // the non-expert parameters' is 1-D over everyone, the experts' is 2-D —
    // EFSDP 32 (the row index) × EP 64 (along a row). An instance showing one view has no buttons, just the label
    for (let [lab, opts] of [
      ['non-expert mesh · FSDP 2,048', [['fsdp', 'FSDP 2,048 · everyone']]],
      ['expert mesh · EFSDP 32 × EP 64', [['efsdp', 'EFSDP 32 · down our column'], ['ep', 'EP 64 · along our row']]],
    ]) {
      opts = opts.filter(([v]) => this.views.includes(v));
      if (!opts.length) continue;
      const g = el('span', 'pargrp'), l = el('div', 'parlab'), r = el('div', 'parrow'), s = el('span', 'stp');
      l.textContent = lab; s.dataset.knob = 'view'; g.append(l);
      for (const [v, t] of opts) {
        const b = document.createElement('button'); b.type = 'button'; b.textContent = t; b.dataset.v = v; b.onclick = () => this.set(v); s.append(b);
      }
      if (this.views.length > 1) { r.append(s); g.append(r); }
      this._top.append(g);
    }
    // hovering a GPU draws its route to ours (no tip card: it would cover the route)
    this._chart.addEventListener('mousemove', (ev) => this._hover(ev));
    this._chart.addEventListener('mouseleave', () => this._hover(null));
    this._sync();
    this.render();
    addEventListener('dsv3-theme', () => this.render());
  }
  _sync() {
    for (const b of this._top.querySelectorAll('button')) b.classList.toggle('on', b.dataset.v === this.view);
    if (this.id) writeState('m:' + this.id, { v: this.view }, JSON.stringify({ v: this.views[0] }));
  }
  set(v) {
    if (v === this.view) return;
    const from = this.view;
    this.view = v;
    this._sync();
    const N = 12; let f = 0;   // ~200 ms: every cell's fill tweens old role → new
    const gen = this._gen = (this._gen ?? 0) + 1;
    const tick = () => {
      if (this._gen !== gen) return;
      f++; this._draw(from, ease(Math.min(1, f / N)));
      if (f < N) setTimeout(tick, 16);
    };
    setTimeout(tick, 16);
  }
  _routeSvg() {
    if (!this._hov) return '';
    const [row, n, k] = this._hov;
    const hops = route(this.view, row, n, k), ib = hops.find((h) => h[2] === 'ib');
    const side = ib && ib[0][0] > ib[1][0] ? -1 : 1;   // IB arriving from below: NVLink goes above
    return hops.map(([a, b, link]) => link === 'ib' ? ibHop(ctr(a), ctr(b)) : nvHop(a, b, side)).join('')
      + `<rect x="${cx(n, k) - 1}" y="${cy(row) - 1}" width="${CS + 2}" height="${CS + 2}" fill="none" stroke="${C('#eda100')}" stroke-width="1.2"/>`;
  }
  _route() { const g = this._chart.querySelector('[data-route]'); if (g) g.innerHTML = this._routeSvg(); }
  render() { this._gen = (this._gen ?? 0) + 1; this._draw(null, 1); }
  _draw(from, t) {
    const B = [], v = this.view, fill = (ro) => C(FILL[ro]);
    const ex = this.views.some((w) => w !== 'fsdp'), flat = this.views.includes('fsdp');
    B.push(`<text class="dims" x="0" y="11">one cell per GPU: ${(ROWS * NODES * GPUS).toLocaleString('en-US')} = ${ROWS} rows${ex ? ' (EFSDP)' : ''} × ${NODES * GPUS} per row (${ex ? 'EP: ' : ''}${NODES} nodes × ${GPUS} GPUs)${flat ? ' · the non-expert mesh flattens all of them into one FSDP axis' : ''}</text>`);
    for (let n = 0; n < NODES; n++) B.push(`<text class="dims" x="${cx(n, 0) + (GPUS * PX - 1) / 2}" y="${TOP - 7}" text-anchor="middle">node ${n}</text>`);
    for (let row = 0; row < ROWS; row++) {
      const y = cy(row), us = row === OURS.row;
      // EP group N: the row (the tip's name for it too); ours says so in the empty slot below
      if (row === 0 || us || row === ROWS - 1) B.push(`<text x="0" y="${y + CS}"${us ? ' class="hd"' : ''}>EP group ${row}</text>`);
      if (us) B.push(`<text class="dims" x="0" y="${y + PY + CS - 1}">(ours)</text>`);
      for (let n = 0; n < NODES; n++) {
        const ourNode = us && n === OURS.node;
        B.push(`<rect x="${cx(n, 0) - 2}" y="${y - 2}" width="${GPUS * PX + 3}" height="${CS + 4}" rx="2" fill="${C('#f3f2ee')}" stroke="${C(ourNode ? '#0b0b0b' : '#e1e0d9')}"${ourNode ? ' stroke-width="1.2" data-node="ours"' : ''}/>`);
        for (let k = 0; k < GPUS; k++) {
          const ro = role(v, row, n, k), f0 = from ? role(from, row, n, k) : ro;
          const c = f0 === ro || t >= 1 ? fill(ro) : mix(fill(f0), fill(ro), t);
          B.push(`<rect data-role="${ro}" x="${cx(n, k)}" y="${y}" width="${CS}" height="${CS}" fill="${c}"/>`);
        }
      }
    }
    // our GPU, outlined on top
    B.push(`<rect x="${cx(OURS.node, OURS.rank) - 1}" y="${cy(OURS.row) - 1}" width="${CS + 2}" height="${CS + 2}" fill="none" stroke="${C('#0b0b0b')}" stroke-width="1.2"/>`);
    B.push(`<g data-route>${this._routeSvg()}</g>`);
    B.push(`<text data-readout x="0" y="${H - 22}">${READOUT[v]}</text>`);
    const ly = H - 4, sw = (x, ro, label) => `<rect x="${x}" y="${ly - 8}" width="${CS}" height="${CS}" fill="${fill(ro)}"${ro === 'out' ? ` stroke="${C('#aba89f')}" stroke-width="0.6"` : ''}/><text class="dims" x="${x + 12}" y="${ly - 1}">${label}</text>`;
    B.push(sw(0, 'ours', 'our GPU'), sw(66, 'ib', 'IB peer'), sw(126, 'nv', 'NVLink peer (same node)'), sw(262, 'grp', 'in the group, reached through a peer'), ex ? sw(456, 'out', 'not in the group') : '');   // FSDP alone has no outsiders
    // the hover route's key, right-aligned
    const key = (x, c, lab) => `<path d="M${x},${ly - 4} h10" stroke="${C(c)}" stroke-width="1.6"/>${head(x + 14, ly - 4, 1, 0, C(c))}<text class="dims" x="${x + 17}" y="${ly - 1}">${lab}</text>`;
    B.push(`<g data-key><text class="dims" x="${W - 146}" y="${ly - 1}">route:</text>${key(W - 115, '#eda100', 'IB')}${key(W - 83, '#1baf7a', 'NVLink')}</g>`);
    this._chart.innerHTML = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="DeepSeek-V3's 2,048 GPUs drawn as 32 rows, one per EP group, each of 8 nodes of 8 GPUs. Under EP our row trades tokens: NVLink to our 7 node-mates, InfiniBand to our local rank on its 7 other nodes. Under FSDP the non-expert parameters are sharded over all of them: our GPU gathers from its 7 node-mates over NVLink and from the 255 GPUs with the same local rank over InfiniBand. Under EFSDP our expert slice's 32 copies form one column, one GPU per EP group, all on other nodes, so every byte crosses InfiniBand.">${B.join('')}</svg>`;
  }
  // the GPU under the pointer, from its position (no dead zones between cells)
  _hover(ev) {
    let hov = null;
    if (ev) {
      const svg = this._chart.querySelector('svg'), R = svg.getBoundingClientRect(), s = W / R.width;
      const x = (ev.clientX - R.left) * s, y = (ev.clientY - R.top) * s;
      const row = Math.floor((y - TOP + (PY - CS) / 2) / PY), n = Math.floor((x - LX + NG / 2) / NP), k = Math.floor((x - cx(n, 0) + (PX - CS) / 2) / PX);
      if (row >= 0 && row < ROWS && n >= 0 && n < NODES && k >= 0 && k < GPUS && x <= GX1 + 2) hov = [row, n, k];
    }
    if (String(hov) !== String(this._hov)) { this._hov = hov; this._route(); }
  }
}
function el(tag, cls) { const e = document.createElement(tag); if (cls) e.className = cls; return e; }
function readState(key) {
  try { const v = new URLSearchParams(location.hash.slice(1)).get(key); return v ? JSON.parse(v) : null; } catch { return null; }
}
function writeState(key, obj, def) {
  const p = new URLSearchParams(location.hash.slice(1)), s = JSON.stringify(obj);
  if (s === def) p.delete(key); else p.set(key, s);   // the hash carries only departures from the defaults
  history.replaceState(null, '', p.size ? '#' + p : location.pathname + location.search);
}
if (typeof customElements !== 'undefined' && !customElements.get('dsv3-mesh')) customElements.define('dsv3-mesh', Dsv3Mesh);
