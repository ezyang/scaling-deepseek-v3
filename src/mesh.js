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
import { attachTip } from './tip.js';

export const ROWS = 32, NODES = 8, GPUS = 8;       // EP groups, nodes per EP group, GPUs per node
export const OURS = { row: 12, node: 5, rank: 3 };  // our GPU: EP group 12, its node 5, local rank 3
const VIEWS = ['ep', 'fsdp', 'efsdp'];
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
    const st = this.id ? readState('m:' + this.id) : null;
    this.view = VIEWS.includes(st?.v) ? st.v : 'fsdp';
    const style = document.createElement('style'); style.textContent = CSS;
    this._root = el('div', 'msh');
    this._top = el('div', 'top');
    this._chart = el('div');
    this._root.append(this._top, this._chart);
    this.append(style, this._root);
    // the two device meshes, one knob group each (the buttons act as one radio):
    // the non-expert parameters' is 1-D over everyone, the experts' is 2-D —
    // EFSDP 32 (the row index) × EP 64 (along a row)
    for (const [lab, opts] of [
      ['non-expert mesh · FSDP 2,048', [['fsdp', 'FSDP 2,048 · everyone']]],
      ['expert mesh · EFSDP 32 × EP 64', [['ep', 'EP 64 · along our row'], ['efsdp', 'EFSDP 32 · down our column']]],
    ]) {
      const g = el('span', 'pargrp'), l = el('div', 'parlab'), r = el('div', 'parrow'), s = el('span', 'stp');
      l.textContent = lab; s.dataset.knob = 'view';
      for (const [v, t] of opts) {
        const b = document.createElement('button'); b.type = 'button'; b.textContent = t; b.dataset.v = v; b.onclick = () => this.set(v); s.append(b);
      }
      r.append(s); g.append(l, r); this._top.append(g);
    }
    attachTip(this._chart, (ev) => this._tip(ev), { parent: this._root });
    this._sync();
    this.render();
    addEventListener('dsv3-theme', () => this.render());
  }
  _sync() {
    for (const b of this._top.querySelectorAll('button')) b.classList.toggle('on', b.dataset.v === this.view);
    if (this.id) writeState('m:' + this.id, { v: this.view });
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
  render() { this._gen = (this._gen ?? 0) + 1; this._draw(null, 1); }
  _draw(from, t) {
    const B = [], v = this.view, fill = (ro) => C(FILL[ro]);
    B.push(`<text class="dims" x="0" y="11">one cell per GPU: ${(ROWS * NODES * GPUS).toLocaleString('en-US')} = ${ROWS} rows (EFSDP) × ${NODES * GPUS} per row (EP: ${NODES} nodes × ${GPUS} GPUs) · the non-expert mesh flattens all of them into one FSDP axis</text>`);
    for (let n = 0; n < NODES; n++) B.push(`<text class="dims" x="${cx(n, 0) + (GPUS * PX - 1) / 2}" y="${TOP - 7}" text-anchor="middle">node ${n}</text>`);
    for (let row = 0; row < ROWS; row++) {
      const y = cy(row), us = row === OURS.row;
      if (row === 0 || us || row === ROWS - 1) B.push(`<text x="0" y="${y + CS}"${us ? ' class="hd"' : ''}>group ${row}${us ? ' (ours)' : ''}</text>`);
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
    B.push(`<text data-readout x="0" y="${H - 22}">${READOUT[v]}</text>`);
    const ly = H - 4, sw = (x, ro, label) => `<rect x="${x}" y="${ly - 8}" width="${CS}" height="${CS}" fill="${fill(ro)}"${ro === 'out' ? ` stroke="${C('#aba89f')}" stroke-width="0.6"` : ''}/><text class="dims" x="${x + 12}" y="${ly - 1}">${label}</text>`;
    B.push(sw(0, 'ours', 'our GPU'), sw(66, 'ib', 'IB peer'), sw(126, 'nv', 'NVLink peer (same node)'), sw(262, 'grp', 'in the group, reached through a peer'), sw(456, 'out', 'not in the group'));
    this._chart.innerHTML = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="DeepSeek-V3's 2,048 GPUs drawn as 32 rows, one per EP group, each of 8 nodes of 8 GPUs. Under EP our row trades tokens: NVLink to our 7 node-mates, InfiniBand to our local rank on its 7 other nodes. Under FSDP the non-expert parameters are sharded over all of them: our GPU gathers from its 7 node-mates over NVLink and from the 255 GPUs with the same local rank over InfiniBand. Under EFSDP our expert slice's 32 copies form one column, one GPU per EP group, all on other nodes, so every byte crosses InfiniBand.">${B.join('')}</svg>`;
  }
  // the GPU under the pointer, from its position (no dead zones between cells)
  _tip(ev) {
    const svg = this._chart.querySelector('svg'), R = svg.getBoundingClientRect(), s = W / R.width;
    const x = (ev.clientX - R.left) * s, y = (ev.clientY - R.top) * s;
    const row = Math.floor((y - TOP + (PY - CS) / 2) / PY), n = Math.floor((x - LX + NG / 2) / NP), k = Math.floor((x - cx(n, 0) + (PX - CS) / 2) / PX);
    if (row < 0 || row >= ROWS || n < 0 || n >= NODES || k < 0 || k >= GPUS || x > GX1 + 2) return null;
    const node = row * NODES + n, epr = n * GPUS + k, ro = role(this.view, row, n, k);
    const slice = epr === OURS.node * GPUS + OURS.rank ? 'our expert slice' : `EP rank ${epr}'s expert slice`;
    const what = {
      ours: 'our GPU',
      ib: { ep: 'same local rank, same EP group: an IB peer, forwarding tokens to and from its node-mates',
        fsdp: 'same local rank: an IB peer for 1/8 of the non-expert bytes', efsdp: 'holds a copy of our expert slice: an IB peer' }[this.view],
      nv: 'our node-mate: an NVLink peer',
      grp: this.view === 'ep' ? `holds EP rank ${epr}'s experts: our tokens reach it through an IB peer on its node` : 'holds a non-expert shard, which reaches us through a peer',
      out: this.view === 'ep' ? 'another EP group: no tokens exchanged' : `holds ${slice}: nothing to exchange with us`,
    }[ro];
    return `GPU ${(node * GPUS + k).toLocaleString('en-US')} · node ${node}, local rank ${k}\nEP group ${row}, EP rank ${epr}\n${what}`;
  }
}
function el(tag, cls) { const e = document.createElement(tag); if (cls) e.className = cls; return e; }
function readState(key) {
  try { const v = new URLSearchParams(location.hash.slice(1)).get(key); return v ? JSON.parse(v) : null; } catch { return null; }
}
function writeState(key, obj) {
  const p = new URLSearchParams(location.hash.slice(1));
  p.set(key, JSON.stringify(obj));
  history.replaceState(null, '', '#' + p.toString());
}
if (typeof customElements !== 'undefined' && !customElements.get('dsv3-mesh')) customElements.define('dsv3-mesh', Dsv3Mesh);
