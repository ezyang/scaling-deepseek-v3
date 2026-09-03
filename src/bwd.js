// <dsv3-bwd>: one DSv3 transformer block with its BACKWARD unrolled —
// three columns, forward | backward | weight-grad. One backward node per
// forward node (plain chain rule, nothing fused), flow reversed, and the
// save-for-backward edges crossing from the forward tensors to the backward
// ops that read them. The purpose is PRECISION, not memory: every op wears
// the dtype it runs in, every tensor the dtype it is carried in. Recompute
// is not modeled (the forward column IS the recompute side, when you replay
// it instead of following a save edge).
//
// DRAFT (scratch study). The per-op backward precisions are our reading of
// the DeepSeek-V3 report §3.3 (Fig. 6: Fprop/Dgrad/Wgrad all E4M3 GEMMs with
// fp32-promoted accumulation; input grads bf16; weight grads fp32; the
// embedding, head, gating, norms and attention kept high precision; the
// backward combine — an activation-gradient DISPATCH — in fp8, the backward
// dispatch — a COMBINE — in bf16; E5M6 attention-output cache; fp8 SwiGLU
// inputs) plus production practice where the paper is silent (marked ~).
import { C } from './theme.js';
import { DSV3 } from './model.js';

const DT = { bf16: '#52514e', e4m3: '#d6408b', e5m6: '#7b2fa8', fp32: '#8a3324' };
const A = DSV3, H = A.hidden, QK = A.qkNope + A.qkRope, I = A.moeInter;

// ---- the rows: one forward op each (a `pair` row is the MLA q | kv fork) ------
// F: forward op {label, dims, dt} · out: its output tensor {name, dt, saved?}
// B: backward op {label, dt} · dout: gradient of F's output {dt} (name derived)
// W: weight-grad op {dt, outDt} reading X = the previous row's out and dY = dout
// saved: which backward consumers read this tensor: 'B:<rowId>' or 'W:<rowId>'
const mm = (label, dims) => ({ label, dims, dt: 'e4m3', kind: 'matmul', emit: 'bf16' });   // GEMMs state their output dtype too: fp32 accumulate, bf16 out
const dgrad = { op: 'dgrad', dims: null, dt: 'e4m3', emit: 'bf16' };   // dims: the forward shape reversed (filled at render)
const wgrad = { op: 'wgrad', dims: 'dW = Xᵀ·dY', dt: 'e4m3', outDt: 'fp32', emit: 'fp32' };   // the caption is rebuilt from the named parameters at render
const dgrad2 = (dt, emit) => ({ op: 'dgrad', dims: null, dt, emit });
const norm = (label, n) => ({ label, dims: `${n} · rstd fp32`, dt: 'bf16', kind: 'vector' });
const normB = { op: 'bwd', dims: null, dt: 'bf16' };   // dims null: the forward caption (what it reads is on the amber edges)
const gammaW = { op: 'dγ', dims: 'dγ · reduce over tokens ~', dt: 'fp32', outDt: 'fp32' };

export const ROWS = [
  { id: 'x0', chip: { name: 'x0 — block input', dt: 'bf16', saved: ['B:norm1'] }, dchip: { name: 'd(x0) → previous block', dt: 'bf16' } },
  // QUANTIZATION PLACEMENT: a cast is a kernel. It rides the nearest vector (grey) kernel when there is
  // one — the box wears a second '⇒ e4m3' tag and the chip it emits is e4m3 — and gets its own box when
  // there is nothing to fuse it into (after attention; after a junction in backward). A wgrad's stash
  // operand is re-tiled 128×1 by a small requant box on its blue edge. (+ᵀ) on a gradient chip = both
  // orientations minted, rows for the dgrad and columns for the wgrad.
  { id: 'norm1', param: ['γ'], F: { ...norm('RMSNorm (pre-attn)', H), emit: 'e4m3' }, out: { name: 'norm1 out', dt: 'e4m3', saved: ['W:qkv_down'], aux: 'rstd fp32 → norm bwd' },
    B: normB, dout: { dt: 'bf16' }, W: gammaW },
  { id: 'qkv_down', param: ['W^DQ', 'W^DKV', 'W^KR'], F: mm('q/kv down-proj', `${H} → ${A.qRank} + ${A.kvRank + A.qkRope}`), out: { name: 'latents (q, kv, k_rope)', dt: 'bf16', saved: ['B:q_norm', 'B:kv_norm'] },
    B: dgrad, dout: { dt: 'e4m3', name: 'd(latents) (+ᵀ)' }, W: wgrad },
  { id: 'lat_norm', pair: [
    { id: 'q_norm', param: ['γ'], F: { ...norm('RMSNorm (q)', A.qRank), emit: 'e4m3' }, out: { name: 'norm(q latent)', dt: 'e4m3', saved: ['W:q_up'] }, B: { ...normB, emit: 'e4m3' }, dout: { dt: 'bf16' }, W: gammaW },
    { id: 'kv_norm', param: ['γ'], F: { ...norm('RMSNorm (kv)', A.kvRank), emit: 'e4m3' }, out: { name: 'norm(kv latent)', dt: 'e4m3', saved: ['W:kv_up'] }, B: { ...normB, emit: 'e4m3' }, dout: { dt: 'bf16' }, W: gammaW },
  ] },
  { id: 'up', pair: [
    { id: 'q_up', param: ['W^UQ', 'W^QR'], F: mm('q up-proj', `${A.qRank} → ${A.heads}×${QK}`), out: { name: 'q (pre-RoPE)', dt: 'bf16' }, B: dgrad, dout: { dt: 'e4m3', name: 'd(q) (+ᵀ)' }, W: wgrad },
    { id: 'kv_up', param: ['W^UK', 'W^UV'], F: mm('kv up-proj', `${A.kvRank} → ${A.heads}×(${A.qkNope}+${A.vHead})`), out: { name: 'k,v (pre-RoPE)', dt: 'bf16' }, B: dgrad, dout: { dt: 'e4m3', name: 'd(k,v) (+ᵀ)' }, W: wgrad },
  ] },
  { id: 'rope', pair: [
    { id: 'rope_q', F: { label: 'RoPE (q)', dims: 'rotation', dt: 'bf16', kind: 'vector' }, out: { name: 'q', dt: 'bf16', saved: ['B:attn'] },
      B: { op: 'bwd', dims: null, dt: 'bf16', emit: 'e4m3' }, dout: { dt: 'bf16' } },
    { id: 'rope_kv', F: { label: 'RoPE (k,v)', dims: 'rotation · concat k_rope', dt: 'bf16', kind: 'vector' }, out: { name: 'k, v', dt: 'bf16', saved: ['B:attn'] },
      B: { op: 'bwd', dims: null, dt: 'bf16', emit: 'e4m3' }, dout: { dt: 'bf16' } },
  ] },
  // the paper: the linear's input is cached E5M6 (1×128 tiles, power-of-two scales, re-tiled 128×1 in backward),
  // read by the attention backward and the out-proj wgrad. Drawn as attention writing the E5M6 in its epilogue
  // (the cast rides the producer, as everywhere) and the GEMM's E4M3 operand DERIVED from that cache — the
  // fake-quant chain; the paper does not say whether the operand comes from the cache or from bf16 (~)
  { id: 'attn', F: { label: 'attention', dims: 'softmax(QKᵀ)V · causal · lse fp32', dt: 'bf16', kind: 'attn', emit: 'e5m6' },
    out: { name: 'attn out', dt: 'e5m6', saved: ['B:attn', 'W:o_proj'], aux: 'lse fp32 saved → attn bwd' },
    B: { op: 'bwd', dims: null, dt: 'bf16' }, dout: { dt: 'bf16' } },
  { id: 'q_attn', F: { label: 'requant (attn out)', dims: 'e5m6 → e4m3, 1×128 · the GEMM operand derived from the cache ~', dt: 'e4m3', kind: 'vector' },
    out: { name: 'attn out (GEMM operand)', dt: 'e4m3' },
    B: null, dout: { dt: 'bf16', name: 'd(attn out)' } },
  { id: 'o_proj', param: ['W^O'], F: mm('attn out-proj', `${A.heads}×${A.vHead} → ${H}`), out: { name: 'attn proj out', dt: 'bf16' },
    B: dgrad, dout: { dt: 'e4m3', name: 'd(attn proj out) (+ᵀ)' }, W: wgrad },
  // the gradient arrives from a junction: nothing to fuse the cast into → its own box
  { id: 'q_oproj', F: null, B: { label: 'quantize (d attn proj out)', dims: 'rows for the dgrad, columns for the wgrad', dt: 'e4m3', kind: 'vector', emit: 'e4m3' },
    dout: { dt: 'bf16', name: 'd(attn proj out)' } },
  { id: 'x1', F: { label: '+ residual (x0)', dims: '', dt: 'bf16', kind: 'add' }, out: { name: 'x1', dt: 'bf16', saved: ['B:norm2'] },
    B: null, dout: { dt: 'bf16' } },   // an add's backward is a junction (the gradient fans out unchanged): no box
  { id: 'norm2', param: ['γ'], F: { ...norm('RMSNorm (pre-FFN)', H), emit: 'e4m3' }, out: { name: 'norm2 out (+ e4m3 copy)', dt: 'bf16', saved: ['W:router', 'W:sh_gate_up'], aux: 'rstd fp32 → norm bwd' },
    B: normB, dout: { dt: 'bf16' }, W: gammaW },
  // the router is TWO ops: the fp32 GEMM (whose wgrad needs d(logits)) and the gating
  // (sigmoid · biased top-k · renorm), whose backward is what d(top-k weights) enters
  { id: 'router', side: true, param: ['e (centroids)'], F: { label: 'router GEMM', dims: `${H} → ${A.routedExperts}`, dt: 'fp32', kind: 'matmul' },
    out: { name: 'logits', dt: 'fp32' },
    B: dgrad2('fp32', 'bf16'), dout: { dt: 'fp32' },   // its input-grad joins d(norm2 out) in bf16
    W: { op: 'wgrad', dims: 'de · bias b: balancing rule, no gradient', dt: 'fp32', outDt: 'fp32' } },
  { id: 'gate', side: true, F: { label: 'gating', dims: `σ, +b, top-${A.topk}, renorm`, dt: 'fp32', kind: 'vector' },
    out: { name: 'top-k weights · indices', dt: 'fp32', saved: ['B:gate', 'B:swiglu', 'B:combine', 'B:dispatch'] },
    B: { op: 'bwd', dims: null, dt: 'fp32' }, dout: { dt: 'fp32', name: 'd(top-k weights)' } },
  { id: 'dispatch', F: { label: 'a2a dispatch · permute + comm', dims: 'payload e4m3 (minted by the norm, power-of-two scales)', dt: 'e4m3', kind: 'comm' },
    out: { name: 'dispatched tokens', dt: 'e4m3', saved: ['W:gate_up'] },
    B: { op: 'bwd (= a combine)', dims: 'gradient payload bf16 (§3.3.3)', dt: 'bf16', kind: 'comm' }, dout: { dt: 'bf16' } },
  { id: 'gate_up', param: ['W_1 (gate)', 'W_3 (up)'], F: mm('ffn gate/up (grouped)', `${H} → 2×${I} per expert`), out: { name: 'gate, up (GEMM out)', dt: 'bf16' },
    B: dgrad, dout: { dt: 'e4m3', name: 'd(gate, up) (+ᵀ)' }, W: wgrad },
  { id: 'quant', F: { label: 'quantize (for the stash)', dims: 'fused into the SwiGLU kernel in production', dt: 'e4m3', kind: 'vector' },
    out: { name: 'gate, up', dt: 'e4m3', saved: ['B:swiglu'] },
    B: null, dout: { dt: 'e4m3', name: 'd(gate, up) (+ᵀ)' } },   // identity backward: no box, the gradient passes straight through
  { id: 'swiglu', F: { label: 'SwiGLU · × top-k weight', dims: 'computes in bf16', dt: 'bf16', kind: 'vector', emit: 'e4m3' },
    out: { name: 'swiglu out', dt: 'e4m3', saved: ['W:ffn_down'] },
    B: { op: 'bwd', dims: 'also emits d(top-k weights) fp32', dt: 'bf16', emit: 'e4m3' }, dout: { dt: 'bf16' } },
  { id: 'ffn_down', param: ['W_2 (down)'], F: mm('ffn down (grouped)', `${I} → ${H} per expert`), out: { name: 'expert outputs', dt: 'bf16' },
    B: dgrad, dout: { dt: 'e4m3' }, W: wgrad },
  { id: 'combine', F: { label: 'a2a combine · comm + unpermute · sum', dims: 'payload bf16', dt: 'bf16', kind: 'comm' },
    out: { name: 'moe out (routed)', dt: 'bf16' },
    B: { op: 'bwd (= a dispatch)', dims: 'activation-gradient payload e4m3 (§3.3.3)', dt: 'e4m3', kind: 'comm' }, dout: { dt: 'e4m3', name: 'd(moe out) (routed)' } },
  // the routed gradient arrives from a junction: its cast for the backward combine gets its own box
  { id: 'q_moe', fSkip: true, F: null, B: { label: 'quantize (d moe out)', dims: 'the backward combine is a dispatch: payload e4m3', dt: 'e4m3', kind: 'vector', emit: 'e4m3' },
    dout: { dt: 'bf16', name: 'd(moe out) (routed)' } },
  // the shared expert, pulled BELOW the routed chain (no side nesting — its
  // input is norm2 out, via the left rail; its output joins the routed sum)
  { id: 'sh_gate_up', rail: 'norm2', param: ['W_1^sh', 'W_3^sh'], F: mm('shared gate/up', `${H} → 2×${I}`), out: { name: 'gate, up (sh, GEMM out)', dt: 'bf16' },
    B: dgrad, dout: { dt: 'e4m3', name: 'd(gate, up) (sh) (+ᵀ)' }, W: wgrad },
  { id: 'sh_quant', F: { label: 'quantize (sh)', dims: 'fused into the SwiGLU kernel', dt: 'e4m3', kind: 'vector' },
    out: { name: 'gate, up (sh)', dt: 'e4m3', saved: ['B:sh_swiglu'] },
    B: null, dout: { dt: 'e4m3', name: 'd(gate, up) (sh) (+ᵀ)' } },
  { id: 'sh_swiglu', F: { label: 'SwiGLU (sh, ungated)', dims: 'computes in bf16', dt: 'bf16', kind: 'vector', emit: 'e4m3' },
    out: { name: 'swiglu out (sh)', dt: 'e4m3', saved: ['W:sh_down'] },
    B: { op: 'bwd', dims: '', dt: 'bf16', emit: 'e4m3' }, dout: { dt: 'bf16' } },
  { id: 'sh_down', param: ['W_2^sh'], F: mm('shared down', `${I} → ${H}`), out: { name: 'shared out', dt: 'bf16' },
    B: dgrad, dout: { dt: 'e4m3', name: 'd(shared out) (+ᵀ)' }, W: wgrad },
  { id: 'sh_qdown', F: null, B: { label: 'quantize (d shared out)', dims: 'rows for the dgrad, columns for the wgrad', dt: 'e4m3', kind: 'vector', emit: 'e4m3' },
    dout: { dt: 'bf16', name: 'd(shared out)' } },
  { id: 'moe_add', F: { label: '+ routed + shared', dims: '', dt: 'bf16', kind: 'add' }, out: { name: 'ffn out', dt: 'bf16' },
    B: null, dout: { dt: 'bf16' } },
  { id: 'x2', F: { label: '+ residual (x1)', dims: '', dt: 'bf16', kind: 'add' }, out: { name: 'x2 → next block', dt: 'bf16' },
    B: null, dout: { dt: 'bf16', name: 'd(x2) ← next block' } },
];

// ---- geometry -------------------------------------------------------------------
const BW = 412, SUBW = 200, SUBGAP = 12, SPX = 18;   // BW = 2·SUBW + SUBGAP: the fork spans the box   // box width · fork box width/gap · spine offset inside a box
const GUT = 130;                                     // column gutter (rails + save-edge channels live here)
const COLS = { F: 76, B: 76 + BW + GUT, W: 76 + 2 * (BW + GUT) };
const H_BOX = 40, H_CHIP = 40, H_CHIP2 = 60;         // box band · chip band (single / two-line — the fork rows carry four lanes)
const RAIL = { x0: -46, x1: -46, norm2: -26 };       // left rails, relative to the column x
const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;');
const dtSpan = (dt) => `<tspan fill="${C(DT[dt])}" font-weight="600">${dt}</tspan>`;
const textW = (s) => 5.3 * s.length;                 // 10px system-ui, rough

const CSS = `
.bw { display: block; font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-0b0b0b);
  border: 1px solid var(--c-e1e0d9); border-radius: 6px; background: var(--c-fcfcfb); padding: 10px 12px; }
.bw-scroll { overflow-x: auto; }
.bw-head { color: var(--c-52514e); padding-bottom: 6px; display: flex; gap: 18px; }
.bw-head input { vertical-align: -2px; }
.bw svg { display: block; }
.bw .box { fill: var(--c-ffffff); stroke: var(--c-c3c2b7); }
.bw .box.vector { fill: var(--c-f3f2ee); stroke: var(--c-d8d6cb); }
.bw .box.comm { fill: var(--c-f3f1fb); stroke: var(--c-6b5bd2); }
.bw .box.add { fill: none; stroke: var(--c-c3c2b7); stroke-dasharray: 3 2; }
.bw .bop .box { stroke-dasharray: 4 2; }
.bw .sup { font-size: 7.5px; baseline-shift: super; }
.bw .sub { font-size: 7.5px; baseline-shift: sub; }
.bw .rqt { font: italic 8px system-ui; fill: var(--c-52514e); }
.bw .plus { font: 600 11px system-ui; fill: var(--c-52514e); }
.bw .name { font: 600 11px system-ui; fill: var(--c-0b0b0b); }
.bw .dims { font: 9.5px system-ui; fill: var(--c-898781); }
.bw .dtagt { font: 600 10.5px system-ui; }
.bw text.tensor { font: 10px system-ui; }
.bw .tsave { fill: var(--c-7a5200); font-weight: 600; }
.bw .tidle { fill: var(--c-a8a69e); }
.bw .tgrad { fill: var(--c-b05f00); }
.bw .tdim { fill: var(--c-898781); font-weight: 400; }
.bw .wire { fill: none; stroke: var(--c-898781); stroke-width: 1.1; }
.bw .save { fill: none; stroke: var(--c-eda100); stroke-width: 1; }
.bw .elide { fill: none; stroke: var(--c-898781); stroke-width: 1.1; stroke-dasharray: 3 3; }
.bw .savew { fill: none; stroke: var(--c-2a78d6); stroke-width: 1; stroke-opacity: 0.45; }
.bw .grad { fill: none; stroke: var(--c-eb6834); stroke-width: 1; }
.bw .hit { fill: none; stroke: transparent; stroke-width: 9; pointer-events: stroke; }
.bw .hl, .bw .pin { stroke-width: 1.7; stroke-opacity: 1; }   /* a touch heavier, not so heavy the fork dots drown */
.bw .wire.hl, .bw .wire.pin { stroke: var(--c-0b0b0b); }
.bw circle.hl, .bw circle.pin { r: 3.5; fill: var(--c-0b0b0b); }
.bw rect.box.hl, .bw rect.box.pin { stroke: var(--c-0b0b0b); stroke-width: 1.5; }
.bw .hit { cursor: pointer; }
.bw .raillab { font: italic 9px system-ui; fill: var(--c-898781); }
.bw .colhdr { font: 600 13px system-ui; fill: var(--c-0b0b0b); }
.bw .colsub { font: 9.5px system-ui; fill: var(--c-898781); }
`;

class Dsv3Bwd extends HTMLElement {
  connectedCallback() {
    this.classList.add('bw');
    if (!document.getElementById('bw-css')) {
      const st = document.createElement('style'); st.id = 'bw-css'; st.textContent = CSS; document.head.append(st);
    }
    this.render();
    addEventListener('dsv3-theme', () => this.render());
  }
  render() {
    const P = [];
    // --- pass 1: bands ---
    let y = 8;
    const SHOW_SH = this._showSh !== false;   // the shared expert is the routed chain again, minus the a2a: hideable
    const rows = ROWS.filter((r) => SHOW_SH || !r.id.startsWith('sh_')).map((r) => {
      const two = !!r.pair;
      const hb = r.chip ? 0 : H_BOX, hc = two ? H_CHIP2 : H_CHIP;
      // channel lanes below the chip text, 8px apart (above the hover-contiguity tolerance): save edges
      // ride lane A, gradient edges lane B; in a fork row each sub-column gets its OWN pair so the q and
      // kv runs never lie on top of each other (a shared lane read as one wire running into the wrong box)
      const bot = y + hb + hc;
      const row = { ...r, yBox: y, yChip: y + hb, h: hb + hc,
        laneA: two ? bot - 32 : bot - 16, laneB: two ? bot - 16 : bot - 8,
        laneAOf: (sub) => two ? bot - 32 + (sub === 1 ? 8 : 0) : bot - 16,
        laneBOf: (sub) => two ? bot - 16 + (sub === 1 ? 8 : 0) : bot - 8 };
      y += row.h;
      return row;
    });
    const total = y + 8;
    const byId = {};
    for (const r of rows) { if (r.pair) r.pair.forEach((p, i) => { byId[p.id] = { ...p, row: r, sub: i }; }); else byId[r.id] = { ...r, row: r, sub: -1 }; }
    const rowOf = (id) => byId[id]?.row ?? rows.find((r) => r.id === id);
    const W_SUBGAP = 80;   // the W column's fork gap hosts the right box's requant box and blue channel
    const bx = (col, e) => e.sub === 1 || e.row.side ? COLS[col] + SUBW + (col === 'W' ? W_SUBGAP : SUBGAP) : COLS[col];
    const bw = (e) => (e.sub >= 0 || e.row.side) ? SUBW : BW;
    const spine = (col, e) => bx(col, e) + SPX;
    const mainSpine = (col) => COLS[col] + SPX;
    const midY = (r) => r.yBox + (H_BOX - 8) / 2;
    const tapY = (r) => r.laneA - 10;   // where rails/forks tap a row's wire: a + circle (7px) must clear lane A below it
    const botY = (r) => r.yBox + H_BOX - 8;

    const SHOW_W = this._showW !== false;
    // every box is uniquely named: backward and weight-grad boxes are named after their forward op
    const box = (x, yy, w, e, cls, base) => {
      const emit = e.emit && e.emit !== e.dt ? e.emit : null;   // a fused cast: '⇒ dt'; a quantize box's own dtype says it already
      P.push(`<g class="${cls}"><rect class="box ${e.kind ?? ''}" x="${x}" y="${yy}" width="${w}" height="${H_BOX - 8}" rx="4"/>` +
      `<text class="name" x="${x + 8}" y="${yy + 13}">${esc(base ? `${base} · ${e.op}` : e.label)}</text>` +
      (e.dimsHtml ? `<text class="dims" x="${x + 8}" y="${yy + 26}">${e.dimsHtml}</text>`
        : e.dims ? `<text class="dims" x="${x + 8}" y="${yy + 26}">${esc(w < BW ? e.dims.split(' · ')[0] : e.dims)}</text>` : '') +   // narrow (fork/side) boxes keep the first clause
      // dtype tags are plain text (nothing here is clickable — no button chrome): compute dtype, then ⇒ emitted dtype
      `<text class="dtagt" x="${x + w - 8}" y="${yy + 16.5}" text-anchor="end"><tspan fill="${C(DT[e.dt])}">${e.dt}</tspan>` +
        (emit ? `<tspan fill="${C('#898781')}"> ⇒ </tspan><tspan fill="${C(DT[emit])}">${emit}</tspan>` : '') + `</text></g>`);
    };
    const chip = (x, yy, text, cls) => P.push(`<text class="tensor ${cls}" x="${x}" y="${yy}">${text}</text>`);
    const MARK = { save: 'arrs', savew: 'arrw', grad: 'arrg' };
    const arrow = (d, cls = 'wire') => P.push(`<path class="${cls}" d="${d}" marker-end="url(#${MARK[cls] ?? 'arr'})"/>`);
    const line = (d, cls = 'wire') => P.push(`<path class="${cls}" d="${d}"/>`);
    const dot = (x, yy) => P.push(`<circle cx="${x}" cy="${yy}" r="2.5" fill="${C('#898781')}"/>`);
    // gradient accumulation: a forward fan-out (x0, x1, norm2 out each feed several ops) becomes a SUM in
    // backward — drawn as a + junction on the gradient wire where the branches merge
    const pluses = [];
    const plus = (x, yy) => { pluses.push({ x, y: yy }); P.push(`<circle cx="${x}" cy="${yy}" r="7" class="box"/>` +
      `<text class="plus" x="${x}" y="${yy + 3.5}" text-anchor="middle">+</text>`); };
    const PR = 7;   // its radius: arriving branches stop at the rim
    const rot = (x, yy, t, anchor = 'middle') => P.push(`<text class="raillab" x="${x}" y="${yy}" text-anchor="${anchor}" transform="rotate(-90 ${x} ${yy})">${esc(t)}</text>`);

    // --- pass 2: boxes, chips, and the per-row edges ---
    const saves = [];   // {x, y, to: ['B:id' | 'W:id'], srcRow}
    for (const r of rows) {
      const entries = r.pair ? r.pair.map((p) => byId[p.id]) : [byId[r.id]];
      for (const e of entries) {
        const xF = bx('F', e), xB = bx('B', e), xW = bx('W', e), w = bw(e), sF = spine('F', e), sB = spine('B', e);
        if (r.chip) {
          chip(sF + 12, r.yChip + 12, `⇢ ${esc(r.chip.name)} · ${dtSpan(r.chip.dt)}`, 'tsave');
          chip(sB + 12, r.yChip + 12, `${esc(r.dchip.name)} · ${dtSpan(r.dchip.dt)}`, 'tgrad');
          saves.push({ x: sF, lane: r.laneA, to: r.chip.saved, srcRow: r });
          continue;
        }
        if (e.F) {
          box(xF, r.yBox, w, e.F, 'fop');
          const saved = !!e.out.saved, nm = esc(e.out.name);
          const aux = e.out.aux && !r.pair ? ` <tspan class="tdim">(+ ${esc(e.out.aux)})</tspan>` : '';
          chip(sF + 12, r.yChip + 12, `${saved ? '⇢ ' : '· '}${nm}${r.pair ? '' : ` · ${dtSpan(e.out.dt)}`}${aux}`, saved ? 'tsave' : 'tidle');
          if (r.pair) chip(sF + 12, r.yChip + 24, dtSpan(e.out.dt), '');
          if (saved) saves.push({ x: sF, lane: r.laneAOf(e.sub), to: e.out.saved, srcRow: r });
        } else if (!(r.fSkip && SHOW_SH)) line(`M ${sF} ${r.yBox} V ${botY(r)}`);   // backward-only row: the forward wire passes through
        if (e.B) {
          const B = { ...e.B, kind: e.B.kind ?? e.F?.kind };
          if (B.dims == null) B.dims = B.op === 'dgrad'
            ? (e.F.dims.match(/^(.*?) → (.*?)( per expert)?$/) ?? [null, e.F.dims, '', '']).slice(1).reduce((_, __, ___, m) => `${m[1]} → ${m[0]}${m[2] ?? ''}`)
            : e.F.dims;
          box(xB, r.yBox, w, B, 'bop', e.F?.label);   // a backward-only box carries its own label
        }
        else line(`M ${sB} ${r.yBox + H_BOX - 8} V ${r.yBox}`);   // identity backward: the gradient passes straight through (no box)
        // derived gradient name: d(name), with a trailing parenthetical kept outside — d(q) (pre-RoPE)
        const dn = e.dout.name ?? e.out?.name.replace(/^(.*?)( \(.*\))?$/, (_, a, b) => `d(${a})${b ?? ''}`);
        chip(sB + 12, r.yChip + 12, `${esc(dn)}${r.pair ? '' : ` · ${dtSpan(e.dout.dt)}`}`, 'tgrad');
        if (r.pair) chip(sB + 12, r.yChip + 24, dtSpan(e.dout.dt), '');
        if (e.W && SHOW_W) {
          // the box's second line names the parameter(s) whose gradient it produces, and the output dtype
          const pname = (n) => 'd' + esc(n).replace(/\^(\w+)/, '<tspan class="sup">$1</tspan>').replace(/_(\w+)/, '<tspan class="sub">$1</tspan>');
          const names = (e.param ?? ['W']).map(pname).join(', ');
          const rest = e.W.dims.split(' · ').slice(1).join(' · ');
          const tail = w < BW || !rest ? '' : ` · ${esc(rest)}`;
          const formula = w < BW || e.W.op !== 'wgrad' ? '' : ' = Xᵀ·dY';   // narrow boxes: just the names and the output dtype
          box(xW, r.yBox, w, { ...e.W, dims: null, dimsHtml: `${names}${formula}${tail}` }, 'wop', e.F?.label);
          // dY: tapped off the gradient wire (fork dot at lane B), right along the lane, up into the box bottom
          const lb = r.laneBOf(e.sub);
          dot(sB, lb);
          arrow(`M ${sB} ${lb} H ${xW + 12} V ${botY(r) + 2}`, 'grad');
        }
      }
    }
    // --- spines: F flows down, B flows up; side rows (router) are bypassed by the main spine ---
    const main = rows.filter((r) => !r.side);
    for (let i = 0; i < main.length - 1; i++) {
      const r = main[i], n = main[i + 1];
      const es = r.pair ? r.pair.map((p) => byId[p.id]) : [byId[r.id]];
      const ns = n.pair ? n.pair.map((p) => byId[p.id]) : [byId[n.id]];
      const from = r.chip ? r.yChip + 16 : botY(r);
      if (es.length === ns.length) for (let k = 0; k < es.length; k++) {
        const f = spine('F', es[k]), b = spine('B', es[k]);
        if (r.id === 'combine' && SHOW_SH) { // F: combine out hops the shared chain by a rail (exit stub to the tap); B: the quantize below feeds the combine bwd
          line(`M ${f} ${from} V ${tapY(r)}`);
          arrow(`M ${b} ${n.yBox} V ${from + 1}`);
          continue;
        }
        if (r.id === 'q_moe' && SHOW_SH) continue;   // the shared chain below is reached by rails, not the spine
        if (n.F || n.pair) arrow(`M ${f} ${from} V ${n.yBox - 1}`); else line(`M ${f} ${from} V ${n.yBox}`);
        if (r.chip || r.pair || r.B) arrow(`M ${b} ${n.yBox} V ${from + 1}`); else line(`M ${b} ${n.yBox} V ${from}`);   // no arrowhead into a box-less (junction) row
      } else if (ns.length === 2) {          // fork: main spine → both sub-spines
        const f = mainSpine('F'), b = mainSpine('B'), kx = COLS.F + SUBW + SUBGAP + SPX, kb = COLS.B + SUBW + SUBGAP + SPX;
        const yy = tapY(r);
        arrow(`M ${f} ${from} V ${n.yBox - 1}`); dot(f, yy); arrow(`M ${f} ${yy} H ${kx} V ${n.yBox - 1}`);
        arrow(`M ${b} ${n.yBox} V ${from + 1}`); dot(b, yy); arrow(`M ${kb} ${n.yBox} V ${yy} H ${b + 4}`);
      } else {                                // merge: both sub-spines → the next (full-width) box
        for (const e of es) { arrow(`M ${spine('F', e)} ${from} V ${n.yBox - 1}`); arrow(`M ${spine('B', e)} ${n.yBox} V ${from + 1}`); }
      }
    }
    {   // the block boundary: x2 leaves for the next block; d(x2) arrives from it
      const L = main[main.length - 1], yEnd = L.yChip + 30;
      line(`M ${mainSpine('F')} ${botY(L)} V ${yEnd}`);
      line(`M ${mainSpine('B')} ${yEnd} V ${L.B ? botY(L) : botY(L)}`);
    }
    {   // the router side rows: norm2 out forks into the GEMM, the GEMM feeds the gating, the gating's outputs leave by rails
      const n2 = rowOf('norm2'), rt = rowOf('router'), gt = rowOf('gate'), yy = tapY(n2);
      for (const col of ['F', 'B']) {
        const sx = mainSpine(col), rx = COLS[col] + SUBW + SUBGAP + SPX;
        if (col === 'F') dot(sx, yy); else plus(sx, yy);
        arrow(col === 'F' ? `M ${sx} ${yy} H ${rx} V ${rt.yBox - 1}` : `M ${rx} ${rt.yBox} V ${yy} H ${sx + PR + 1}`);
        arrow(col === 'F' ? `M ${rx} ${botY(rt)} V ${gt.yBox - 1}` : `M ${rx} ${gt.yBox} V ${botY(rt) + 1}`);   // GEMM ⇄ gating
        if (col === 'F') {   // the gating's output runs on into the dispatch (its routing indices); the gate rail and the saves fork off it
          const dp = rowOf('dispatch');
          arrow(`M ${rx} ${botY(gt)} V ${dp.yBox - 1}`);
        }
        // (B: the d(top-k weights) rail itself runs up into the gating bwd — one arrowhead, no stub)
      }
    }
    // --- rails: residuals, the shared expert's input, the top-k gate, the routed sum ---
    {
      const rail = (col, from, to, key, label) => {
        const x = COLS[col] + RAIL[key], sx = mainSpine(col), yF = tapY(from), yT = midY(to), bxT = COLS[col];
        if (col === 'F') dot(sx, yF); else plus(sx, yF);
        if (col === 'F') arrow(`M ${sx} ${yF} H ${x} V ${yT} H ${bxT - 1}`);
        else {
          // the gradient leaves on the box's output wire: a fork dot just above the box (or, where no
          // wire continues upward — the shared gate/up dgrad — a bare stub), then the rail
          const yTap = to.B ? to.yBox - 8 : midY(to);   // a box-less add: the junction sits on the pass-through wire
          if (to.id === 'sh_gate_up') line(`M ${sx} ${to.yBox} V ${yTap}`); else dot(sx, yTap);
          arrow(`M ${sx} ${yTap} H ${x} V ${yF} H ${sx - PR - 1}`);
        }
        rot(x - 4, (yF + yT) / 2, label);
      };
      rail('F', rowOf('x0'), rowOf('x1'), 'x0', 'x0 (residual) · bf16');
      rail('B', rowOf('x0'), rowOf('x1'), 'x0', 'd(x0) += d(x1) · bf16');
      rail('F', rowOf('x1'), rowOf('x2'), 'x1', 'x1 (residual) · bf16');
      rail('B', rowOf('x1'), rowOf('x2'), 'x1', 'd(x1) += d(x2) · bf16');
      if (SHOW_SH) {
        rail('F', rowOf('norm2'), rowOf('sh_gate_up'), 'norm2', 'norm2 out → shared expert · bf16');
        rail('B', rowOf('norm2'), rowOf('sh_gate_up'), 'norm2', 'd(norm2 out) += shared path · bf16');
      } else {
        // the shared expert elided: its four edges still leave and re-enter, dashed, off into "⋯"
        const n2 = rowOf('norm2'), ma = rowOf('moe_add');
        const elide = (col, x0, y0, x1, label, out) => {   // out: the stub leaves (arrowhead at the ⋯ end); else it arrives (arrowhead at the wire)
          arrow(out ? `M ${x0} ${y0} H ${x1}` : `M ${x1} ${y0} H ${x0}`, 'elide');
          P.push(`<text class="raillab" x="${x1 - 4}" y="${y0 + 3}" text-anchor="end">⋯</text>` +
            `<title>${esc(label)}</title>`);
        };
        for (const col of ['F', 'B']) {
          const sx = mainSpine(col), x = COLS[col] + RAIL.norm2, yN = tapY(n2);
          if (col === 'F') dot(sx, yN); else plus(sx, yN);
          if (col === 'F') {
            elide(col, sx, yN, x, 'norm2 out → shared expert (hidden)', true);
            elide(col, COLS.F - 1, midY(ma), x, 'shared out (hidden) → + routed + shared', false);
          } else {
            elide(col, sx - PR - 1, yN, x, 'd(norm2 out) += shared path (hidden)', false);
            const yT = ma.B ? ma.yBox - 8 : midY(ma);
            dot(sx, yT);
            elide(col, sx, yT, x, 'd(shared out) → shared down dgrad (hidden)', true);
          }
        }
      }
      const rt = rowOf('gate'), sw = rowOf('swiglu'), cb = rowOf('combine'), qm = rowOf('q_moe'), ma = rowOf('moe_add');
      for (const col of ['F', 'B']) {
        const rbx = COLS[col] + SUBW + SUBGAP, right = COLS[col] + BW;
        // gate rail: router → SwiGLU (× top-k weight); back: d(top-k weights) → router bwd
        const gx = right + 14, rx = rbx + SPX, sx = mainSpine(col);
        if (col === 'F') {   // F: tapped off the router's output stub; enters the SwiGLU box from the right (its second input)
          dot(rx, tapY(rt));
          arrow(`M ${rx} ${tapY(rt)} H ${gx} V ${midY(sw)} H ${right + 1}`);
        } else {             // B: tapped off the SwiGLU bwd's output wire (its second output); feeds the router bwd's input stub
          dot(sx, sw.yBox - 8);
          arrow(`M ${sx} ${sw.yBox - 8} H ${gx} V ${rt.laneB + 4} H ${rx} V ${botY(rt) + 1}`);
        }
        rot(gx + 9, (midY(rt) + midY(sw)) / 2, col === 'F' ? 'top-k weights · fp32' : 'd(top-k weights) · fp32');
        // the routed sum hops the shared chain: combine out → (+ routed + shared); back, d(moe out) off the add bwd's output wire
        if (!SHOW_SH) continue;   // adjacent rows: the spine runs straight through
        const hx = right + 28, y1 = tapY(cb);
        if (col === 'F') { dot(sx, y1); arrow(`M ${sx} ${y1} H ${hx} V ${midY(ma)} H ${right + 1}`); }
        else { const yT = ma.B ? ma.yBox - 8 : midY(ma), yq = tapY(qm); dot(sx, yT); arrow(`M ${sx} ${yT} H ${hx} V ${yq} H ${sx} V ${botY(qm) + 1}`); }
        rot(hx + 9, (y1 + midY(ma)) / 2, col === 'F' ? 'moe out (routed) · bf16' : 'd(moe out) (routed) · bf16');
      }
    }
    // --- save-for-backward edges: F tensor → the B / W op that reads it, entering the box from the LEFT
    // through a vertical channel in the target column's gutter (short hops inner, long hops outer) ---
    // z-order: weight-grad (blue, muted) edges first, then the amber ones — where both leave one
    // dot they share the first run, and the stash read by a backward op is the one to see
    const targets = saves.flatMap((s) => s.to.map((t) => ({ s, t })));
    targets.sort((a, b) => (b.t.startsWith('W:') ? 1 : 0) - (a.t.startsWith('W:') ? 1 : 0));
    for (const { s, t } of targets) {
      const [col, id] = t.split(':');
      const e = byId[id];
      if (!e || (col === 'W' && !SHOW_W)) continue;
      const long = rows.indexOf(e.row) - rows.indexOf(s.srcRow) > 1;
      const requant = col === 'W' && e.W?.dt === 'e4m3';   // a stash re-tiled 128×1 for the wgrad: a kernel with nothing to fuse into
      const cx = bx(col, e) - (long ? 22 : 10) - (requant ? 66 : 0);
      // amber: a stash read by a backward op · blue: a stash that exists for a weight-grad GEMM
      // entry height staggered by channel: the short (inner) edge enters just above the box's midline,
      // the long (outer) one just below — so the outer run never crosses the inner channel (no closed square)
      const ey = midY(e.row) + (long ? 5 : -3);
      dot(s.x, s.lane);
      if (!requant) arrow(`M ${s.x} ${s.lane} H ${cx} V ${ey} H ${bx(col, e) - 1}`, col === 'W' ? 'savew' : 'save');
      else {
        const qx = bx(col, e) - 64;
        arrow(`M ${s.x} ${s.lane} H ${cx} V ${ey} H ${qx - 1}`, 'savew');
        P.push(`<g class="rq"><rect class="box vector" x="${qx}" y="${ey - 8}" width="50" height="16" rx="3"/>` +
          `<text class="rqt" x="${qx + 25}" y="${ey + 3.5}" text-anchor="middle">requant ᵀ</text></g>`);
        arrow(`M ${qx + 50} ${ey} H ${bx(col, e) - 1}`, 'savew');
      }
    }
    // --- column headers ---
    for (const [col, t] of [['F', 'forward'], ['B', 'backward'], ...(SHOW_W ? [['W', 'weight grads']] : [])])
      P.push(`<text class="colhdr" x="${COLS[col]}" y="-8">${t}</text>`);

    const width = SHOW_W ? COLS.W + 2 * SUBW + W_SUBGAP + 40 : COLS.B + BW + 60;
    this.innerHTML = `<div class="bw-head"><label><input type="checkbox" data-k="w" ${SHOW_W ? 'checked' : ''}> weight-grad column</label>` +
      `<label><input type="checkbox" data-k="sh" ${SHOW_SH ? 'checked' : ''}> shared expert</label></div>` +
      `<div class="bw-scroll"><svg width="${width}" height="${total + 28}" viewBox="0 -28 ${width} ${total + 28}">` +
      `<defs>` +
      `<marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 8 4 L 0 8 z" fill="${C('#898781')}"/></marker>` +
      `<marker id="arrs" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 8 4 L 0 8 z" fill="${C('#eda100')}"/></marker>` +
      `<marker id="arrw" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 8 4 L 0 8 z" fill="${C('#2a78d6')}" fill-opacity="0.45"/></marker>` +
      `<marker id="arrg" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 8 4 L 0 8 z" fill="${C('#eb6834')}"/></marker>` +
      `</defs>${P.join('')}</svg></div>`;
    this._wireHover(this.querySelector('svg'), pluses);
    for (const cb of this.querySelectorAll('.bw-head input'))
      cb.onchange = (ev) => { this[ev.target.dataset.k === 'w' ? '_showW' : '_showSh'] = ev.target.checked; this.render(); };
  }
}
// Hover a wire → it and every wire CONTIGUOUS with it light up: a cue that
// tells a fork (a dot: the same signal continuing) from a mere crossing.
// Contiguity is geometric — an endpoint of one wire lying on another (fork
// dots, arrowheads landing on a wire) — and TERMINATES at boxes (ops) and at
// the + junctions (a sum's inputs are different signals), so a wire passing
// through a + is split there.
Dsv3Bwd.prototype._wireHover = function (svg, pluses) {
  const NS = 'http://www.w3.org/2000/svg';
  const sel = 'path.wire, path.save, path.savew, path.grad, path.elide, line.wire';
  const parse = (el) => {   // our wires are M/H/V/L polylines (or a <line>)
    if (el.tagName === 'line') return [['x1', 'y1'], ['x2', 'y2']].map(([a, b]) => [+el.getAttribute(a), +el.getAttribute(b)]);
    const t = el.getAttribute('d').trim().split(/\s+/), pts = [];
    let cmd = 'M';
    for (let i = 0; i < t.length;) {
      if (/^[MHVL]$/.test(t[i])) { cmd = t[i++]; continue; }
      const [px, py] = pts[pts.length - 1] ?? [0, 0];
      if (cmd === 'H') pts.push([+t[i++], py]);
      else if (cmd === 'V') pts.push([px, +t[i++]]);
      else { pts.push([+t[i], +t[i + 1]]); i += 2; }
    }
    return pts;
  };
  const nearPlus = (p) => pluses.some((q) => Math.hypot(p[0] - q.x, p[1] - q.y) <= 9);
  // 1. split every wire that passes THROUGH a + into two elements (the halves are different signals)
  const pieces = [];
  for (const el of [...svg.querySelectorAll(sel)]) {
    let pts = parse(el);
    const cls = el.getAttribute('class'), marker = el.getAttribute('marker-end');
    const parts = [];
    let cur = [pts[0]];
    for (let i = 1; i < pts.length; i++) {
      const [a, b] = [pts[i - 1], pts[i]];
      const hit = pluses.find((q) => Math.abs((b[0] - a[0]) * (q.y - a[1]) - (b[1] - a[1]) * (q.x - a[0])) < 1
        && Math.min(a[0], b[0]) - 1 < q.x && q.x < Math.max(a[0], b[0]) + 1 && Math.min(a[1], b[1]) - 1 < q.y && q.y < Math.max(a[1], b[1]) + 1
        && Math.hypot(q.x - a[0], q.y - a[1]) > 8 && Math.hypot(q.x - b[0], q.y - b[1]) > 8);
      if (hit) { cur.push([hit.x, hit.y]); parts.push(cur); cur = [[hit.x, hit.y]]; }
      cur.push(b);
    }
    parts.push(cur);
    // a part ENDING at a + center stops at the rim and gets an arrowhead (a sum's inputs arrive);
    // a part STARTING there begins at the far rim
    const atPlus = (p) => pluses.find((q) => Math.hypot(p[0] - q.x, p[1] - q.y) <= 4);   // (arriving wires may stop a few px short of the center)
    const rim = (a, q) => { const L = Math.hypot(q.x - a[0], q.y - a[1]) || 1; return [q.x - (q.x - a[0]) / L * 8, q.y - (q.y - a[1]) / L * 8]; };   // 8 = PR + 1
    let touched = parts.length > 1 || el.tagName === 'line';
    for (const pp of parts) {
      const n = pp.length, qe = n > 1 && atPlus(pp[n - 1]), qs = n > 1 && atPlus(pp[0]);
      if (qe) { pp[n - 1] = rim(pp[n - 2], qe); pp.endsAtPlus = true; touched = true; }
      if (qs) { pp[0] = rim(pp[1], qs); touched = true; }
    }
    if (!touched) { pieces.push({ el, pts }); continue; }
    const made = parts.map((pp, k) => {
      const ne = document.createElementNS(NS, 'path');
      ne.setAttribute('class', cls);
      ne.setAttribute('d', pp.map((p, i) => `${i ? 'L' : 'M'} ${p[0]} ${p[1]}`).join(' '));
      if ((marker && k === parts.length - 1) || pp.endsAtPlus) ne.setAttribute('marker-end', marker ?? (cls.includes('wire') ? 'url(#arr)' : cls.includes('grad') ? 'url(#arrg)' : cls.includes('savew') ? 'url(#arrw)' : 'url(#arrs)'));
      el.before(ne);
      return { el: ne, pts: pp };
    });
    el.remove();
    pieces.push(...made);
  }
  // 2. contiguity: an endpoint of one piece on a segment of another (not at a +) → same component
  const distSeg = (p, a, b) => {
    const dx = b[0] - a[0], dy = b[1] - a[1], l2 = dx * dx + dy * dy || 1;
    const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / l2));
    return Math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy));
  };
  // tolerance 5: arrowheads stop 4px short of the wire they land on; the two lanes are 8px apart
  const onPiece = (p, pc) => { for (let i = 1; i < pc.pts.length; i++) if (distSeg(p, pc.pts[i - 1], pc.pts[i]) <= 5) return true; return false; };
  const comp = pieces.map((_, i) => i);
  const find = (i) => (comp[i] === i ? i : (comp[i] = find(comp[i])));
  for (let i = 0; i < pieces.length; i++) for (let j = 0; j < pieces.length; j++) {
    if (i === j) continue;
    const ends = [pieces[i].pts[0], pieces[i].pts[pieces[i].pts.length - 1]];
    if (ends.some((p) => !nearPlus(p) && onPiece(p, pieces[j]))) comp[find(i)] = find(j);
  }
  // 3. wide transparent hit paths on top; hovering one lights its whole component
  const groups = {};
  pieces.forEach((pc, i) => (groups[find(i)] ??= []).push(pc.el));
  // the SOURCE of each component: a piece whose start lies on no other member is a root; the box whose
  // edge that root leaves from is the producer (unique per wire family) — it lights with the wires
  const rects = [...svg.querySelectorAll('rect.box')].map((r) => ({ r, x: +r.getAttribute('x'), y: +r.getAttribute('y'), w: +r.getAttribute('width'), h: +r.getAttribute('height') }));
  const byComp = {}, srcGroups = new Map();
  pieces.forEach((pc, i) => (byComp[find(i)] ??= []).push(pc));
  for (const [k, members] of Object.entries(byComp)) for (const pc of members) {
    const p0 = pc.pts[0];
    if (members.some((o) => o !== pc && onPiece(p0, o))) continue;
    const src = rects.find(({ x, y, w, h }) => p0[0] >= x - 2 && p0[0] <= x + w + 2 && (Math.abs(p0[1] - (y + h)) <= 3 || Math.abs(p0[1] - y) <= 3));
    if (src && !groups[k].includes(src.r)) { groups[k].push(src.r); (srcGroups.get(src.r) ?? srcGroups.set(src.r, []).get(src.r)).push(groups[k]); }
  }
  // and the converse: hovering a box lights what it produces — every wire family rooted at it
  const pinToggle = (all) => { const on = all.every((e) => e.classList.contains('pin')); all.forEach((e) => e.classList.toggle('pin', !on)); };
  for (const [r, gs] of srcGroups) {
    const g = r.parentElement, all = [...new Set(gs.flat())];
    g.onmouseenter = () => all.forEach((e) => e.classList.add('hl'));
    g.onmouseleave = () => all.forEach((e) => e.classList.remove('hl'));
    g.onclick = (ev) => { ev.stopPropagation(); pinToggle(all); };
    g.style.cursor = 'pointer';
  }
  svg.onclick = () => svg.querySelectorAll('.pin').forEach((e) => e.classList.remove('pin'));
  // fork dots light with the wire they sit on (the + junctions do not: they end a component)
  for (const d of svg.querySelectorAll('circle[r="2.5"]')) {
    const c = [+d.getAttribute('cx'), +d.getAttribute('cy')];
    const seen = new Set();
    pieces.forEach((pc, i) => { const k = find(i); if (!seen.has(k) && onPiece(c, pc)) { seen.add(k); groups[k].push(d); } });
  }
  pieces.forEach((pc, i) => {
    const h = document.createElementNS(NS, 'path');
    h.setAttribute('class', 'hit');
    h.setAttribute('d', pc.pts.map((p, k) => `${k ? 'L' : 'M'} ${p[0]} ${p[1]}`).join(' '));
    const g = groups[find(i)];
    h.onmouseenter = () => g.forEach((e) => e.classList.add('hl'));
    h.onmouseleave = () => g.forEach((e) => e.classList.remove('hl'));
    h.onclick = (ev) => { ev.stopPropagation(); pinToggle(g); };
    svg.append(h);
  });
};
customElements.define('dsv3-bwd', Dsv3Bwd);

