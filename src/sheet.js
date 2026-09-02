// <dsv3-sheet>: the full model's formula sheet — a spreadsheet-like readout
// of the cell graph (src/cells.js) that the bound local widget prices from.
// Presentation only: every number and formula string comes from the layer's
// _cells; this file renders, edits (by driving the widget's own controls),
// jumps, and exports. The element is DEFINED in viewer.js so upgrade order
// (sheet after layer) is unchanged.
import { fmtBytes, fmtP } from './viewer.js';   // runtime-only (no eval-time cycle)
import { mmSig, markSig, HAZIZA_CFG } from './localmodel.js';

const el = (tag, cls) => { const e = document.createElement(tag); if (cls) e.className = cls; return e; };
const esc = (s) => String(s).replace(/[<>&]/g, (c) => ({ '<': '&lt;', '>': '&gt;', '&': '&amp;' }[c]));

// ---- <dsv3-sheet> — the full model's formula sheet --------------------------
// A spreadsheet-like readout of the cell graph (src/cells.js) that the bound
// local widget prices from: NAME · quantity · formula · live value. The names
// are the coordinates the fit chart's tooltips speak; rows update live as
// the widget's knobs move — same cells, one source of truth by construction.
const SHEET_CSS = `
dsv3-sheet { display: block; margin: 14px 0; position: relative; }
.cellsheet { font: 12px system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--c-0b0b0b);
  border: 1px solid var(--c-e1e0d9); border-radius: 6px; background: var(--c-fcfcfb); padding: 8px 12px; position: relative; }
.cellsheet td.fx .cellref { cursor: pointer; }
.cellsheet .hd { color: var(--c-52514e); font-size: 11.5px; margin-bottom: 5px; }
.cellsheet table { border-collapse: collapse; width: 100%; table-layout: fixed; }   /* fixed columns: values changing never resizes them */
.cellsheet td.lb { overflow: hidden; text-overflow: ellipsis; }
.cellsheet th { text-align: left; font-size: 10.5px; color: var(--c-898781); font-weight: 600; padding: 1px 10px 3px 2px; }
.cellsheet th.vl, .cellsheet td.vl { text-align: right; }
.cellsheet td { padding: 1.5px 10px 1.5px 2px; border-top: 1px solid var(--c-f0efe9); font-size: 11.5px; vertical-align: baseline; }
.cellsheet td.nm { font: 600 11px ui-monospace, monospace; color: var(--c-2a78d6); }
.cellsheet td.fx { font: 11px ui-monospace, monospace; color: var(--c-52514e); }   /* long rate decompositions wrap */
.cellsheet td.fx .cellref { color: var(--c-2a78d6); font-weight: 600; }
.cellsheet td.vl { font-variant-numeric: tabular-nums; white-space: nowrap; }
.cellsheet td.ap { color: var(--c-898781); }
.cellsheet tr.hl td { background: var(--c-fff8ea); }
.cellsheet td.lb { white-space: nowrap; }   /* labels are one line by fiat */
/* edit affordances live IN the exact-value cell: the ± glyphs are ::before/
   ::after content (never copied), with generous padding for the hitbox;
   toggle values wear button language (dashed underline, hover face) and
   the WHOLE cell is the target */
.cellsheet .sb { cursor: pointer; color: var(--c-52514e); padding: 2px 8px; user-select: none; }
@media (hover: hover) { .cellsheet .sb:hover { color: var(--c-0b0b0b); } }
.cellsheet .sb.dn::before { content: '−'; font: 600 11px ui-monospace, monospace; }
.cellsheet .sb.up::after { content: '+'; font: 600 11px ui-monospace, monospace; }
.cellsheet .sb.dis, .cellsheet .sb.dis:hover { color: var(--c-d5d4cc); cursor: default; }
/* EDITABLE cells wear a tinted face (the button language); a pinned toggle
   renders as a plain cell — clearly not clickable */
.cellsheet td.vl.edv, .cellsheet td.vl.tg { background: var(--c-eef4fc); }
.cellsheet td.vl.tg { cursor: pointer; }
@media (hover: hover) { .cellsheet td.vl.tg:hover { background: var(--c-dcebfa); } }
.cellsheet td.lb .lnk { color: var(--c-2a78d6); cursor: pointer; }
@media (hover: hover) { .cellsheet td.lb .lnk:hover { text-decoration: underline; } }
/* the jump spotlight: everything but the target grays out behind the
   ring's giant veil; a click/scroll/key anywhere dismisses it */
.cell-spot { position: fixed; z-index: 60; pointer-events: none; border-radius: 8px;
  box-shadow: 0 0 0 2px var(--c-eda100), 0 0 0 200vmax rgba(252, 252, 251, 0.78);
  animation: spotin 0.18s ease-out; }
@keyframes spotin { from { box-shadow: 0 0 0 2px rgba(237, 161, 0, 0), 0 0 0 200vmax rgba(252, 252, 251, 0); } }
`;
// ---- minimal store-only ZIP (the xlsx download) ---------------------------
const CRC_T = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; }
  return t;
})();
const crc32 = (u8) => { let c = 0xffffffff; for (let i = 0; i < u8.length; i++) c = CRC_T[(c ^ u8[i]) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
function zipStore(files) {   // files: [name, string][] → Blob (method 0, no compression)
  const enc = new TextEncoder();
  const parts = [], central = [];
  let off = 0;
  const u16 = (v) => [v & 255, (v >> 8) & 255];
  const u32 = (v) => [v & 255, (v >> 8) & 255, (v >> 16) & 255, (v >>> 24) & 255];
  for (const [name, text] of files) {
    const nm = enc.encode(name), data = enc.encode(text), crc = crc32(data);
    const head = Uint8Array.from([...u32(0x04034b50), ...u16(20), ...u16(0), ...u16(0), ...u16(0), ...u16(0),
      ...u32(crc), ...u32(data.length), ...u32(data.length), ...u16(nm.length), ...u16(0)]);
    parts.push(head, nm, data);
    central.push(Uint8Array.from([...u32(0x02014b50), ...u16(20), ...u16(20), ...u16(0), ...u16(0), ...u16(0), ...u16(0),
      ...u32(crc), ...u32(data.length), ...u32(data.length), ...u16(nm.length), ...u16(0), ...u16(0), ...u16(0), ...u16(0),
      ...u32(0), ...u32(off)]), nm);
    off += head.length + nm.length + data.length;
  }
  const cdSize = central.reduce((t, u) => t + u.length, 0);
  const eocd = Uint8Array.from([...u32(0x06054b50), ...u16(0), ...u16(0), ...u16(files.length), ...u16(files.length),
    ...u32(cdSize), ...u32(off), ...u16(0)]);
  return new Blob([...parts, ...central, eocd], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' });
}

export class Dsv3Sheet extends HTMLElement {
  connectedCallback() {
    const style = document.createElement('style'); style.textContent = SHEET_CSS;
    this._root = el('div', 'cellsheet');
    this.append(style, this._root);
    const lid = this.getAttribute('layer');
    const bind = () => {   // the layer upgrades async; poll briefly until it's live
      const l = document.getElementById(lid);
      if (l?._cells) { this._layer = l; l.addEventListener('recipe', () => this.sync()); this.sync(); }
      else setTimeout(bind, 30);
    };
    if (lid) bind();
    this._root.addEventListener('change', (ev) => {
      if (ev.target.closest?.('.simp')) { this._sim = ev.target.checked; this.sync(); }
      else if (ev.target.closest?.('.nos')) { this._nos = ev.target.checked; this.sync(); }
    });
    this._root.addEventListener('click', (ev) => {
      if (ev.target.closest?.('.dlb')) this._download();
    });
    // the Haziza preset applies through the layer (tween + resync as
    // usual); clicking the LIT button toggles back to the config you came
    // from (the segGrp chips' ping-pong convention)
    this._root.addEventListener('click', (ev) => {
      if (!ev.target.closest?.('.hzb') || !this._layer) return;
      const l = this._layer;
      const snap = () => ({
        matmuls: { ...l.matmuls }, marks: { ...l.marks },
        transposed: l.transposed, fp8Params: l.fp8Params,
        ep: l.ep, pp: l.pp, zero: l.zero, world: l.world,
        stage: l.stage, sched: l.sched, vpp: l.vpp, fold: l.fold,
      });
      if (this._hzOn()) {
        const back = this._hzPrev;
        if (!back) return;
        this._hzPrev = snap();
        l.setLocal(() => Object.assign(l, { ...back, matmuls: { ...back.matmuls }, marks: { ...back.marks } }));
        return;
      }
      this._hzPrev = snap();
      l.setLocal(() => Object.assign(l, {
        matmuls: { ...HAZIZA_CFG.matmuls }, marks: { ...HAZIZA_CFG.marks },
        transposed: HAZIZA_CFG.transposed, fp8Params: HAZIZA_CFG.fp8Params,
        ep: HAZIZA_CFG.ep, pp: HAZIZA_CFG.pp, zero: HAZIZA_CFG.zero, world: HAZIZA_CFG.world,
        stage: HAZIZA_CFG.stage, sched: HAZIZA_CFG.sched, vpp: 2, fold: 'reflect',
      }));
    });
    // formula variables get the same hover card as the chart's numbers
    // (.lv-tip styling rides the layer's stylesheet); clicking one jumps to
    // its row
    this._tip = el('div', 'lv-tip');
    this.append(this._tip);   // outside _root: sync() rewrites _root.innerHTML
    const fmtC = (c) => {
      const rawv = c.unit === 'B/e' ? `${c.value} B/elem`
        : c.value.toLocaleString('en-US', { maximumFractionDigits: 2 }) + (c.unit === 'B' ? ' B' : c.unit === 'B/tok' ? ' B/tok' : '');
      return c.unit === 'B' ? `${fmtBytes(c.value)} (${rawv})` : rawv;
    };
    this._root.addEventListener('mousemove', (ev) => {
      const ref = ev.target.closest?.('.cellref');
      const c = ref && this._layer?._cells?.().byId.get(ref.textContent);
      if (!c) { this._tip.style.display = 'none'; return; }
      this._tip.textContent = `${c.id} · ${c.label}\n${c.expr ? `= ${c.expr} ` : ''}= ${fmtC(c)}`;
      const r = this.getBoundingClientRect();
      this._tip.style.left = Math.min(ev.clientX - r.left + 14, r.width - 300) + 'px';
      this._tip.style.top = (ev.clientY - r.top + 14) + 'px';
      this._tip.style.display = 'block';
    });
    this._root.addEventListener('mouseleave', () => { this._tip.style.display = 'none'; });
    this._root.addEventListener('mousedown', (ev) => {
      if (ev.button !== 0) return;
      const sb = ev.target.closest?.('.sb');
      const tg = ev.target.closest?.('td.vl.tg');
      if (sb?.classList.contains('dis')) return;
      const tr2 = (sb ?? tg)?.closest('tr');
      if (!tr2?.dataset.et) return;
      ev.preventDefault();
      this._edit(tr2.dataset.et, tr2.dataset.ek, sb?.classList.contains('up') ? 'up' : 'dn');
    });
    this._root.addEventListener('click', (ev) => {
      const ref = ev.target.closest?.('.cellref');
      if (ref) { this.reveal(ref.textContent); return; }
      const tr = ev.target.closest?.('tr');
      if (!tr?.dataset.cell) return;
      const lnk = ev.target.closest?.('td.lb .lnk');
      if (lnk) this._jump(tr.dataset.jk, tr.dataset.jc);
      // clicking a row highlights it (click the highlighted row to clear);
      // the same persistent .hl the tooltip's jump uses, so it survives
      // syncs. SELECTION gestures are not row clicks: a click that ends a
      // drag-select (live selection) or extends into a double-click is
      // skipped — the action runs on a short fuse the second click defuses
      clearTimeout(this._hlT);
      if (ev.detail > 1 || !getSelection().isCollapsed) return;
      this._hlT = setTimeout(() => {
        this._hl = this._hl === tr.dataset.cell && !lnk ? null : tr.dataset.cell;
        this.sync();
      }, 250);
    });
  }
  // build the worksheet XML: one row per cell, the VALUE column carrying
  // LIVE formulas (ids → C-column addresses; × → *, ≥ → >= — booleans
  // coerce in arithmetic in both Excel and Sheets). Inputs export as plain
  // numbers, so the downloaded workbook RECOMPUTES when you edit them.
  _sheetXml(cells) {
    const rowOf = new Map(cells.cells.map((c, i) => [c.id, i + 2]));
    const xf = (e2) => e2.replace(/[A-Z]\d+[a-z]?/g, (id) => `C${rowOf.get(id) ?? '#REF!'}`)
      .replace(/×/g, '*').replace(/≥/g, '>=');
    const xesc = (t2) => String(t2).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    const str = (ref, t2, st = 0) => `<c r="${ref}" t="inlineStr"${st ? ` s="${st}"` : ''}><is><t xml:space="preserve">${xesc(t2)}</t></is></c>`;
    const APPROX = { B: [2 ** 30, 'GiB'], 'B/tok': [1024, 'KiB/tok'], p: [1e9, 'B params'] };
    const rows = [
      `<row r="1">${['cell', 'quantity', 'value', '≈', ''].map((h, i2) => str('ABCDE'[i2] + '1', h, 1)).join('')}</row>`,
      ...cells.cells.map((c, i) => {
        const r = i + 2;
        const val = c.expr
          ? `<c r="C${r}" s="2"><f>${xesc(xf(c.expr))}</f></c>`
          : `<c r="C${r}" s="2"><v>${c.value}</v></c>`;
        const ap = APPROX[c.unit ?? ''];
        return `<row r="${r}">${str('A' + r, c.id)}${str('B' + r, '  '.repeat(c.depth ?? 0) + c.label)}${val}`
          + (ap ? `<c r="D${r}" s="3"><f>C${r}/${ap[0]}</f></c>${str('E' + r, ap[1], 4)}` : '')
          + '</row>';
      }),
    ];
    return '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
      + '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
      + '<cols><col min="1" max="1" width="7" customWidth="1"/><col min="2" max="2" width="46" customWidth="1"/>'
      + '<col min="3" max="3" width="20" customWidth="1"/><col min="4" max="4" width="12" customWidth="1"/>'
      + '<col min="5" max="5" width="9" customWidth="1"/></cols>'
      + `<sheetData>${rows.join('')}</sheetData></worksheet>`;
  }
  _download() {
    const cells = this._layer?._cells?.({ simplify: !!this._sim, noScale: !!this._nos });
    if (!cells) return;
    const XMLNS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main';
    const files = [
      ['[Content_Types].xml', '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        + '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        + '<Default Extension="xml" ContentType="application/xml"/>'
        + '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
        + '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        + '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/></Types>'],
      ['_rels/.rels', '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/></Relationships>'],
      ['xl/workbook.xml', '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        + `<workbook xmlns="${XMLNS}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">`
        + '<sheets><sheet name="DSv3 memory" sheetId="1" r:id="rId1"/></sheets></workbook>'],
      ['xl/_rels/workbook.xml.rels', '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
        + '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/></Relationships>'],
      ['xl/styles.xml', '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        + `<styleSheet xmlns="${XMLNS}">`
        + '<numFmts count="2"><numFmt numFmtId="164" formatCode="#,##0.#####"/><numFmt numFmtId="165" formatCode="#,##0.0##"/></numFmts>'
        + '<fonts count="3"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font>'
        + '<font><sz val="11"/><color rgb="FF898781"/><name val="Calibri"/></font></fonts>'
        + '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
        + '<borders count="1"><border/></borders><cellStyleXfs count="1"><xf/></cellStyleXfs>'
        + '<cellXfs count="5"><xf/><xf fontId="1" applyFont="1"/><xf numFmtId="164" applyNumberFormat="1"/>'
        + '<xf numFmtId="165" applyNumberFormat="1" fontId="2" applyFont="1"/><xf fontId="2" applyFont="1"/></cellXfs></styleSheet>'],
      ['xl/worksheets/sheet1.xml', this._sheetXml(cells)],
    ];
    const url = URL.createObjectURL(zipStore(files));
    const a = document.createElement('a');
    a.href = url; a.download = 'dsv3-memory-sheet.xlsx';
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  }
  // is the layer sitting exactly on the Haziza config?
  _hzOn() {
    const l = this._layer;
    return !!l && mmSig(l.matmuls) === mmSig(HAZIZA_CFG.matmuls)
      && markSig(l.marks) === markSig(HAZIZA_CFG.marks)
      && !l.transposed && !!l.fp8Params
      && ['ep', 'pp', 'zero', 'world', 'stage', 'sched'].every((k5) => l[k5] === HAZIZA_CFG[k5]);
  }
  // can this edit go that way RIGHT NOW? Read the widget's own controls —
  // the same elements _edit clicks — so the sheet's disabled states can't
  // disagree with the diagram's
  _canEdit(t, k) {
    const l = this._layer;
    if (!l) return { dn: false, up: false, tg: false };
    const host = l.parentElement;
    if (t === 'step') {
      const btns = host.querySelector(`.stp[data-knob="${k}"]`)?.querySelectorAll('button');
      return { dn: !!btns && !btns[0].disabled, up: !!btns && !btns[1].disabled };
    }
    if (t === 'seg') {
      const btns = [...(host.querySelector(`.stp[data-knob="${k}"]`)?.querySelectorAll('button') ?? [])];
      const i = btns.findIndex((b) => b.classList.contains('on'));
      return { dn: i > 0, up: i >= 0 && i < btns.length - 1 };
    }
    if (t === 'flip') {
      const btns = [...(host.querySelector(`.stp[data-knob="${k}"]`)?.querySelectorAll('button') ?? [])];
      return { tg: btns.some((b) => !b.classList.contains('on') && !b.disabled) };
    }
    if (t === 'cb') { const i2 = host.querySelector(`input[data-knob="${k}"]`); return { tg: !!i2 && !i2.disabled }; }
    if (t === 'dt') return { tg: !!l.querySelector(`button[data-dt="${k}"]:not([disabled])`) };
    if (t === 'mark') return { tg: !!l.querySelector(`button[data-mark="${k}"]`) };
    return { tg: false };
  }
  // sheet edits drive the widget's OWN controls (never a second mutation
  // path): steppers step, segments step/flip, checkboxes/dtype/mark buttons
  // click — so bounds, tweens, URL state and the diagram all follow
  _edit(t, k, dir) {
    const l = this._layer;
    if (!l) return;
    const host = l.parentElement;
    if (t === 'step') host.querySelector(`.stp[data-knob="${k}"]`)?.querySelectorAll('button')[dir === 'up' ? 1 : 0]?.click();
    else if (t === 'seg') {
      const btns = [...(host.querySelector(`.stp[data-knob="${k}"]`)?.querySelectorAll('button') ?? [])];
      const i = btns.findIndex((b) => b.classList.contains('on'));
      btns[i + (dir === 'up' ? 1 : -1)]?.click();
    } else if (t === 'flip') {
      const btns = [...(host.querySelector(`.stp[data-knob="${k}"]`)?.querySelectorAll('button') ?? [])];
      btns.find((b) => !b.classList.contains('on'))?.click();
    } else if (t === 'cb') host.querySelector(`input[data-knob="${k}"]`)?.click();
    else if (t === 'dt') l.querySelector(`button[data-dt="${k}"]:not([disabled])`)?.click();
    else if (t === 'mark') l.querySelector(`button[data-mark="${k}"]`)?.click();
  }
  // jump to the diagram: a model input lands on its controlling knob, an
  // activation row on its chip (or aux label); the target pulses
  _jump(jk, jc) {
    const l = this._layer;
    if (!l) return;
    const el2 = jk
      ? (l.parentElement.querySelector(`.stp[data-knob="${jk}"]`)
        ?? l.parentElement.querySelector(`input[data-knob="${jk}"]`)?.closest('label'))
      : jc ? l.querySelector(`.lv-scroll g[data-chip="${jc}"]`) : null;
    if (!el2) return;
    el2.scrollIntoView({ block: 'center', inline: 'center' });
    // SPOTLIGHT the target: a fixed ring whose giant shadow veils everything
    // else; any click / scroll / key dismisses it
    document.querySelector('.cell-spot')?.remove();
    // the instant scroll above is synchronous — measure directly (no rAF:
    // headless on-demand-frame modes may never fire one)
    const r = el2.getBoundingClientRect();
    const d = el('div', 'cell-spot');
    d.style.cssText = `left:${(r.left - 7).toFixed(1)}px;top:${(r.top - 6).toFixed(1)}px;`
      + `width:${(r.width + 14).toFixed(1)}px;height:${(r.height + 12).toFixed(1)}px;`;
    document.body.append(d);
    // dismissal listens for USER-initiated input only (wheel / pointer /
    // key — every real scroll starts as one of these): the jump's own
    // programmatic scroll fires a 'scroll' event on a racy async schedule
    // that would self-dismiss the spotlight. setTimeout: the click that
    // triggered this jump must not dismiss it via its own bubbling events.
    const evs = ['pointerdown', 'wheel', 'keydown'];
    const off = () => { d.remove(); evs.forEach((e2) => removeEventListener(e2, off, true)); };
    setTimeout(() => evs.forEach((e2) => addEventListener(e2, off, true)), 0);
  }
  // tooltip jump target: scroll the row into view and highlight it (the
  // highlight survives re-syncs until the next jump)
  reveal(id) {
    this._hl = id;
    this.sync();
    this._root.querySelector(`tr[data-cell="${id}"]`)?.scrollIntoView({ block: 'center' });
  }
  sync() {
    const cells = this._layer?._cells?.({ simplify: !!this._sim, noScale: !!this._nos });
    if (!cells) return;
    // the exact value is the PRIMARY column (byte counts are exact — every
    // divisor is a power of two on integer counts); the rounded reading is
    // its own convenience column
    const raw = (c) => c.unit === 'B/e' ? `${c.value} B/elem`   // dyadic: String() is exact
      : c.value.toLocaleString('en-US', { maximumFractionDigits: 2 })
      + (c.unit === 'B' ? ' B' : c.unit === 'B/tok' ? ' B/tok' : '');
    const approx = (c) => c.unit === 'B' ? fmtBytes(c.value)
      : c.unit === 'p' ? fmtP(c.value)
        : c.unit === 'B/tok' ? `${(c.value / 1024).toFixed(1)} KiB` : '';
    const fx = (c) => !c.expr ? `<span style="color:var(--c-898781)">${esc(c.note ?? '(model input)')}</span>`
      : '= ' + c.expr.split(/([A-Z]\d+[a-z]?)/).map((tok) =>
        /^[A-Z]\d+[a-z]?$/.test(tok) ? `<span class="cellref">${tok}</span>` : esc(tok)).join('');
    this._root.innerHTML = '<div class="hd">the fit chart’s formula sheet — every number the chart below shows is one of these cells, '
      + 'computed by evaluating exactly the formula printed here (hover a chart number for its formula; click to pin, then click names to drill)'
      + `<span style="float:right;display:inline-flex;gap:14px;align-items:center;">`
      + (() => {
        if (this._layer?.fold === 'wrap') return '';   // the Megatron-family pages: no Hopper cross-check preset
        const on = this._hzOn();
        return `<button class="hzb" style="font:11px ui-monospace,monospace;padding:1px 8px;border:1px solid `
          + (on ? 'var(--c-eda100);background:var(--c-fff8ea);font-weight:600;' : 'var(--c-c3c2b7);background:var(--c-ffffff);')
          + `border-radius:4px;cursor:pointer;" title="${on
            ? 'click again — back to the config you came from'
            : 'the roofline analysis this essay credits (Daniel Haziza): dsv3-style fp8 GEMMs with a BF16 attn-out stash, '
              + 'e4m3+ᵀ-resident params, and its exact stash policy — one click to line the sheet up against his numbers'}">Haziza cfg</button>`;
      })()
      + `<label class="simp" style="display:inline-flex;align-items:center;gap:4px;cursor:pointer;"><input type="checkbox"${this._sim ? ' checked' : ''}> simplify — drop negligible terms</label>`
      + `<label class="nos" style="display:inline-flex;align-items:center;gap:4px;cursor:pointer;"><input type="checkbox"${this._nos ? ' checked' : ''}> no act scale factors</label>`
      + `<button class="dlb" style="font:11px ui-monospace,monospace;padding:1px 8px;border:1px solid var(--c-c3c2b7);background:var(--c-ffffff);border-radius:4px;cursor:pointer;" `
      + `title="download this sheet as .xlsx with LIVE formulas (ids become cell references) — edit the inputs in Excel/Sheets and it recomputes">⤓ .xlsx</button>`
      + '</span>'
      + (this._sim || this._nos ? `<div style="color:var(--c-8c5a19)">${[
        this._sim ? 'the lse/rstd artifacts and the final norm are dropped' : '',
        this._nos ? 'fp8 ACTIVATION stashes counted at their payload rate (the 1×128 tile-scale share dropped; weights keep theirs)' : '',
      ].filter(Boolean).join('; ')} — these values drift slightly from the (exact) chart</div>` : '')
      + '</div>'
      + '<table><colgroup><col style="width:38px"><col style="width:322px"><col style="width:150px"><col style="width:64px"><col></colgroup>'
      + '<tr><th>cell</th><th>quantity</th><th class="vl">value (exact)</th><th class="vl">≈</th><th>formula</th></tr>'
      + cells.cells.map((c) => `<tr data-cell="${c.id}"${c.id === this._hl ? ' class="hl"' : ''}`
        + `${c.ui?.k ? ` data-jk="${c.ui.k}"` : ''}${c.ui?.c ? ` data-jc="${c.ui.c}"` : ''}`
        + `${c.edit ? ` data-et="${c.edit.t}" data-ek="${c.edit.k}"` : ''}>`
        + `<td class="nm">${c.id}</td>`
        + `<td class="lb"${c.depth ? ` style="padding-left:${2 + c.depth * 14}px"` : ''}>${c.ui
          ? `<span class="lnk" title="jump to it in the diagram">${esc(c.label)}</span>` : esc(c.label)}</td>`
        + (!c.edit ? `<td class="vl">${raw(c)}</td>`
          : c.edit.t === 'step' || c.edit.t === 'seg'
            ? (() => {
              const can = this._canEdit(c.edit.t, c.edit.k);
              return `<td class="vl edv"><span class="sb dn${can.dn ? '' : ' dis'}" title="${can.dn ? 'step down (the widget’s own knob moves)' : 'at its bound'}"></span>`
                + `${raw(c)}<span class="sb up${can.up ? '' : ' dis'}" title="${can.up ? 'step up' : 'at its bound'}"></span></td>`;
            })()
            : this._canEdit(c.edit.t, c.edit.k).tg
              ? `<td class="vl tg" title="click to toggle (the widget’s own control flips)"><span class="tgv">${raw(c)}</span></td>`
              : `<td class="vl" title="pinned here — no control to flip">${raw(c)}</td>`)
        + `<td class="vl ap">${approx(c)}</td>`
        + `<td class="fx">${fx(c)}</td></tr>`).join('')
      + '</table>';
  }
}
