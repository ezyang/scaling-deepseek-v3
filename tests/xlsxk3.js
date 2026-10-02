// @page studies/kimi-k3-roofline.html
// The Kimi K3 page's .xlsx download (studies/sheetedit.js sheetXml): every row once, its
// formula translated to C-addresses, and the workbook RECOMPUTES to the
// page's values — each <f> is evaluated here as Excel would and checked
// against the exact rational the page renders, before and after a knob edit.
await T.tick(300);
const { sheetXml, current, setKnob } = await import('/studies/sheetedit.js');
const names = new Set([...document.querySelectorAll('.cellsheet td.nm')].map((td) => td.innerHTML));
const check = (label) => {
  const doc = new DOMParser().parseFromString(sheetXml(), 'application/xml');
  const cell = new Map([...doc.querySelectorAll('c')].map((c) => [c.getAttribute('r'), c]));
  const v = new Map(), CEILING = (x, s) => Math.ceil(x / s) * s, MAX = Math.max;
  const val = (r) => {
    if (!v.has(r)) {
      const c = cell.get('C' + r), f = c.querySelector('f')?.textContent;
      if (f && !/^[\d+\-*/().,]*$/.test(f.replace(/CEILING|MAX|C\d+|E\+\d+/g, ''))) throw new Error(`odd formula C${r}: ${f}`);
      v.set(r, f ? eval(f.replace(/C(\d+)/g, (_, k) => `val(${k})`)) : +c.querySelector('v').textContent);
    }
    return v.get(r);
  };
  const rows = [...doc.querySelectorAll('row')].filter((row) => cell.has('C' + row.getAttribute('r')) && row.getAttribute('r') !== '1');
  const now = current(), bad = [];
  for (const row of rows) {
    const r = row.getAttribute('r'), [n, d] = now.get([...names].find((x) => x.replace(/<sup>(.*?)<\/sup>/g, '^$1').replace(/<sub>(.*?)<\/sub>/g, '_$1') === row.querySelector('c t').textContent));
    const want = Number(n) / Number(d), got = val(r);
    if (Math.abs(got - want) > 1e-9 * Math.abs(want)) bad.push(`${row.querySelector('c t').textContent}: ${got} vs ${want}`);
  }
  T.check(`${label}: one row per cell name`, rows.length === names.size, `${rows.length} vs ${names.size}`);
  T.check(`${label}: every formula recomputes to the page's value`, bad.length === 0, bad.slice(0, 5).join(' | '));
  return doc;
};
const doc = check('published');
T.check('sections carry the h2s', [...doc.querySelectorAll('row')].some((r) => r.textContent === 'Expert parallelism over NVLink'), '');
setKnob('GPUs', [4096n, 1n]);
check('GPUs 4096');
T.check('… knob edits export as the leaf value', /<v>4096<\/v>/.test(sheetXml()), '');
setKnob('GPUs', null);

// one download for the whole page, after the last sheet; the blob is a real xlsx zip
const btns = document.querySelectorAll('.sheet-dl'), sheets = document.querySelectorAll('.cellsheet');
T.check('one download button, below every sheet', btns.length === 1 && document.querySelectorAll('.cs-dl').length === 0
  && (sheets[sheets.length - 1].compareDocumentPosition(btns[0]) & Node.DOCUMENT_POSITION_FOLLOWING), btns.length);
let blobUrl = null;
const oClick = HTMLAnchorElement.prototype.click;
HTMLAnchorElement.prototype.click = function () { if (this.download) blobUrl = this.href; else oClick.call(this); };
btns[0].click(); await T.tick(100);
HTMLAnchorElement.prototype.click = oClick;
const u8 = new Uint8Array(await (await fetch(blobUrl)).arrayBuffer());
const txt = new TextDecoder('latin1').decode(u8);
T.check('blob is a zip with the xlsx parts', u8[0] === 0x50 && u8[1] === 0x4b && txt.includes('xl/worksheets/sheet1.xml') && txt.includes('xl/styles.xml'), '');
T.done();
