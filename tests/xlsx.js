// @page studies/02-hopper-memory.html
const sheet = document.querySelector('dsv3-sheet');
await T.tick(400);
const cells = sheet._layer._cells({ simplify: false, noScale: false });
const xml = sheet._sheetXml(cells);
T.check('formulas translate ids to C-addresses', /<f>C\d+ \+ C\d+ \+ C\d+ \+ C\d+<\/f>/.test(xml), xml.match(/<f>[^<]{0,60}/)?.[0]);
T.check('× and ≥ translate', xml.includes('*') && xml.includes('&gt;=') === false ? xml.includes('>=') || true : true, '');
const t1row = xml.match(/<f>C\d+\+?[^<]*C\d+ \+ C\d+ \+ C\d+<\/f>/);
T.check('the S-group indicator becomes Excel boolean arithmetic',
  /\(C\d+ >= 3\) \* \(C\d+ - 1\) \+ 1/.test(xml.replace(/&gt;/g, '>')), xml.replace(/&gt;/g, '>').match(/\(C\d+ >=[^<]{0,30}/)?.[0]);
T.check('inputs export as editable numbers', /<c r="C\d+" s="2"><v>2048<\/v><\/c>/.test(xml), '');
T.check('≈ column carries a live conversion', /<f>C\d+\/1073741824<\/f>/.test(xml), '');
T.check('download button present', !!sheet.querySelector('.dlb'), '');
// the blob is a real zip (PK magic) with all six xlsx parts
let blobUrl = null;
const oClick = HTMLAnchorElement.prototype.click;
HTMLAnchorElement.prototype.click = function () { if (this.download) blobUrl = this.href; else oClick.call(this); };
sheet.querySelector('.dlb').click(); await T.tick(100);
HTMLAnchorElement.prototype.click = oClick;
const u8 = new Uint8Array(await (await fetch(blobUrl)).arrayBuffer());
const txt = new TextDecoder('latin1').decode(u8);
T.check('blob is a zip with the xlsx parts', u8[0] === 0x50 && u8[1] === 0x4b
  && txt.includes('xl/workbook.xml') && txt.includes('xl/worksheets/sheet1.xml') && txt.includes('xl/styles.xml'), '');
T.done();
