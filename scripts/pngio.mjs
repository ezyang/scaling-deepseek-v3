// Zero-dep PNG codec for the pixel goldens: decodes Chrome screenshots
// (8-bit RGB/RGBA, non-interlaced) to raw RGBA and encodes RGBA back.
import { inflateSync, deflateSync } from 'node:zlib';

export function decode(buf) {
  let off = 8, w = 0, h = 0, ct = 0;
  const idat = [];
  while (off < buf.length) {
    const len = buf.readUInt32BE(off), type = buf.toString('latin1', off + 4, off + 8);
    const data = buf.subarray(off + 8, off + 8 + len);
    if (type === 'IHDR') {
      w = data.readUInt32BE(0); h = data.readUInt32BE(4); ct = data[9];
      if (data[8] !== 8 || (ct !== 2 && ct !== 6) || data[12] !== 0) throw new Error('unsupported png');
    } else if (type === 'IDAT') idat.push(data);
    off += 12 + len;
  }
  const bpp = ct === 6 ? 4 : 3, stride = w * bpp, s = stride + 1;
  // unfilter in place; a zero scanline above row 0 stands in for "no previous row"
  const px = Buffer.alloc(s + h * s);
  inflateSync(Buffer.concat(idat)).copy(px, s);
  for (let r = s + 1; r < px.length; r += s) {
    const f = px[r - 1], end = r + stride;
    if (f === 1) for (let i = r + bpp; i < end; i++) px[i] += px[i - bpp];
    else if (f === 2) for (let i = r; i < end; i++) px[i] += px[i - s];
    else if (f === 3) {
      for (let i = r; i < r + bpp; i++) px[i] += px[i - s] >> 1;
      for (let i = r + bpp; i < end; i++) px[i] += (px[i - bpp] + px[i - s]) >> 1;
    } else if (f === 4) {
      for (let i = r; i < r + bpp; i++) px[i] += px[i - s];   // Paeth with a = c = 0 picks b
      for (let i = r + bpp; i < end; i++) {
        const a = px[i - bpp], b = px[i - s], c = px[i - s - bpp];
        const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
        px[i] += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      }
    }
  }
  const out = Buffer.alloc(w * h * 4);
  for (let y = 0, r = s + 1; y < h; y++, r += s) {
    if (ct === 6) { px.copy(out, y * stride, r, r + stride); continue; }
    for (let x = 0, o = y * w * 4; x < stride; x += 3, o += 4) {
      out[o] = px[r + x]; out[o + 1] = px[r + x + 1]; out[o + 2] = px[r + x + 2]; out[o + 3] = 255;
    }
  }
  return { w, h, data: out };
}

const CRC = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; }
  return t;
})();
const crc32 = (b) => { let c = ~0; for (const x of b) c = CRC[(c ^ x) & 255] ^ (c >>> 8); return (~c) >>> 0; };
const chunk = (type, data) => {
  const out = Buffer.alloc(12 + data.length);
  out.writeUInt32BE(data.length, 0);
  out.write(type, 4, 'latin1');
  data.copy(out, 8);
  out.writeUInt32BE(crc32(out.subarray(4, 8 + data.length)), 8 + data.length);
  return out;
};

export function encode(w, h, data) {
  const stride = w * 4;
  const raw = Buffer.alloc((stride + 1) * h);   // filter 0 per scanline
  for (let y = 0; y < h; y++) data.copy(raw, y * (stride + 1) + 1, y * stride, (y + 1) * stride);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 6;
  return Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw, { level: 9 })), chunk('IEND', Buffer.alloc(0))]);
}
