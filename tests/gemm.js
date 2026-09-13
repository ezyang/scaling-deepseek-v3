// @page studies/03-sol.html
// the first picture: one GEMM on both meters — 2MKN FLOP on the tensor cores
// vs X + W + Y bytes through HBM; the tokens knob walks a shape from
// weight-read-bound to compute-bound, the dtype knob halves bytes and doubles peak
const w = document.querySelector('dsv3-gemm');
await T.tick(300);
const ro = () => w.querySelector('.ro').textContent;
const bar = (k) => +w.querySelector(`rect[data-bar="${k}"]`).dataset.true;
const sheet = (k) => w.querySelector(`[data-sheet="${k}"]`).textContent;
T.check('sheet: 1 µs = 1.98 GFLOP fp8 · 3.35 MB = 3.35 M fp8 numbers · 591 FLOP per HBM byte',
  sheet('tensor cores') === '1.98 GFLOP' && sheet('HBM') === '3.35 MB = 3.35 M fp8 numbers' && sheet('FLOP per HBM byte') === '591', [sheet('tensor cores'), sheet('HBM'), sheet('FLOP per HBM byte')].join(' | '));
T.check('expert gate+up, 4,096 tokens, fp8: 240.5 GFLOP = 121.5 µs; 92.3 MB = 27.5 µs', Math.abs(bar('compute') - 121.5e-6) < 0.1e-6 && Math.abs((bar('X') + bar('W') + bar('Y')) / 1e6 - 92.27) < 0.05, [bar('compute'), bar('X'), bar('W'), bar('Y')].join(' '));
T.check('readout: 2,607 FLOP/B vs 591, compute-bound ×4.4', /2,607 FLOP per byte against the 591/.test(ro()) && /compute-bound, ×4\.4/.test(ro()), ro());
const h0 = w.getBoundingClientRect().height;
// 128 tokens per expert: the weight read dominates
for (let i = 0; i < 5; i++) w.querySelector('[data-knob="tokens"] button[data-dir="-1"]').click();
await T.tick(400);
T.check('128 tokens: 3.80 µs of FLOPs vs 9.35 µs of bytes — memory-bound ×2.5', Math.abs(bar('compute') - 3.80e-6) < 0.01e-6 && /memory-bound, ×2\.5/.test(ro()) && /W streams in/.test(ro()), ro());
T.check('W is 29.4 MB of the 31.3 MB', Math.abs(bar('W') / 1e6 - 29.36) < 0.01 && Math.abs((bar('X') + bar('W') + bar('Y')) / 1e6 - 31.33) < 0.01, bar('W'));
// bf16: input bytes double, peak halves — compute doubles, X and W double, Y (already bf16) holds
w.querySelector('[data-knob="dtype"] button[data-v="bf16"]').click(); await T.tick(350);
T.check('bf16: compute 7.60 µs, W 58.7 MB, Y unchanged, sheet 989 MFLOP / 295 FLOP per byte', Math.abs(bar('compute') - 7.60e-6) < 0.01e-6 && Math.abs(bar('W') / 1e6 - 58.72) < 0.01 && Math.abs(bar('Y') - 128 * 4096 * 2) < 1 && sheet('tensor cores') === '989 MFLOP' && sheet('FLOP per HBM byte') === '295', [bar('compute'), bar('W'), sheet('tensor cores')].join(' '));
// another shape through the select
const sel = w.querySelector('select[data-knob="shape"]'); sel.value = 'head'; sel.dispatchEvent(new Event('change')); await T.tick(350);
T.check('lm head, 128 tokens, bf16: W = 7,168 × 129,280 × 2 B = 1.85 GB', Math.abs(bar('W') - 7168 * 129280 * 2) < 1 && /lm head/.test(ro()), bar('W'));
T.check('knob flips: height reserved (no reflow)', Math.abs(w.getBoundingClientRect().height - h0) < 1, w.getBoundingClientRect().height - h0);
T.check('URL hash carries the state', /g:gemm=/.test(decodeURIComponent(location.hash)) && /head/.test(location.hash) && /128/.test(location.hash), location.hash);
sel.value = 'expert_up'; sel.dispatchEvent(new Event('change'));
w.querySelector('[data-knob="dtype"] button[data-v="fp8"]').click();
for (let i = 0; i < 5; i++) w.querySelector('[data-knob="tokens"] button[data-dir="1"]').click();
await T.tick(400);
T.log('widget height', w.getBoundingClientRect().height);
T.done();
