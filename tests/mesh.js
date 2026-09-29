// @page studies/03-roofline.html
// the mesh axes: under EP our row (7 NVLink node-mates, 7 IB peers with our
// local rank, 49 more reached through them); under FSDP our GPU's peers are its 7 node-mates
// (NVLink) and the 255 GPUs with its local rank (IB), every other GPU holding
// a shard it reaches through them; under EFSDP only our column (the expert
// slice's 31 other copies, one per EP group, none on our node) and nothing on
// NVLink; the flip tweens and lands on exact fills; the tip names the GPU
// under the pointer; state rides the hash
const { role, OURS, ROWS, NODES, GPUS } = await import('/src/mesh.js');
const w = document.getElementById('mesh');
const q = (s) => w.querySelector(s), qa = (s) => [...w.querySelectorAll(s)];
const btn = (v) => q(`[data-knob=view] [data-v="${v}"]`);
const count = () => Object.fromEntries(['ours', 'ib', 'nv', 'grp', 'out'].map((r) => [r, qa(`rect[data-role=${r}]`).length]));
const fmt = (c) => JSON.stringify(c);

T.check('resting: FSDP', btn('fsdp').classList.contains('on') && !btn('efsdp').classList.contains('on'), '');
T.check('FSDP readout fits', q('[data-readout]').getComputedTextLength() < 738, q('[data-readout]').getComputedTextLength());
let c = count();
T.check('FSDP: 1 ours · 255 IB · 7 NVLink · 1,785 via a peer', fmt(c) === fmt({ ours: 1, ib: 255, nv: 7, grp: 1785, out: 0 }), fmt(c));
T.check('FSDP: NVLink peers are exactly our node', qa('rect[data-role=nv]').every((r) => Math.abs(+r.getAttribute('y') - +q('rect[data-role=ours]').getAttribute('y')) < 0.5), '');
T.check('our node box is outlined', qa('rect[data-node=ours]').length === 1, '');
const roles = [];
for (let row = 0; row < ROWS; row++) for (let n = 0; n < NODES; n++) for (let k = 0; k < GPUS; k++) roles.push(role('efsdp', row, n, k));
T.check('role(): EFSDP peers never share our node', roles.every((r, i) => r !== 'ib' || Math.floor(i / (NODES * GPUS)) !== OURS.row), '');

// tip: hover the first IB peer in the row below ours (node 0, our local rank)
const cell = qa('rect[data-role=ib]').find((r) => +r.getAttribute('y') > +q('rect[data-role=ours]').getAttribute('y'));
const hov = (r) => { const b = r.getBoundingClientRect(); q('svg').parentElement.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: b.x + b.width / 2, clientY: b.y + b.height / 2 })); };
hov(cell);
await T.tick(30);
const tip = () => w.querySelector('.dsv3-tip')?.textContent ?? '';
const gpu = (OURS.row + 1) * NODES * GPUS + OURS.rank;
T.check('tip: GPU, node, EP group, IB peer', tip().includes(`GPU ${gpu.toLocaleString('en-US')} · node ${(OURS.row + 1) * NODES}, local rank ${OURS.rank}`) && tip().includes(`EP group ${OURS.row + 1}, EP rank ${OURS.rank}`) && /IB peer for 1\/8/.test(tip()), tip());

btn('ep').click();
await T.tick(400);
c = count();
T.check('EP: 1 ours · 7 IB · 7 NVLink · 49 via a peer, all in our row', fmt(c) === fmt({ ours: 1, ib: 7, nv: 7, grp: 49, out: 1984 }), fmt(c));
T.check('EP readout fits', q('[data-readout]').getComputedTextLength() < 738, q('[data-readout]').getComputedTextLength());
T.check('EP: every peer is in our row', qa('rect[data-role=ib], rect[data-role=grp]').every((r) => +r.getAttribute('y') === +q('rect[data-role=ours]').getAttribute('y')), '');
btn('fsdp').click();
await T.tick(400);
btn('efsdp').click();
await T.tick(60);
T.check('mid-tween: some fill is between the two roles', qa('rect[data-role=out]').some((r) => r.getAttribute('fill') !== '#ffffff'), '');
await T.tick(400);
c = count();
T.check('EFSDP: 1 ours · 31 IB · 0 NVLink · 2,016 elsewhere', fmt(c) === fmt({ ours: 1, ib: 31, nv: 0, grp: 0, out: 2016 }), fmt(c));
T.check('EFSDP: fills land exactly', qa('rect[data-role=out]').every((r) => r.getAttribute('fill') === '#ffffff') && qa('rect[data-role=ib]').every((r) => r.getAttribute('fill') === '#1c1c1a'), '');
T.check('EFSDP readout fits', q('[data-readout]').getComputedTextLength() < 738, q('[data-readout]').getComputedTextLength());
T.check('EFSDP readout: all IB', /IB: all of it \(19\.8 GB a move\)/.test(q('[data-readout]').textContent), q('[data-readout]').textContent);
hov(qa('rect[data-role=out]')[0]);
await T.tick(30);
T.check('tip: a non-peer holds another slice', /EP rank 0's expert slice: nothing to exchange with us/.test(tip()), tip());
T.check('state in the hash', /m%3Amesh=.*efsdp/.test(location.hash), location.hash);
T.done();
