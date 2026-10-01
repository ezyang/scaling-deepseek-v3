// @page studies/03-roofline.html
// the mesh axes: under EP our row (7 NVLink node-mates, 7 IB peers with our
// local rank, 49 more reached through them); under FSDP our GPU's peers are its 7 node-mates
// (NVLink) and the 255 GPUs with its local rank (IB), every other GPU holding
// a shard it reaches through them; under EFSDP only our column (the expert
// slice's 31 other copies, one per EP group, none on our node) and nothing on
// NVLink; the flip tweens and lands on exact fills; hovering a GPU draws its
// route to ours, keyed at the legend's right end (FSDP: IB to the
// node-mate with its local rank, then NVLink; EP dispatch: the reverse way
// out); state rides the hash. Two instances: #mesh-dp (FSDP
// alone, no buttons) and #mesh-ep (EFSDP by default, flips to EP)
const { role, route, OURS, ROWS, NODES, GPUS } = await import('/src/mesh.js');
let w = document.getElementById('mesh-dp');
const q = (s) => w.querySelector(s), qa = (s) => [...w.querySelectorAll(s)];
const btn = (v) => q(`[data-knob=view] [data-v="${v}"]`);
const count = () => Object.fromEntries(['ours', 'ib', 'nv', 'grp', 'out'].map((r) => [r, qa(`rect[data-role=${r}]`).length]));
const fmt = (c) => JSON.stringify(c);

T.check('FSDP instance: no buttons, no outsider swatch', qa('button').length === 0 && !/not in the group/.test(q('svg').textContent), qa('button').length);
const keyClear = () => { const k = q('[data-key]').getBBox(); return k.x + k.width <= 738 && qa('svg > text.dims').every((t) => { const b = t.getBBox(); return b.y + b.height < k.y || b.x + b.width < k.x - 6; }); };
T.check('route key clears the legend', keyClear(), q('[data-key]').getBBox().x);
T.check('FSDP readout fits', q('[data-readout]').getComputedTextLength() < 738, q('[data-readout]').getComputedTextLength());
let c = count();
T.check('FSDP: 1 ours · 255 IB · 7 NVLink · 1,785 via a peer', fmt(c) === fmt({ ours: 1, ib: 255, nv: 7, grp: 1785, out: 0 }), fmt(c));
T.check('FSDP: NVLink peers are exactly our node', qa('rect[data-role=nv]').every((r) => Math.abs(+r.getAttribute('y') - +q('rect[data-role=ours]').getAttribute('y')) < 0.5), '');
T.check('our node box is outlined', qa('rect[data-node=ours]').length === 1, '');
const roles = [];
for (let row = 0; row < ROWS; row++) for (let n = 0; n < NODES; n++) for (let k = 0; k < GPUS; k++) roles.push(role('efsdp', row, n, k));
T.check('role(): EFSDP peers never share our node', roles.every((r, i) => r !== 'ib' || Math.floor(i / (NODES * GPUS)) !== OURS.row), '');

// hover the first IB peer in the row below ours (node 0, our local rank)
const cell = qa('rect[data-role=ib]').find((r) => +r.getAttribute('y') > +q('rect[data-role=ours]').getAttribute('y'));
const hov = (r) => { const b = r.getBoundingClientRect(); q('svg').parentElement.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: b.x + b.width / 2, clientY: b.y + b.height / 2 })); };
hov(cell);
await T.tick(30);
T.check('rows labeled EP group N, ours marked', /EP group 12.*\(ours\)/.test(q('svg').textContent) && !/(^|[^P] )group \d/.test(q('svg').textContent), '');
// route: an IB peer's is one IB hop into us
const hops = () => qa('path[data-hop]').map((p) => p.dataset.hop).join(',');
// endpoints: the IB curve is M x,y C x,y x,y x,y; the NVLink bracket is M x,edge V y H x V y (it lands on the edge it left)
const ends = (p) => { const n = p.getAttribute('d').match(/-?[\d.]+/g).map(Number); return p.dataset.hop === 'ib' ? [n.slice(0, 2), n.slice(6, 8)] : [n.slice(0, 2), [n[3], n[1]]]; };
const near = ([x, y], r) => { const b = r.getBoundingClientRect(), s = q('svg').getBoundingClientRect(); return Math.hypot(x - (b.x - s.x + b.width / 2), y - (b.y - s.y + b.height / 2)) < 12; };
T.check('route: IB peer → one IB hop', hops() === 'ib', hops());
// a GPU in another node at another local rank: IB to our node-mate at its rank, then NVLink to us
const far = qa('rect[data-role=grp]')[0];   // row 0, node 0, rank 0
hov(far);
await T.tick(30);
const [ib, nv] = qa('path[data-hop]'), mate = qa('rect[data-role=nv]').find((r) => +r.getAttribute('x') === +far.getAttribute('x') + (OURS.node * 79));
T.check('route: via a peer → IB then NVLink', hops() === 'ib,nv', hops());
T.check('route: IB arrives from above, so the NVLink bracket runs below our row', +ends(nv)[0][1] > +q('rect[data-role=ours]').getAttribute('y'), ends(nv)[0][1]);
T.check('route: IB from the hovered GPU to our node-mate at its local rank, NVLink on to us', near(ends(ib)[0], far) && near(ends(ib)[1], mate) && near(ends(nv)[0], mate) && near(ends(nv)[1], q('rect[data-role=ours]')), JSON.stringify([ends(ib), ends(nv)]));
q('svg').parentElement.dispatchEvent(new MouseEvent('mouseleave'));
await T.tick(30);
T.check('no tip card', !w.querySelector('.dsv3-tip'), '');
T.check('route clears on leave', hops() === '', hops());
T.check('route(): EP dispatch goes IB out to our local rank on its node, then NVLink', JSON.stringify(route('ep', OURS.row, 0, 0)) === JSON.stringify([[[12, 5, 3], [12, 0, 3], 'ib'], [[12, 0, 3], [12, 0, 0], 'nv']]), JSON.stringify(route('ep', OURS.row, 0, 0)));
T.check('route(): nothing for outsiders or us', route('efsdp', 0, 0, 0).length === 0 && route('fsdp', OURS.row, OURS.node, OURS.rank).length === 0, '');

w = document.getElementById('mesh-ep');
T.check('route key clears the legend (expert)', keyClear(), q('[data-key]').getBBox().x);
T.check('expert instance rests on EFSDP', btn('efsdp').classList.contains('on') && !btn('ep').classList.contains('on') && !btn('fsdp'), '');
T.check('expert instance: EFSDP 1 ours · 31 IB · 2,016 elsewhere', fmt(count()) === fmt({ ours: 1, ib: 31, nv: 0, grp: 0, out: 2016 }), fmt(count()));
btn('ep').click();
await T.tick(400);
c = count();
T.check('EP: 1 ours · 7 IB · 7 NVLink · 49 via a peer, all in our row', fmt(c) === fmt({ ours: 1, ib: 7, nv: 7, grp: 49, out: 1984 }), fmt(c));
T.check('EP readout fits', q('[data-readout]').getComputedTextLength() < 738, q('[data-readout]').getComputedTextLength());
T.check('EP: every peer is in our row', qa('rect[data-role=ib], rect[data-role=grp]').every((r) => +r.getAttribute('y') === +q('rect[data-role=ours]').getAttribute('y')), '');
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
T.check('route: an outsider has none', hops() === '', hops());
hov(qa('rect[data-role=ib]')[0]);
await T.tick(30);
T.check('route: EFSDP copy → one IB hop', hops() === 'ib', hops());
T.check('the default view leaves the hash clean', !/mesh-ep/.test(location.hash), location.hash);
btn('ep').click(); await T.tick(30);
T.check('state in the hash', /m%3Amesh-ep=.*%22ep%22/.test(location.hash), location.hash);
T.done();
