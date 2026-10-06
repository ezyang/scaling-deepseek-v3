// @page studies/coda.html
// The CODA study: every launch of the DeepSeek EP64 profile's layer cycle is
// modeled (240, one spec each, with the tensors it reads and writes), the
// non-GEMM sums reproduce 03's kernel table to the microsecond, the GEMM
// speed-of-light cells reproduce 03's, the dependency graph closes in every
// pass (every tensor read has a producer, every tensor written a reader) and
// every CODA fold reaches its host along the wires; the figure draws every
// launch as a node and every tensor as a wire (forks dotted, cross-pass
// tensors stubbed), the CODA flip keeps anchors (nodes do not move; folded
// nodes ghost in place and name their fate) and rides the hash, tips name the
// kernel and its operands, and the prose numbers are live.
const M = await import('/src/kerntrace.js');
const KD = await import('/src/kerndata.js');
const KN = await import('/src/kernels.js');
const near = (a, b, e = 1e-9) => Math.abs(a - b) < e;
const P = M.priced(), TT = M.tally();
T.check('240 launches, all specified', KD.LAUNCHES.length === 240 && P.length === 240 && P.every((e) => e.lab && e.kind), P.length);
// 03's "everything else (not a matmul)" row: raw sums of the non-GEMM, non-attention compute-stream launches
const other = (p) => P.filter((e) => e.pass === p && !e.comm && e.kind !== 'gemm' && e.kind !== 'attn').reduce((t, e) => t + e.raw, 0);
T.check('non-GEMM sums = 03 (2,713.25 / 5,307.25 / 577.75 µs)', near(other('F'), 2713.25, 1e-6) && near(other('B'), 5307.25, 1e-6) && near(other('W'), 577.75, 1e-6), `${other('F')} ${other('B')} ${other('W')}`);
// 03's per-kernel SOL cells (µs): q down 61.9 · kv down 23.2 · q up 212 · kv up 94.4 · attention 907 · o 660 · router 19.8 · shared 165/82.6 · routed 1,320/660 · attention bwd 2,360
const sol = (i) => P[i].sol * 1e6;
T.check('GEMM speed of light = 03\'s cells', [[3, 61.9], [6, 23.2], [15, 212], [18, 94.4], [23, 907], [27, 660], [30, 19.8], [103, 165], [118, 82.6], [109, 1320], [114, 660], [174, 2360]].every(([i, v]) => Math.abs(sol(i) - v) / v < 0.004), [3, 6, 15, 18, 23, 27, 30, 103, 118, 109, 114, 174].map((i) => sol(i).toFixed(1)).join(' '));
T.check('routed rows normalized like 03 (32,512 F / 32,128 B)', near(KD.ROUTED_ROWS.F.reduce((a, b) => a + b) / 4, 32512) && near(KD.ROUTED_ROWS.B.reduce((a, b) => a + b) / 4, 32128), '');
T.check('every GEMM compute-bound today (the bf16 router GEMMs sit on the ridge)', P.filter((e) => e.kind === 'gemm' && e.dt === 'fp8').every((e) => e.flop >= e.hbm) && P.filter((e) => e.kind === 'gemm' && e.dt === 'bf16').every((e) => e.flop >= e.hbm * 0.97), '');
T.check('after CODA every host hides its epilogue but the Q-down input gradient', TT.F.gemm.over.length === 0 && TT.W.gemm.over.length === 0 && TT.B.gemm.over.join() === '214' && P[214].hbmAfter < 2 * P[214].flop, TT.B.gemm.over.join());
// epilogue execution (the field guide's meter): at DeepSeek's K no host is exposed if overlapped; serialized, the SwiGLU epilogues add less than the kernels they retire, at either rate
const hosts = P.filter((e) => e.epx);
T.check('epilogue execution: 19 hosts, 5 transcendental, none exposed if overlapped, serialized add < retired SwiGLU kernels', hosts.length === 19 && hosts.filter((e) => e.epx.cls === 'transcendental').map((e) => e.i).join() === '30,68,75,103,109' && hosts.every((e) => e.epx.epiOver.precise === 0) && hosts.every((e) => e.epx.cls === 'transcendental' || e.epx.epiAdd.precise === 0)
  && TT.F.gemm.epiAddPrecise + TT.B.gemm.epiAddPrecise < [111, 116, 76, 71, 70].reduce((t, i) => t + P[i].us, 0) * 1e-6 && TT.F.gemm.epiAddFast < TT.F.gemm.epiAddPrecise / 5, `${hosts.map((e) => e.i).join(' ')} add ${((TT.F.gemm.epiAddPrecise + TT.B.gemm.epiAddPrecise) * 1e6).toFixed(0)} µs`);
T.check('bytes shrink under CODA, in every pass', ['F', 'B', 'W'].every((p) => TT[p].other.bytesAfter < TT[p].other.bytes), '');
T.check('inferred shapes are marked', P.filter((e) => e.inf).length > 10 && P.filter((e) => e.inf).every((e) => !e.kind || e.kind !== 'gemm'), P.filter((e) => e.inf).length);

// the dependency graph: closes in every pass (graph() throws otherwise), every launch a node, the stash falls out of it
const G = Object.fromEntries(['F', 'B', 'W'].map((p) => [p, M.graph(p)]));
T.check('graph closes: 89 + 134 + 17 = 240 launches, 113 / 155 / 8 tensor edges', G.F.nodes.length + G.B.nodes.length + G.W.nodes.length === 240 && G.F.edges.length === 113 && G.B.edges.length === 155 && G.W.edges.length === 8, `${G.F.edges.length} ${G.B.edges.length} ${G.W.edges.length}`);
const stash = new Set(G.B.nodes.flatMap((e) => e.sources.filter((s) => s.from === 'F').map((s) => s.t)));
T.check('the forward → backward stash falls out of the wiring (latents, E5M6 attention out, lse, rstds, the FP8 gate/up)', ['q latent', 'kv latent ‖ k_rope', 'attn out e5m6', 'lse', 'r₁', 'r_q', 'r_kv', 'r₂', 'gu_r fp8 (stash)', 'gu_sh fp8 (stash)', 'h₁', 'o′'].every((t) => stash.has(t)) && !stash.has('n₂ fp8'), [...stash].join(' · '));
T.check('every weight-gradient operand is a backward (or forward) output', G.W.nodes.every((e) => e.sources.every((s) => s.from === 'B' || s.from === 'F')) && G.W.nodes.filter((e) => e.kind === 'gemm').every((e) => e.in.length === 2), '');
T.check('the SwiGLU-input stash: forward casts 119/120 feed backward dequantizes 72/65', P[119].out.includes('gu_r fp8 (stash)') && P[72].in.includes('gu_r fp8 (stash)') && P[120].out.includes('gu_sh fp8 (stash)') && P[65].in.includes('gu_sh fp8 (stash)'), '');
T.check('every numeric CODA host is in the folded launch\'s pass and wired to it', P.filter((e) => typeof e.host === 'number').every((e) => P[e.host].pass === e.pass || (e.comm)), '');

// the figure
const w = document.getElementById('kern');
const q = (s) => w.querySelector(s), qa = (s) => [...w.querySelectorAll(s)];
const launchesDrawn = () => qa('g[data-node]').flatMap((g) => g.dataset.node.split('+')).length;
const NF = KN.nodesFor('F');
T.check('forward: every launch of the pass (DeepEP included) is a node', launchesDrawn() === G.F.nodes.length && qa('g[data-node]').length === NF.nodes.length, launchesDrawn());
const drawnF = NF.edges.filter((e) => e.kind !== 'none' && e.kind !== 'xcol');
T.check('forward: every activation edge is a wire with an arrowhead; side tensors (scales, handles, maps, positions, Wᵀ) draw none of their own', qa('path[data-edge]').length <= drawnF.length && qa('path[data-edge]').length >= drawnF.length - 4 && qa('path[data-edge]').every((p) => p.getAttribute('marker-end')) && NF.edges.some((e) => e.kind === 'none') && !qa('path[data-edge]:not([data-wire="rejoin"])').some((p) => /scalesᵀ$/.test(p.dataset.t)), `${qa('path[data-edge]').length} of ${NF.edges.length}`);
T.check('the main path threads the spine: quantize → Q down, Q down → q norm, SwiGLU → FC2 enter from the top; pearls (scale transposes, Wᵀ, arange) sit indented and tapped', qa('path[data-wire="spine"]').length >= 25 && q('path[data-edge="1>3"][data-wire="spine"]') && q('path[data-edge="3>9"][data-wire="spine"]') && q('path[data-edge="111>114"][data-wire="spine"]') && q('g[data-node="2"][data-side="1"]') && q('g[data-node="107"][data-side="1"]') && +q('g[data-node="2"] rect').getAttribute('x') > +q('g[data-node="1"] rect').getAttribute('x') && q('path[data-edge="1>2"][data-wire="tap"]') && q('path[data-edge="2>3"][data-wire="rejoin"]'), `${qa('path[data-wire="spine"]').length} spine, ${qa('path[data-wire="rail"]').length} rails, ${qa('path[data-wire="tap"]').length} taps, ${qa('path[data-wire="rejoin"]').length} rejoins`);
T.check('rails are the genuine bypasses only (k_rope, the MoE input to the dispatch, y_sh to the combine …): at most 16 in the forward', qa('path[data-wire="rail"]').length <= 16 && q('path[data-edge="64>231"][data-wire="rail"]') && q('path[data-edge="118>239"][data-wire="rail"]') && q('path[data-edge="6>12"][data-wire="rail"]'), qa('path[data-wire="rail"]').length);
T.check('across the sub-layers a labeled connector, not a wire: h₁ and o′ leave attention by stubs and arrive at the MoE residual add', !q('path[data-edge="0>29"]') && !q('path[data-edge="28>29"]') && qa('path[data-stub="in"][data-xcol]').length === 2 && qa('path[data-stub="out"][data-xcol]').length === 2 && /h₁/.test(w.querySelector('text.xcon')?.parentNode.textContent) && qa('text.xcon').some((t) => t.textContent === 'o′, h₁'), qa('text.xcon').map((t) => t.textContent).join(' | '));
T.check('indented launches send no wires forward (pearls rejoin; leaves only stub out), and every main-path GEMM stays on the spine', NF.edges.every((e) => !e.p.indent || e.kind === 'rejoin' || e.kind === 'none' || e.kind === 'xcol') && qa('g[data-node] rect.box').every((r) => [56, 272, 508, 724].includes(+r.getAttribute('x'))), '');
T.check('every node title fits its node (measured, no overlap with the output label)', qa('g[data-node]').every((g) => { const ts = [...g.querySelectorAll('text')], r = g.querySelector('rect').getBBox(); if (ts.some((t) => t.getBBox().x < r.x - 0.5 || t.getBBox().x + t.getBBox().width > r.x + r.width + 0.5)) return false; const [a, b] = ts; return !b || +a.getAttribute('y') !== +b.getAttribute('y') || a.getBBox().x + a.getBBox().width <= b.getBBox().x + 1; }), qa('g[data-node]').filter((g) => { const [a, b] = g.querySelectorAll('text'); return b && +a.getAttribute('y') === +b.getAttribute('y') && a.getBBox().x + a.getBBox().width > b.getBBox().x + 1; }).map((g) => g.dataset.node).join(' '));
T.check('forks are dotted: n₁ to both quantizes, attn out to the stash and the cast, X_e, row map', qa('circle[data-dot]').length >= 8 && qa('path[data-edge][data-t="n₁"]').length === 2 && qa('path[data-edge][data-t="attn out"]').length === 2, qa('circle[data-dot]').length);
T.check('stubs: layer inputs grey, the stash for the backward amber', qa('path[data-stub="in"]:not(.stash)').length >= 2 && qa('path[data-stub="out"].stash').length >= 8 && qa('path[data-stub="out"][data-t="attn out e5m6"].stash').length === 1, `${qa('path[data-stub="in"]').length} in, ${qa('path[data-stub="out"]').length} out`);
T.check('wires run right-angled, downhill within a column: the router GEMM sits below the MoE norm, the dispatch below the router glue', +q('g[data-node="30"]').dataset.row > +q('g[data-node="29"]').dataset.row && +q('g[data-node="231"]').dataset.row > +q('g[data-node="41"]').dataset.row && qa('path[data-edge]').every((p) => /^M [\d.]+ [\d.]+( [VH] [\d.]+)+$/.test(p.getAttribute('d'))), '');
T.check('attention left (lanes 0–1), MoE right (2–3): the O proj is in lane 0, the MoE norm in lane 2, the shared expert in lane 3', q('g[data-node="27"]').dataset.lane === '0' && q('g[data-node="29"]').dataset.lane === '2' && q('g[data-node="103"]').dataset.lane === '3', '');
T.check('resting: as traced, no ghosts', !q('[data-knob=mode] [data-v=coda]').classList.contains('on') && qa('g[data-gone="1"]').length === 0, '');
const yOf = () => Object.fromEntries(qa('g[data-node]').map((g) => [g.dataset.node, g.querySelector('rect').getAttribute('y') + ',' + g.querySelector('rect').getAttribute('x')]));
const dOf = () => qa('path[data-edge]').map((p) => p.getAttribute('d')).join('|');
const y0 = yOf(), d0 = dOf();
q('[data-knob=mode] [data-v=coda]').click();
await T.tick(400);
const y1 = yOf();
T.check('CODA flip preserves every anchor and every wire', Object.keys(y0).every((k) => y0[k] === y1[k]) && Object.keys(y0).length === Object.keys(y1).length && dOf() === d0, '');
T.check('CODA: folded nodes ghost in place and name their fate', qa('g[data-gone="1"]').length === TF_ghosts() && /Q down epilogue/.test(q('g[data-node="13"] .fate')?.textContent) && /DeepEP combine/.test(q('g[data-node="1"] .fate')?.textContent), qa('g[data-gone="1"]').length);
function TF_ghosts() { return NF.nodes.filter((n) => n.gone).length; }
T.check('CODA: hosts wear an epilogue tag', qa('.epibox').length >= 6 && q('g[data-node="27"] .epi'), qa('.epibox').length);
T.check('hash state', decodeURIComponent(location.hash).includes('k:kern={"pass":"F","mode":"coda"}'), location.hash);
const foot = q('text[data-foot=floor]').textContent;
T.check('footer: floor → CODA floor', /1\.40 ms → with CODA 0\.54 ms in 41 launches/.test(foot), foot);
// tip: hover the output projection
const r = q('g[data-node="27"] rect'), b = r.getBoundingClientRect();
r.dispatchEvent(new MouseEvent('mousemove', { bubbles: true, clientX: b.x + 40, clientY: b.y + b.height / 2 }));
await T.tick(30);
const tip = w.querySelector('.dsv3-tip')?.textContent ?? '';
T.check('tip: kernel, both meters, reads / writes, the epilogue', /attention output projection/.test(tip) && /4,096 × 16,384 → 7,168/.test(tip) && /compute-bound/.test(tip) && /reads: attn out fp8 \(from #25/.test(tip) && /writes: o \(→ #28/.test(tip) && /CODA epilogue: × c · \+ residual/.test(tip) && /fp8_ptp128c/.test(tip), tip.slice(0, 200));
// the backward pass
q('[data-knob=pass] [data-v=B]').click();
await T.tick(200);
T.check('backward: every launch of the pass is a node', launchesDrawn() === G.B.nodes.length, launchesDrawn());
T.check('backward: the router\'s 25 glue launches collapse to one node', q('g[data-node^="121+"]')?.dataset.node.split('+').length === 25, q('g[data-node^="121+"]')?.dataset.node);
T.check('backward: FlashAttention backward is a box reading the recomputed Q, K, V', q('g[data-node="174"] rect.box') && /FlashAttention backward/.test(q('g[data-node="174"] text').textContent) && qa('path[data-edge$=">174"]').length >= 6, qa('path[data-edge$=">174"]').length);
T.check('backward: d(x₁) leaves the MoE column for the attention column by connectors (the scale backward and the input-norm backward)', qa('path[data-stub="in"][data-xcol][data-t="d(x₁)"]').length === 2 && qa('path[data-stub="out"][data-xcol][data-t="d(x₁)"]').length === 1 && q('g[data-node="149"]').dataset.lane === '3' && q('g[data-node="152"]').dataset.lane === '0', '');
T.check('backward: weight-gradient operands leave by amber stubs', qa('path[data-stub="out"].stash').length >= 14, qa('path[data-stub="out"].stash').length);
// the weight gradients
q('[data-knob=pass] [data-v=W]').click();
await T.tick(200);
T.check('weight gradients: 17 nodes, every operand an amber stub in, every dW a grey stub out', launchesDrawn() === 17 && qa('path[data-stub="in"].stash').length >= 24 && qa('path[data-stub="out"]:not(.stash)').length === 15, `${qa('path[data-stub="in"].stash').length} in`);

// the page: tables and live prose
T.check('per-launch tables: 240 rows', document.querySelectorAll('.tally[data-pass] tr[data-i]').length === 240, '');
T.check('inferred rows marked ~', [...document.querySelectorAll('.tally[data-pass] tr[data-i] td.lab')].filter((td) => td.textContent.startsWith('~')).length === P.filter((e) => e.inf).length, '');
T.check('savings table: layer row', /one layer, one microbatch/.test(document.querySelector('.tally.savings tr.hl')?.textContent), '');
T.check('prose numbers live', document.querySelector('[data-kt="F.other.trace"]').textContent === '2.71 ms' && document.querySelector('[data-kt="F.other.hbm"]').textContent === '1.40 ms' && document.querySelector('[data-kt="B.other.trace"]').textContent === '5.31 ms' && /%$/.test(document.querySelector('[data-kt="step.pct"]').textContent), '');
T.check('AI disclosure present', document.getElementById('fn1')?.textContent.includes('Disclosure'), '');
T.done();
