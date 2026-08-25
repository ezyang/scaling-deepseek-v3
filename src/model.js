// Cost model: DeepSeek-V3 architecture + hardware specs + per-op roofline.
// All numbers are editable assumptions; the point is pedagogy, not certification.

export const HARDWARE = {
  h800: {
    label: 'H800 (Hopper, export NVLink)',
    flops: { bf16: 989e12, fp8: 1979e12 },
    // Fraction of peak a well-tuned GEMM achieves. Hopper has no native MX
    // block scaling, so the DeepSeek-style fine-grained FP8 recipe pays for
    // scale handling / higher-precision accumulation out of its 2x.
    gemmEff: { bf16: 0.80, mxfp8: 0.70 },
    attnEff: 0.60, // causal flash attention vs bf16 peak
    hbm: 3.35e12, hbmEff: 0.78, memGB: 80,
    nvl: 200e9, nvlEff: 0.75, domain: 8, // per-direction B/s; H800 NVLink is cut to 400GB/s total
    nic: 50e9, nicEff: 0.80,             // 400 Gb/s IB per GPU
  },
  h100: {
    label: 'H100 SXM',
    flops: { bf16: 989e12, fp8: 1979e12 },
    gemmEff: { bf16: 0.80, mxfp8: 0.70 },
    attnEff: 0.60,
    hbm: 3.35e12, hbmEff: 0.78, memGB: 80,
    nvl: 450e9, nvlEff: 0.75, domain: 8,
    nic: 50e9, nicEff: 0.80,
  },
  gb200: {
    label: 'GB200 NVL72',
    flops: { bf16: 2.5e15, fp8: 5.0e15 },
    gemmEff: { bf16: 0.80, mxfp8: 0.85 }, // Blackwell has native MX support
    attnEff: 0.65,
    hbm: 8e12, hbmEff: 0.78, memGB: 192,
    nvl: 900e9, nvlEff: 0.75, domain: 72, // NVL72: 72 GPUs in one NVLink domain
    nic: 50e9, nicEff: 0.80,              // CX-7, 400 Gb/s
  },
  gb300: {
    label: 'GB300 NVL72',
    flops: { bf16: 2.5e15, fp8: 5.0e15 }, // dense FP8 ~B200; GB300 mostly adds FP4 + HBM capacity
    gemmEff: { bf16: 0.80, mxfp8: 0.85 },
    attnEff: 0.65,
    hbm: 8e12, hbmEff: 0.78, memGB: 288,
    nvl: 900e9, nvlEff: 0.75, domain: 72,
    nic: 100e9, nicEff: 0.80,             // CX-8, 800 Gb/s
  },
};

export const DSV3 = {
  layers: 61, denseLayers: 3,
  hidden: 7168, vocab: 129280,
  heads: 128, qkNope: 128, qkRope: 64, vHead: 128,
  qRank: 1536, kvRank: 512,
  denseInter: 18432,
  moeInter: 2048, routedExperts: 256, topk: 8, sharedExperts: 1,
};

export function peakFlops(hw, dtype) { return dtype === 'mxfp8' ? hw.flops.fp8 : hw.flops.bf16; }
export function gemmRate(hw, dtype) { return peakFlops(hw, dtype) * hw.gemmEff[dtype]; }
export function actBytes(dtype) { return dtype === 'mxfp8' ? 1 : 2; } // gemm-input / wire bytes

// --- per-layer op specs -----------------------------------------------------
// ftok: forward flops per token; wparams: weight params touched per pass;
// abytes: activation bytes per token (fwd); bwdMult: bwd flops multiplier.
export function layerOps(kind, a, seqLen) {
  const h = a.hidden, qk = a.qkNope + a.qkRope;
  const attnDown = h * a.qRank + h * (a.kvRank + a.qkRope);
  const attnUp = a.qRank * a.heads * qk + a.kvRank * a.heads * (a.qkNope + a.vHead);
  const attnOut = a.heads * a.vHead * h;
  const ops = [
    { name: 'norm/rope/residual', cat: 'vector', ftok: 40 * h, wparams: 0, abytes: 12 * h },
    { name: 'mla_down_proj', cat: 'gemm', ftok: 2 * attnDown, wparams: attnDown, abytes: 4 * h },
    { name: 'mla_up_proj', cat: 'gemm', ftok: 2 * attnUp, wparams: attnUp, abytes: 4 * h },
    // causal: average context seqLen/2; bwd of flash is ~2.5x fwd, we use 2x
    { name: 'attn_core', cat: 'attn', ftok: 2 * a.heads * (qk + a.vHead) * seqLen / 2, wparams: 0, abytes: 6 * h },
    { name: 'attn_out_proj', cat: 'gemm', ftok: 2 * attnOut, wparams: attnOut, abytes: 4 * h },
  ];
  if (kind === 'dense') {
    ops.push({ name: 'mlp (gate/up/down)', cat: 'gemm', ftok: 2 * 3 * h * a.denseInter, wparams: 3 * h * a.denseInter, abytes: 8 * h });
  } else {
    const ex = 3 * h * a.moeInter;
    ops.push({ name: 'router', cat: 'vector', ftok: 2 * h * a.routedExperts,
      wparams: (h + 1) * a.routedExperts, abytes: 2 * h });
    ops.push({ name: 'shared_expert', cat: 'gemm', ftok: 2 * ex * a.sharedExperts, wparams: ex * a.sharedExperts, abytes: 6 * h });
    // grouped GEMM over ~topk*tokens/ep tokens per local expert: lower efficiency
    ops.push({ name: 'routed_experts', cat: 'gemm', ftok: 2 * ex * a.topk, wparams: 0, localExpertParams: ex, effMult: 0.75, abytes: 6 * h * a.topk });
  }
  return ops;
}

export const EMBED_OP = (a) => ({ name: 'embedding', cat: 'vector', ftok: 0, wparams: 0, abytes: 4 * a.hidden });
export const HEAD_OPS = (a) => [
  { name: 'lm_head', cat: 'gemm', ftok: 2 * a.hidden * a.vocab, wparams: a.hidden * a.vocab, abytes: 2 * a.hidden + 2 * a.vocab },
  { name: 'softmax/loss', cat: 'vector', ftok: 8 * a.vocab, wparams: 0, abytes: 6 * a.vocab, bwdMult: 0 },
];

// --- parallelism-aware sizes ------------------------------------------------
export function stageLayerKinds(a, pp, stage) {
  const base = Math.floor(a.layers / pp), rem = a.layers % pp;
  const start = stage * base + Math.min(stage, rem);
  const n = base + (stage < rem ? 1 : 0);
  return Array.from({ length: n }, (_, i) => ({
    index: start + i,
    kind: start + i < a.denseLayers ? 'dense' : 'moe',
  }));
}

export function attnLayerParams(a) {
  const qk = a.qkNope + a.qkRope;
  return h(a) * a.qRank + a.qRank * a.heads * qk + h(a) * (a.kvRank + a.qkRope)
    + a.kvRank * a.heads * (a.qkNope + a.vHead) + a.heads * a.vHead * h(a);
}
export function layerNormParams(a) { return 2 * h(a) + a.qRank + a.kvRank; }
const h = (a) => a.hidden;

// Non-expert params on a stage (FSDP-sharded over full dp) and expert params
// per rank (sharded ep-ways, FSDP over the remaining dp/ep replicas).
export function stageParams(a, pp, ep, stage) {
  let dense = 0, expert = 0;
  for (const l of stageLayerKinds(a, pp, stage)) {
    dense += attnLayerParams(a) + layerNormParams(a);
    if (l.kind === 'dense') dense += 3 * a.hidden * a.denseInter;
    else {
      dense += (a.hidden + 1) * a.routedExperts + a.sharedExperts * 3 * a.hidden * a.moeInter;
      expert += (a.routedExperts / ep) * 3 * a.hidden * a.moeInter;
    }
  }
  if (stage === 0) dense += a.hidden * a.vocab;
  if (stage === pp - 1) dense += a.hidden * a.vocab + a.hidden; // final RMSNorm + head
  return { dense, expert };
}

// Model flops per token (fwd+bwd = 3x fwd), for MFU accounting.
export function modelFlopsPerToken(a, seqLen) {
  let f = 0;
  for (let i = 0; i < a.layers; i++)
    for (const op of layerOps(i < a.denseLayers ? 'dense' : 'moe', a, seqLen)) f += op.ftok;
  for (const op of HEAD_OPS(a)) f += op.ftok;
  return 3 * f;
}

// --- op timing (µs) ----------------------------------------------------------
export function opTimeUs(op, tokens, cfg, hw, phase) {
  const mult = phase === 'B' ? (op.bwdMult ?? 2) : 1;
  if (mult === 0) return 0;
  const dtype = cfg.dtype;
  const flops = op.ftok * tokens * mult;
  const rate = op.cat === 'attn' ? hw.flops.bf16 * hw.attnEff
    : op.cat === 'gemm' ? gemmRate(hw, dtype) * (op.effMult ?? 1)
      : hw.flops.bf16; // vector flops rarely bind; memory term below dominates
  const wbytes = (op.wparams + (op.localExpertParams ?? 0) * DSV3.routedExperts / cfg.ep) * actBytes(dtype);
  const membytes = wbytes + op.abytes * tokens * 2 * mult; // activations move as bf16
  return Math.max(flops / rate, membytes / (hw.hbm * hw.hbmEff)) * 1e6;
}

// --- communication timing (µs) ----------------------------------------------
// EP all-to-all for one MoE layer, one direction. Tokens fan out to `topk`
// experts; cross-node copies are deduplicated per node and capped by
// node-limited routing (DeepSeek caps each token at 4 nodes).
export function a2aBytesPerGpu(tokens, bytesPerElem, cfg, hw) {
  const a = DSV3;
  const epNodes = Math.ceil(cfg.ep / hw.domain);
  const intra = tokens * a.topk * a.hidden * bytesPerElem;
  if (epNodes <= 1) return { intra, cross: 0, epNodes };
  // expected nodes a token's copies cross, minus the share landing on its own node
  const lim = Math.min(cfg.nodeLimit, epNodes);
  const cross = tokens * (lim - lim / epNodes) * a.hidden * bytesPerElem;
  return { intra, cross, epNodes };
}

export function a2aTimeUs(tokens, bytesPerElem, cfg, hw) {
  const { intra, cross, epNodes } = a2aBytesPerGpu(tokens, bytesPerElem, cfg, hw);
  if (epNodes <= 1) return intra / (hw.nvl * hw.nvlEff) * 1e6;
  // IB and NVLink phases pipeline; the slower one binds.
  return Math.max(cross / (hw.nic * hw.nicEff), intra / (hw.nvl * hw.nvlEff)) * 1e6;
}

export function p2pTimeUs(tokens, cfg, hw) {
  const bytes = tokens * DSV3.hidden * 2; // activations in bf16
  const bw = cfg.pp <= 1 ? Infinity : (cfg.dp >= hw.domain ? hw.nic * hw.nicEff : hw.nvl * hw.nvlEff);
  return bytes / bw * 1e6;
}

// All-gather / reduce-scatter time for `params` over a group of size g (ring).
export function collTimeUs(params, bytesPerElem, g, hw) {
  if (g <= 1) return 0;
  return params * bytesPerElem * (g - 1) / g / (hw.nic * hw.nicEff) * 1e6;
}

// --- formatting helpers -------------------------------------------------------
export function fmtUs(us) {
  if (us >= 1e6) return (us / 1e6).toFixed(2) + ' s';
  if (us >= 1e3) return (us / 1e3).toFixed(2) + ' ms';
  return us.toFixed(1) + ' µs';
}
export function fmtNum(x) {
  for (const [v, s] of [[1e12, 'T'], [1e9, 'G'], [1e6, 'M'], [1e3, 'K']])
    if (Math.abs(x) >= v) return (x / v).toPrecision(3) + s;
  return String(Math.round(x));
}

export function mulberry32(seed) {
  let t = seed >>> 0;
  return () => {
    t = (t + 0x6D2B79F5) >>> 0;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r = (r + Math.imul(r ^ (r >>> 7), 61 | r)) ^ r;
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}
