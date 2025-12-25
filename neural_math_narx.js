// neural_math_narx.js
//
// A tiny "math coprocessor" built from NARX networks, intended to plug into
// pl0_cpu_sim.js as neural execution for unary transcendental-ish ops.
//
// Fixed-point convention (default Q16.16):
//   - CPU registers/memory hold integers
//   - a real value v is represented as vFx = round(v * scale)
//
// Internally, each op maps its fixed-point input and output into [0,1]
// using a per-op bounded range. The NARX predicts in [0,1], then we decode
// back into fixed-point.
//
// Supported ops (string keys):
//   FSIN, FCOS, FTAN, FTANH, FSINH, FCOSH, FLN, FLOG10, FEXP, FSQRT

const NARX = require('./narx');

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

// ------------------------- Fixed-point ranges -------------------------
// These ranges define the normalization from fixed-point <-> [0,1].
// You can widen them, but training will become harder and saturation more likely.
const DEFAULT_OP_SPECS = {
  // Trig: input in radians
  FSIN:   { inLo: -Math.PI, inHi:  Math.PI, outLo: -1,  outHi:  1 },
  FCOS:   { inLo: -Math.PI, inHi:  Math.PI, outLo: -1,  outHi:  1 },
  FTAN:   { inLo: -1.3,     inHi:  1.3,     outLo: -8,  outHi:  8 },
  // Hyperbolic
  FTANH:  { inLo: -3,       inHi:  3,       outLo: -1,  outHi:  1 },
  FSINH:  { inLo: -3,       inHi:  3,       outLo: -8,  outHi:  8 },
  FCOSH:  { inLo: -3,       inHi:  3,       outLo:  0,  outHi:  10 },
  // Logs (domain x>0)
  FLN:    { inLo:  1e-6,    inHi:  256,     outLo: -16, outHi:  16 },
  FLOG10: { inLo:  1e-6,    inHi:  256,     outLo: -16, outHi:  16 },
  // Exp (can grow)
  FEXP:   { inLo: -8,       inHi:  8,       outLo:  0,  outHi:  256 },
  // Sqrt (domain x>=0)
  FSQRT:  { inLo:  0,       inHi:  256,     outLo:  0,  outHi:  16 },
};

function fxToFloat(xFx, scale) {
  return xFx / scale;
}

function floatToFx(x, scale) {
  const MAX_FX = 0x7fffffff;
  const v = Math.round(x * scale);
  return clamp(v, -MAX_FX, MAX_FX);
}

function encodeNormFromRange(x, lo, hi) {
  if (!(hi > lo)) return 0.5;
  return clamp((x - lo) / (hi - lo), 0, 1);
}

function decodeNormToRange(u, lo, hi) {
  const uu = clamp(u, 0, 1);
  return lo + uu * (hi - lo);
}

// Deterministic reference for fixed-point unary math ops.
function refMathFx(op, xFx, scale) {
  const spec = DEFAULT_OP_SPECS[op];
  const x0 = fxToFloat(xFx, scale);
  // Clamp into the modeled input range so deterministic and neural paths match.
  const x = spec ? clamp(x0, spec.inLo, spec.inHi) : x0;
  let y;
  switch (op) {
    case 'FSIN': y = Math.sin(x); break;
    case 'FCOS': y = Math.cos(x); break;
    case 'FTAN': y = clamp(Math.tan(x), -8, 8); break;
    case 'FTANH': y = Math.tanh(x); break;
    case 'FSINH': y = clamp(Math.sinh(x), -8, 8); break;
    case 'FCOSH': y = clamp(Math.cosh(x), 0, 10); break;
    case 'FLN':
      if (x <= 0) return floatToFx(DEFAULT_OP_SPECS.FLN.outLo, scale);
      y = clamp(Math.log(x), -16, 16);
      break;
    case 'FLOG10':
      if (x <= 0) return floatToFx(DEFAULT_OP_SPECS.FLOG10.outLo, scale);
      y = clamp(Math.log10(x), -16, 16);
      break;
    case 'FEXP':
      y = clamp(Math.exp(x), 0, 256);
      break;
    case 'FSQRT':
      y = (x <= 0) ? 0 : Math.sqrt(x);
      y = clamp(y, 0, 16);
      break;
    default:
      throw new Error(`refMathFx: unknown op '${op}'`);
  }
  return floatToFx(y, scale);
}

// ----------------------------- Coprocessor ----------------------------

class NeuralMathNARX {
  constructor(options = {}) {
    this.scale = Number.isFinite(options.scale) ? options.scale : 65536;
    this.inputLag = options.inputLag ?? 3;
    this.outputLag = options.outputLag ?? 2;
    this.hiddenUnits = options.hiddenUnits ?? 24;
    this.mix = options.mix ?? 1.0; // 1.0 = pure neural, 0.0 = pure reference
    this.safetyFallback = options.safetyFallback ?? true;
    // In fixed-point mode, small norm errors can produce visible integer deltas.
    // Default to a strict threshold so programs remain stable unless you relax it.
    this.fallbackAbsError = options.fallbackAbsError ?? 0.001; // in outNorm units

    // Per-op normalization ranges (float domain).
    this.opSpecs = { ...DEFAULT_OP_SPECS, ...(options.opSpecs ?? {}) };

    this.ops = options.ops ?? [
      'FSIN', 'FCOS', 'FTAN',
      'FTANH', 'FSINH', 'FCOSH',
      'FLN', 'FLOG10', 'FEXP', 'FSQRT',
    ];

    this.opSpecs = { ...DEFAULT_OP_SPECS, ...(options.opSpecs ?? {}) };
    for (const op of this.ops) {
      if (!this.opSpecs[op]) throw new Error(`NeuralMathNARX: missing opSpec for '${op}'`);
    }

    this.nets = {};
    this.state = {};
    for (const op of this.ops) {
      if (!this.opSpecs[op]) throw new Error(`NeuralMathNARX: missing opSpec for '${op}'`);
      this.nets[op] = new NARX(this.inputLag, this.outputLag, this.hiddenUnits);
      this.state[op] = {
        inHist: new Array(this.inputLag).fill(0),
        outHist: new Array(this.outputLag).fill(0),
      };
    }

    this.stats = {
      enabled: true,
      calls: Object.fromEntries(this.ops.map((o) => [o, 0])),
      fallbacks: Object.fromEntries(this.ops.map((o) => [o, 0])),
      absErrorSum: Object.fromEntries(this.ops.map((o) => [o, 0])),
    };
  }

  reset() {
    for (const op of this.ops) {
      this.state[op].inHist.fill(0);
      this.state[op].outHist.fill(0);
    }
  }

  // Supervised pre-training using teacher forcing (built into narx.js).
  // Generates a random input sequence in [0,1] (i.e., random values across the
  // op's input range) and targets the reference fixed-point mapping.
  trainQuick(options = {}) {
    const length = options.length ?? 6000;
    const epochs = options.epochs ?? 80;
    const learningRate = options.learningRate ?? 0.02;
    const seedInputs = options.inputs ?? null; // optional array of inNorm

    for (const op of this.ops) {
      const spec = this.opSpecs[op];
      const inSeq = new Array(length);
      const outSeq = new Array(length);
      for (let i = 0; i < length; i++) {
        const u = Array.isArray(seedInputs) ? seedInputs[i % seedInputs.length] : Math.random();
        const inNorm = clamp(u, 0, 1);
        const x = decodeNormToRange(inNorm, spec.inLo, spec.inHi);
        const xFx = floatToFx(x, this.scale);
        const yFx = refMathFx(op, xFx, this.scale);
        const y = fxToFloat(yFx, this.scale);
        const outNorm = encodeNormFromRange(y, spec.outLo, spec.outHi);
        inSeq[i] = inNorm;
        outSeq[i] = outNorm;
      }
      this.nets[op].train(inSeq, outSeq, { epochs, learningRate });
    }
  }

  // Compute unary op on fixed-point integer input; returns fixed-point integer output.
  computeDetailed(op, xFx) {
    if (!(op in this.nets)) throw new Error(`NeuralMathNARX: unsupported op '${op}'`);

    const spec = this.opSpecs[op];

    const s = this.state[op];
    const x = fxToFloat(xFx, this.scale);
    const inNorm = encodeNormFromRange(x, spec.inLo, spec.inHi);
    // Feed the *current* input by shifting it into the input history before forward.
    s.inHist.pop();
    s.inHist.unshift(inNorm);
    const combined = s.inHist.concat(s.outHist);
    const ySig = this.nets[op].forward(combined).output; // already in (0,1)

    const exactFx = refMathFx(op, xFx, this.scale);
    const exact = fxToFloat(exactFx, this.scale);
    const exactNorm = encodeNormFromRange(exact, spec.outLo, spec.outHi);
    const mixedNorm = clamp(exactNorm * (1 - this.mix) + ySig * this.mix, 0, 1);

    let outNorm = mixedNorm;
    let usedFallback = false;
    const absErr = Math.abs(ySig - exactNorm);
    if (this.safetyFallback && absErr > this.fallbackAbsError) {
      outNorm = exactNorm;
      usedFallback = true;
    }

    // Update output history (NARX state)
    s.outHist.pop();
    s.outHist.unshift(clamp(outNorm, 0, 1));

    const resultFloat = decodeNormToRange(outNorm, spec.outLo, spec.outHi);
    const predFloat = decodeNormToRange(ySig, spec.outLo, spec.outHi);
    const result = floatToFx(resultFloat, this.scale);
    const pred = floatToFx(predFloat, this.scale);

    this.stats.calls[op] = (this.stats.calls[op] || 0) + 1;
    this.stats.absErrorSum[op] = (this.stats.absErrorSum[op] || 0) + absErr;
    if (usedFallback) this.stats.fallbacks[op] = (this.stats.fallbacks[op] || 0) + 1;

    return { result, pred, exact: exactFx, usedFallback, outNorm, exactNorm, predNorm: ySig };
  }

  compute(op, xFx) {
    return this.computeDetailed(op, xFx).result;
  }
}

module.exports = { NeuralMathNARX, refMathFx, DEFAULT_OP_SPECS };
