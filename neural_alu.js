// neural_alu.js
//
// A tiny neural ALU intended to plug into pl0_cpu_sim.js.
//
// Philosophy:
//   - Keep memory, stacks, CALL/RET, and branching deterministic.
//   - Let *arithmetic* be computed by small neural nets (optionally mixed
//     with the exact arithmetic for stability).
//
// The default networks are simple 2-layer MLPs:
//   features(a,b) -> tanh(hidden) -> tanh(output)
//
// For “neural-only but reliable” execution, this file also supports a
// *linear* architecture (no hidden layer). With the right feature map,
// ADD/SUB/MUL can be made exact (within a chosen integer scale) without
// needing a long training run.

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

// Small helper: Gaussian noise via Box-Muller
function randn() {
  const u1 = Math.random() || 1e-12;
  const u2 = Math.random() || 1e-12;
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// Feature maps
// ---------------------------------------------------------------------------
// We keep a small feature map for the MLP case, and a richer one for a
// linear model (which can be made exact for ADD/SUB/MUL).

// a,b already normalized into [-1,1]
function featuresMLP(a, b, scale) {
  // Key trick: include (a*b*scale) which equals (aInt*bInt)/scale in normalized units.
  const mulNorm = clamp(a * b * scale, -1, 1);
  return [a, b, mulNorm, Math.abs(a), Math.abs(b)];
}

// aInt,bInt are integers
function featuresLinear(aInt, bInt, scale) {
  const a = clamp(aInt / scale, -1, 1);
  const b = clamp(bInt / scale, -1, 1);
  const sum = clamp((aInt + bInt) / scale, -1, 1);
  const diff = clamp((aInt - bInt) / scale, -1, 1);
  const mul = clamp((aInt * bInt) / scale, -1, 1);
  // "Real" division (not floored) — close to the desired target.
  const div = (bInt === 0) ? 0 : clamp((aInt / bInt) / scale, -1, 1);
  const absB = clamp(Math.abs(bInt) / scale, 0, 1);
  return [a, b, sum, diff, mul, div, absB];
}

// ---------------------------------------------------------------------------

class MLP {
  constructor(hidden = 16, inputDim = 5, options = {}) {
    this.hidden = hidden;
    this.inputDim = inputDim;
    this.outputActivation = options.outputActivation ?? 'tanh'; // 'tanh' | 'linear'

    // W1: [hidden][inputDim]
    this.W1 = Array.from({ length: hidden }, () => Array.from({ length: inputDim }, () => randn() * 0.1));
    this.b1 = Array.from({ length: hidden }, () => 0);

    // W2: [hidden]
    this.W2 = Array.from({ length: hidden }, () => randn() * 0.1);
    this.b2 = 0;
  }

  // Forward pass returning intermediates (for training)
  forward(x) {
    if (!Array.isArray(x) || x.length !== this.inputDim) {
      throw new Error(`MLP: expected inputDim=${this.inputDim}, got ${x?.length}`);
    }
    const hRaw = new Array(this.hidden);
    const h = new Array(this.hidden);
    for (let i = 0; i < this.hidden; i++) {
      let z = this.b1[i];
      for (let j = 0; j < this.inputDim; j++) {
        z += this.W1[i][j] * x[j];
      }
      hRaw[i] = z;
      h[i] = Math.tanh(z);
    }
    let yRaw = this.b2;
    for (let i = 0; i < this.hidden; i++) yRaw += this.W2[i] * h[i];
    const y = (this.outputActivation === 'linear') ? yRaw : Math.tanh(yRaw);
    return { y, yRaw, h, hRaw };
  }

  predict(x) {
    return this.forward(x).y;
  }

  // One SGD step on a single sample (a,b)->target, all are in [-1,1]
  trainOne(x, target, lr) {
    const { y, yRaw, h, hRaw } = this.forward(x);

    // loss = (y - t)^2
    const dL_dy = 2 * (y - target);
    // y = activation(yRaw)
    const dy_dyRaw = (this.outputActivation === 'linear') ? 1 : (1 - Math.tanh(yRaw) ** 2);
    const dL_dyRaw = dL_dy * dy_dyRaw;

    // Gradients for W2, b2
    for (let i = 0; i < this.hidden; i++) {
      this.W2[i] -= lr * (dL_dyRaw * h[i]);
    }
    this.b2 -= lr * dL_dyRaw;

    // Backprop into hidden
    for (let i = 0; i < this.hidden; i++) {
      const dL_dhi = dL_dyRaw * this.W2[i];
      const dhi_dhRaw = 1 - Math.tanh(hRaw[i]) ** 2;
      const dL_dhRaw = dL_dhi * dhi_dhRaw;

      for (let j = 0; j < this.inputDim; j++) {
        this.W1[i][j] -= lr * (dL_dhRaw * x[j]);
      }
      this.b1[i] -= lr * dL_dhRaw;
    }

    return (y - target) * (y - target);
  }
}

class LinearNet {
  constructor(inputDim = 7) {
    this.inputDim = inputDim;
    this.w = Array.from({ length: inputDim }, () => randn() * 0.05);
    this.b = 0;
  }

  predict(x) {
    if (!Array.isArray(x) || x.length !== this.inputDim) {
      throw new Error(`LinearNet: expected inputDim=${this.inputDim}, got ${x?.length}`);
    }
    let y = this.b;
    for (let i = 0; i < this.inputDim; i++) y += this.w[i] * x[i];
    return y;
  }

  trainOne(x, target, lr) {
    const y = this.predict(x);
    const err = (y - target);
    // loss = err^2
    const g = 2 * err;
    for (let i = 0; i < this.inputDim; i++) {
      this.w[i] -= lr * g * x[i];
    }
    this.b -= lr * g;
    return err * err;
  }
}

class NeuralALU {
  constructor(options = {}) {
    this.hidden = options.hidden ?? 32;
    // int <-> [-1,1] normalization range.
    // Larger scale reduces saturation and makes the linear architecture exact
    // for bigger integer ranges.
    this.scale = options.scale ?? 65536;
    this.architecture = options.architecture ?? 'mlp'; // 'mlp' | 'linear'
    this.mix = options.mix ?? 1.0; // 1.0 = pure neural, 0.0 = pure exact
    this.safetyFallback = options.safetyFallback ?? false;
    this.fallbackAbsError = options.fallbackAbsError ?? 2;

    // Division decoding is more stable with floor() than round().
    // (The manual PDF’s reference uses Math.floor for DIV.)
    this.divDecode = options.divDecode ?? 'floor'; // 'floor' | 'round' | 'trunc'

    // MLP uses 5 features; Linear uses 7 features.
    const outputActivation = options.outputActivation ?? ((this.architecture === 'linear') ? 'linear' : 'tanh');

    if (this.architecture === 'linear') {
      this.addNet = new LinearNet(7);
      this.subNet = new LinearNet(7);
      this.mulNet = new LinearNet(7);
      this.divNet = new LinearNet(7);
      if (options.initAnalytic ?? true) this.initAnalytic();
    } else {
      this.addNet = new MLP(this.hidden, 5, { outputActivation });
      this.subNet = new MLP(this.hidden, 5, { outputActivation });
      this.mulNet = new MLP(this.hidden, 5, { outputActivation });
      this.divNet = new MLP(this.hidden, 5, { outputActivation });
    }
  }

  _encInt(x) { return clamp(x / this.scale, -1, 1); }

  _decodeInt(y, op) {
    const yy = clamp(y, -1, 1) * this.scale;
    if (op === 'DIV') {
      if (this.divDecode === 'trunc') return Math.trunc(yy);
      if (this.divDecode === 'round') return Math.round(yy);
      return Math.floor(yy);
    }
    return Math.round(yy);
  }

  // Compute an arithmetic op. Returns an integer.
  // op: 'ADD' | 'SUB' | 'MUL' | 'DIV'
  computeDetailed(op, aInt, bInt) {
    // Exact reference
    let exact;
    switch (op) {
      case 'ADD': exact = aInt + bInt; break;
      case 'SUB': exact = aInt - bInt; break;
      case 'MUL': exact = aInt * bInt; break;
      case 'DIV':
        exact = (bInt === 0) ? 0 : Math.floor(aInt / bInt);
        break;
      default:
        throw new Error(`NeuralALU: unknown op '${op}'`);
    }

    // Neural prediction
    const a = this._encInt(aInt);
    const b = this._encInt(bInt);
    const x = (this.architecture === 'linear')
      ? featuresLinear(aInt, bInt, this.scale)
      : featuresMLP(a, b, this.scale);
    let y;
    switch (op) {
      case 'ADD': y = this.addNet.predict(x); break;
      case 'SUB': y = this.subNet.predict(x); break;
      case 'MUL': y = this.mulNet.predict(x); break;
      case 'DIV': y = this.divNet.predict(x); break;
      default: y = 0;
    }
    const pred = this._decodeInt(y, op);

    // Mix neural with exact to keep programs runnable while you experiment.
    const mixed = Math.round(exact * (1 - this.mix) + pred * this.mix);

    let result = mixed;
    let usedFallback = false;
    if (this.safetyFallback) {
      if (Math.abs(mixed - exact) > this.fallbackAbsError) {
        result = exact;
        usedFallback = true;
      }
    }

    return { result, exact, pred, mixed, usedFallback };
  }

  compute(op, aInt, bInt) {
    return this.computeDetailed(op, aInt, bInt).result;
  }

  // Quick supervised training so demos work.
  // This trains each op on random pairs within a limited range.
  trainQuick(options = {}) {
    const epochs = options.epochs ?? 10;
    const stepsPerEpoch = options.stepsPerEpoch ?? 4000;
    const lr = options.lr ?? 0.01;
    const operandRange = options.operandRange ?? 256; // base range for ADD/SUB/DIV
    // For MUL, avoid saturating targets by default: a*b should usually fit inside `scale`.
    const mulRange = options.mulRange ?? Math.max(8, Math.floor(Math.sqrt(this.scale) * 0.9));

    const nets = [
      ['ADD', this.addNet],
      ['SUB', this.subNet],
      ['MUL', this.mulNet],
      ['DIV', this.divNet],
    ];

    for (let e = 0; e < epochs; e++) {
      for (const [op, net] of nets) {
        let loss = 0;
        const range = (op === 'MUL') ? mulRange : operandRange;
        for (let s = 0; s < stepsPerEpoch; s++) {
          const aInt = Math.floor((Math.random() * 2 - 1) * range);
          let bInt = Math.floor((Math.random() * 2 - 1) * range);
          if (op === 'DIV') {
            if (bInt === 0) bInt = 1;
          }

          let t;
          switch (op) {
            case 'ADD': t = aInt + bInt; break;
            case 'SUB': t = aInt - bInt; break;
            case 'MUL': t = aInt * bInt; break;
            case 'DIV': t = Math.floor(aInt / bInt); break;
            default: t = 0;
          }

          const a = this._encInt(aInt);
          const b = this._encInt(bInt);
          const x = (this.architecture === 'linear')
            ? featuresLinear(aInt, bInt, this.scale)
            : featuresMLP(a, b, this.scale);
          const target = this._encInt(t);
          loss += net.trainOne(x, target, lr);
        }
        // Lightly anneal learning rate per epoch
        // (kept simple; callers can override lr if desired)
      }
    }
  }

  // Analytic initialization for the *linear* architecture.
  // This makes ADD/SUB/MUL exact (within scale, i.e. without feature clamping).
  initAnalytic() {
    if (this.architecture !== 'linear') return;
    // featuresLinear: [a, b, sum, diff, mul, div, absB]
    // ADD: y = sum
    this.addNet.w.fill(0); this.addNet.w[2] = 1; this.addNet.b = 0;
    // SUB: y = diff
    this.subNet.w.fill(0); this.subNet.w[3] = 1; this.subNet.b = 0;
    // MUL: y = mul
    this.mulNet.w.fill(0); this.mulNet.w[4] = 1; this.mulNet.b = 0;
    // DIV: start with "real division" feature, then let training refine.
    this.divNet.w.fill(0); this.divNet.w[5] = 1; this.divNet.b = 0;
  }
}

module.exports = { NeuralALU };
