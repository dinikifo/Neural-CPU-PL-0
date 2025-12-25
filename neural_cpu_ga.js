// neural_cpu_ga.js
//
// RISC-like CPU + feedforward neural core + NARX-based "complex" instructions + GA
//
// Usage:
//   1) Place this file next to narx.js (provided separately)
//   2) Run: node neural_cpu_ga.js

const NARX = require("./narx");

// -----------------------------------------------------------------------------
// Instruction set
// -----------------------------------------------------------------------------

const OPCODES = {
  ADD: 0,
  SUB: 1,
  ADDI: 2,
  AND: 3,
  OR: 4,
  XOR: 5,
  LD: 6,
  ST: 7,
  BEQ: 8,
  JMP: 9,
  // --- NARX-based complex ops (unary functions) ---
  // Input convention for all: inputByte = R(rs1) in [0..255], inputNorm = inputByte/255.
  // Output convention for all: NARX returns outNorm in [0..1], stored to rd as outByte = round(outNorm*255).
  FSIN: 10,    // sin(angle), angle mapped from inputNorm -> [-pi, pi]
  FCOS: 11,    // cos(angle), angle mapped from inputNorm -> [-pi, pi]
  FTAN: 12,    // tan(angle), angle mapped from inputNorm -> [-pi/2+eps, pi/2-eps], scaled to [-1,1]
  FTANH: 13,   // tanh(x), x mapped from inputNorm -> [-3, 3]
  FSINH: 14,   // sinh(x), x mapped from inputNorm -> [-2, 2], scaled to [-1,1]
  FCOSH: 15,   // cosh(x), x mapped from inputNorm -> [-2, 2], normalized to [0,1]
  FLN: 16,     // ln(1 + inputByte), normalized by ln(256)
  FLOG10: 17,  // log10(1 + inputByte), normalized by log10(256)
  FEXP: 18,    // exp(x)-1, x mapped from inputNorm -> [0, ln(256)] so output is ~[0..255]
  FSQRT: 19,   // sqrt(inputNorm)

  HALT: 20,
};

const NUM_REGS = 4;
const MEM_SIZE = 16;

// -----------------------------------------------------------------------------
// Reference CPU (symbolic ground truth)
// -----------------------------------------------------------------------------

function makeInitialState() {
  return {
    pc: 0,
    regs: [0, 0, 0, 0],
    halted: false,
  };
}

function u8(x) {
  return x & 0xff;
}

function refStepCPU(state, memory, instr) {
  if (state.halted) return;

  const { opcode, rd, rs1, rs2, imm } = instr;
  let pcNext = (state.pc + 1) & 0xff;

  const regs = state.regs;
  const R = (i) => (i === 0 ? 0 : regs[i] & 0xff);

  switch (opcode) {
    case OPCODES.ADD: {
      if (rd !== 0) regs[rd] = u8(R(rs1) + R(rs2));
      break;
    }
    case OPCODES.SUB: {
      if (rd !== 0) regs[rd] = u8(R(rs1) - R(rs2));
      break;
    }
    case OPCODES.ADDI: {
      if (rd !== 0) regs[rd] = u8(R(rs1) + imm);
      break;
    }
    case OPCODES.AND: {
      if (rd !== 0) regs[rd] = u8(R(rs1) & R(rs2));
      break;
    }
    case OPCODES.OR: {
      if (rd !== 0) regs[rd] = u8(R(rs1) | R(rs2));
      break;
    }
    case OPCODES.XOR: {
      if (rd !== 0) regs[rd] = u8(R(rs1) ^ R(rs2));
      break;
    }
    case OPCODES.LD: {
      const addr = u8(R(rs1) + imm);
      if (rd !== 0) regs[rd] = memory[addr % MEM_SIZE];
      break;
    }
    case OPCODES.ST: {
      const addr = u8(R(rs1) + imm);
      memory[addr % MEM_SIZE] = R(rd);
      break;
    }
    case OPCODES.BEQ: {
      if (R(rs1) === R(rs2)) {
        pcNext = u8(state.pc + imm);
      }
      break;
    }
    case OPCODES.JMP: {
      pcNext = u8(imm);
      break;
    }
    case OPCODES.FSIN: {
      const angleByte = R(rs1);
      const angle = (angleByte / 255) * 2 * Math.PI - Math.PI;
      const s = Math.sin(angle);
      const encoded = Math.round(((s + 1) / 2) * 255);
      if (rd !== 0) regs[rd] = u8(encoded);
      break;
    }
    case OPCODES.FCOS: {
      const angleByte = R(rs1);
      const angle = (angleByte / 255) * 2 * Math.PI - Math.PI;
      const c = Math.cos(angle);
      const encoded = Math.round(((c + 1) / 2) * 255);
      if (rd !== 0) regs[rd] = u8(encoded);
      break;
    }
    case OPCODES.FTAN: {
      // Keep tan bounded by mapping the input to [-pi/2+eps, pi/2-eps]
      const eps = 0.1;
      const maxAngle = Math.PI / 2 - eps;
      const angleByte = R(rs1);
      const angle = (angleByte / 255) * (2 * maxAngle) - maxAngle;

      const y = Math.tan(angle);
      const scale = Math.tan(maxAngle);
      const yNorm = (y / scale + 1) / 2;
      const encoded = Math.round(Math.max(0, Math.min(1, yNorm)) * 255);
      if (rd !== 0) regs[rd] = u8(encoded);
      break;
    }
    case OPCODES.FTANH: {
      const xByte = R(rs1);
      const x = (xByte / 255) * 6 - 3; // [-3, 3]
      const y = Math.tanh(x); // [-1, 1]
      const encoded = Math.round(((y + 1) / 2) * 255);
      if (rd !== 0) regs[rd] = u8(encoded);
      break;
    }
    case OPCODES.FSINH: {
      const xByte = R(rs1);
      const x = (xByte / 255) * 4 - 2; // [-2, 2]
      const y = Math.sinh(x);
      const scale = Math.sinh(2);
      const yNorm = (y / scale + 1) / 2;
      const encoded = Math.round(Math.max(0, Math.min(1, yNorm)) * 255);
      if (rd !== 0) regs[rd] = u8(encoded);
      break;
    }
    case OPCODES.FCOSH: {
      const xByte = R(rs1);
      const x = (xByte / 255) * 4 - 2; // [-2, 2]
      const y = Math.cosh(x); // [1, cosh(2)]
      const yNorm = (y - 1) / (Math.cosh(2) - 1);
      const encoded = Math.round(Math.max(0, Math.min(1, yNorm)) * 255);
      if (rd !== 0) regs[rd] = u8(encoded);
      break;
    }
    case OPCODES.FLN: {
      const xByte = R(rs1);
      const x = 1 + xByte; // [1, 256]
      const yNorm = Math.log(x) / Math.log(256); // [0, 1]
      const encoded = Math.round(yNorm * 255);
      if (rd !== 0) regs[rd] = u8(encoded);
      break;
    }
    case OPCODES.FLOG10: {
      const xByte = R(rs1);
      const x = 1 + xByte; // [1, 256]
      const yNorm = Math.log10(x) / Math.log10(256); // [0, 1]
      const encoded = Math.round(yNorm * 255);
      if (rd !== 0) regs[rd] = u8(encoded);
      break;
    }
    case OPCODES.FEXP: {
      const xByte = R(rs1);
      const x = (xByte / 255) * Math.log(256); // [0, ln(256)]
      // exp(x) in [1, 256], so exp(x)-1 in [0, 255]
      const y = Math.exp(x) - 1;
      const encoded = Math.round(Math.max(0, Math.min(255, y)));
      if (rd !== 0) regs[rd] = u8(encoded);
      break;
    }
    case OPCODES.FSQRT: {
      const xByte = R(rs1);
      const xNorm = xByte / 255;
      const yNorm = Math.sqrt(Math.max(0, xNorm));
      const encoded = Math.round(yNorm * 255);
      if (rd !== 0) regs[rd] = u8(encoded);
      break;
    }
    case OPCODES.HALT: {
      state.halted = true;
      break;
    }
    default:
      state.halted = true;
      break;
  }

  state.pc = pcNext;
}

function runRefProgram(program, maxSteps = 64) {
  const memory = new Array(MEM_SIZE).fill(0);
  const state = makeInitialState();

  let steps = 0;
  while (!state.halted && steps < maxSteps && state.pc < program.length) {
    const instr = program[state.pc];
    refStepCPU(state, memory, instr);
    steps++;
  }
  return { state, memory, steps };
}

// -----------------------------------------------------------------------------
// Encoding / decoding
// -----------------------------------------------------------------------------

function encodeByte(x) {
  return (x & 0xff) / 255;
}
function decodeByte(x) {
  let v = Math.round(Math.max(0, Math.min(1, x)) * 255);
  return v & 0xff;
}

function encodeState(state, memory) {
  const v = [];
  v.push(encodeByte(state.pc));
  for (let i = 0; i < NUM_REGS; i++) v.push(encodeByte(state.regs[i]));
  for (let i = 0; i < 4; i++) v.push(encodeByte(memory[i]));
  v.push(state.halted ? 1 : 0);
  return v;
}

function decodeState(vec) {
  let idx = 0;
  const pc = decodeByte(vec[idx++]);
  const regs = [];
  for (let i = 0; i < NUM_REGS; i++) regs.push(decodeByte(vec[idx++]));
  const memHead = [];
  for (let i = 0; i < 4; i++) memHead.push(decodeByte(vec[idx++]));
  const halted = vec[idx++] > 0.5;
  return { pc, regs, memHead, halted };
}

const NUM_OPCODES = Object.keys(OPCODES).length;
const REG_BITS = NUM_REGS;

function encodeInstruction(instr) {
  const v = [];
  for (let k = 0; k < NUM_OPCODES; k++) {
    v.push(instr.opcode === k ? 1 : 0);
  }
  for (let i = 0; i < REG_BITS; i++) v.push(instr.rd === i ? 1 : 0);
  for (let i = 0; i < REG_BITS; i++) v.push(instr.rs1 === i ? 1 : 0);
  for (let i = 0; i < REG_BITS; i++) v.push(instr.rs2 === i ? 1 : 0);
  const imm = instr.imm | 0;
  const immNorm = (imm + 128) / 255;
  v.push(immNorm);
  return v;
}

// -----------------------------------------------------------------------------
// Feedforward CPU core network (MLP)
// -----------------------------------------------------------------------------

class CPUCoreNet {
  constructor(inputDim, hiddenDim, outputDim) {
    this.inputDim = inputDim;
    this.hiddenDim = hiddenDim;
    this.outputDim = outputDim;

    this.W1 = randomMatrix(hiddenDim, inputDim);
    this.b1 = randomArray(hiddenDim);
    this.W2 = randomMatrix(outputDim, hiddenDim);
    this.b2 = randomArray(outputDim);
  }

  forward(input) {
    if (input.length !== this.inputDim) {
      throw new Error("CPUCoreNet: bad input length");
    }
    const h = new Array(this.hiddenDim);
    for (let i = 0; i < this.hiddenDim; i++) {
      let sum = this.b1[i];
      const wRow = this.W1[i];
      for (let j = 0; j < this.inputDim; j++) {
        sum += wRow[j] * input[j];
      }
      h[i] = Math.tanh(sum);
    }
    const out = new Array(this.outputDim);
    for (let i = 0; i < this.outputDim; i++) {
      let sum = this.b2[i];
      const wRow = this.W2[i];
      for (let j = 0; j < this.hiddenDim; j++) {
        sum += wRow[j] * h[j];
      }
      out[i] = 1 / (1 + Math.exp(-sum));
    }
    return out;
  }

  getNumWeights() {
    return (
      this.hiddenDim * this.inputDim +
      this.hiddenDim +
      this.outputDim * this.hiddenDim +
      this.outputDim
    );
  }

  getWeightsFlat() {
    const arr = [];
    for (let i = 0; i < this.hiddenDim; i++) {
      const wRow = this.W1[i];
      for (let j = 0; j < this.inputDim; j++) arr.push(wRow[j]);
    }
    for (let i = 0; i < this.hiddenDim; i++) arr.push(this.b1[i]);
    for (let i = 0; i < this.outputDim; i++) {
      const wRow = this.W2[i];
      for (let j = 0; j < this.hiddenDim; j++) arr.push(wRow[j]);
    }
    for (let i = 0; i < this.outputDim; i++) arr.push(this.b2[i]);
    return arr;
  }

  setWeightsFlat(flat) {
    let idx = 0;
    for (let i = 0; i < this.hiddenDim; i++) {
      const wRow = this.W1[i];
      for (let j = 0; j < this.inputDim; j++) {
        wRow[j] = flat[idx++];
      }
    }
    for (let i = 0; i < this.hiddenDim; i++) {
      this.b1[i] = flat[idx++];
    }
    for (let i = 0; i < this.outputDim; i++) {
      const wRow = this.W2[i];
      for (let j = 0; j < this.hiddenDim; j++) {
        wRow[j] = flat[idx++];
      }
    }
    for (let i = 0; i < this.outputDim; i++) {
      this.b2[i] = flat[idx++];
    }
  }
}

// -----------------------------------------------------------------------------
// GA helpers
// -----------------------------------------------------------------------------

function randomScalar() {
  return (Math.random() * 2 - 1) * 0.5;
}
function randomArray(len) {
  const a = new Array(len);
  for (let i = 0; i < len; i++) a[i] = randomScalar();
  return a;
}
function randomMatrix(rows, cols) {
  const m = new Array(rows);
  for (let r = 0; r < rows; r++) m[r] = randomArray(cols);
  return m;
}

function randomGenome(numGenes) {
  const g = new Array(numGenes);
  for (let i = 0; i < numGenes; i++) g[i] = randomScalar();
  return g;
}

function mutate(genome, mutationRate, mutationStd) {
  const out = genome.slice();
  for (let i = 0; i < out.length; i++) {
    if (Math.random() < mutationRate) {
      const u1 = Math.random();
      const u2 = Math.random();
      const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      out[i] += z * mutationStd;
    }
  }
  return out;
}

function crossover(a, b) {
  const child = new Array(a.length);
  for (let i = 0; i < a.length; i++) {
    child[i] = Math.random() < 0.5 ? a[i] : b[i];
  }
  return child;
}

// -----------------------------------------------------------------------------
// Test programs (including NARX op usage)
// -----------------------------------------------------------------------------

const testPrograms = [
  [
    { opcode: OPCODES.ADDI, rd: 1, rs1: 0, rs2: 0, imm: 1 },
    { opcode: OPCODES.ADDI, rd: 1, rs1: 1, rs2: 0, imm: 1 },
    { opcode: OPCODES.ADDI, rd: 1, rs1: 1, rs2: 0, imm: 1 },
    { opcode: OPCODES.HALT, rd: 0, rs1: 0, rs2: 0, imm: 0 },
  ],
  [
    { opcode: OPCODES.ADDI, rd: 1, rs1: 0, rs2: 0, imm: 5 },
    { opcode: OPCODES.ST,   rd: 1, rs1: 0, rs2: 0, imm: 0 },
    { opcode: OPCODES.LD,   rd: 2, rs1: 0, rs2: 0, imm: 0 },
    { opcode: OPCODES.HALT, rd: 0, rs1: 0, rs2: 0, imm: 0 },
  ],
  [
    { opcode: OPCODES.ADDI, rd: 1, rs1: 0, rs2: 0, imm: 64 },
    { opcode: OPCODES.FSIN, rd: 2, rs1: 1, rs2: 0, imm: 0 },
    { opcode: OPCODES.HALT, rd: 0, rs1: 0, rs2: 0, imm: 0 },
  ],
  [
    { opcode: OPCODES.ADDI, rd: 1, rs1: 0, rs2: 0, imm: 64 },
    { opcode: OPCODES.FCOS, rd: 2, rs1: 1, rs2: 0, imm: 0 },
    { opcode: OPCODES.HALT, rd: 0, rs1: 0, rs2: 0, imm: 0 },
  ],
  [
    { opcode: OPCODES.ADDI, rd: 1, rs1: 0, rs2: 0, imm: 200 },
    { opcode: OPCODES.FTAN, rd: 2, rs1: 1, rs2: 0, imm: 0 },
    { opcode: OPCODES.HALT, rd: 0, rs1: 0, rs2: 0, imm: 0 },
  ],
  [
    { opcode: OPCODES.ADDI, rd: 1, rs1: 0, rs2: 0, imm: 200 },
    { opcode: OPCODES.FTANH, rd: 2, rs1: 1, rs2: 0, imm: 0 },
    { opcode: OPCODES.HALT, rd: 0, rs1: 0, rs2: 0, imm: 0 },
  ],
  [
    { opcode: OPCODES.ADDI, rd: 1, rs1: 0, rs2: 0, imm: 200 },
    { opcode: OPCODES.FSINH, rd: 2, rs1: 1, rs2: 0, imm: 0 },
    { opcode: OPCODES.HALT, rd: 0, rs1: 0, rs2: 0, imm: 0 },
  ],
  [
    { opcode: OPCODES.ADDI, rd: 1, rs1: 0, rs2: 0, imm: 200 },
    { opcode: OPCODES.FCOSH, rd: 2, rs1: 1, rs2: 0, imm: 0 },
    { opcode: OPCODES.HALT, rd: 0, rs1: 0, rs2: 0, imm: 0 },
  ],
  [
    { opcode: OPCODES.ADDI, rd: 1, rs1: 0, rs2: 0, imm: 10 },
    { opcode: OPCODES.FLN, rd: 2, rs1: 1, rs2: 0, imm: 0 },
    { opcode: OPCODES.HALT, rd: 0, rs1: 0, rs2: 0, imm: 0 },
  ],
  [
    { opcode: OPCODES.ADDI, rd: 1, rs1: 0, rs2: 0, imm: 10 },
    { opcode: OPCODES.FLOG10, rd: 2, rs1: 1, rs2: 0, imm: 0 },
    { opcode: OPCODES.HALT, rd: 0, rs1: 0, rs2: 0, imm: 0 },
  ],
  [
    { opcode: OPCODES.ADDI, rd: 1, rs1: 0, rs2: 0, imm: 200 },
    { opcode: OPCODES.FEXP, rd: 2, rs1: 1, rs2: 0, imm: 0 },
    { opcode: OPCODES.HALT, rd: 0, rs1: 0, rs2: 0, imm: 0 },
  ],
  [
    { opcode: OPCODES.ADDI, rd: 1, rs1: 0, rs2: 0, imm: 100 },
    { opcode: OPCODES.FSQRT, rd: 2, rs1: 1, rs2: 0, imm: 0 },
    { opcode: OPCODES.HALT, rd: 0, rs1: 0, rs2: 0, imm: 0 },
  ],
];

// -----------------------------------------------------------------------------
// Fitness evaluation: core MLP + NARX complex ops
// -----------------------------------------------------------------------------

const NARX_OPS = [
  OPCODES.FSIN,
  OPCODES.FCOS,
  OPCODES.FTAN,
  OPCODES.FTANH,
  OPCODES.FSINH,
  OPCODES.FCOSH,
  OPCODES.FLN,
  OPCODES.FLOG10,
  OPCODES.FEXP,
  OPCODES.FSQRT,
];

function makeNarxStepper(narxModel) {
  const inputs = [];
  const outputs = [];

  return function step(inputNorm) {
    inputs.push(inputNorm);
    const t = inputs.length - 1;

    const laggedInput = [];
    const laggedOutput = [];
    for (let i = narxModel.inputLag; i >= 1; i--) {
      const idx = t - i;
      laggedInput.push(idx >= 0 ? inputs[idx] : 0.0);
    }
    for (let i = narxModel.outputLag; i >= 1; i--) {
      const idx = outputs.length - i;
      laggedOutput.push(idx >= 0 ? outputs[idx] : 0.0);
    }

    const combined = laggedInput.concat(laggedOutput);
    const { output } = narxModel.forward(combined);
    outputs.push(output);
    return output;
  };
}

function evaluateFitness(genome, coreTemplate, coreGeneCount, narxGeneCount, narxOps = NARX_OPS) {
  const coreModel = new CPUCoreNet(
    coreTemplate.inputDim,
    coreTemplate.hiddenDim,
    coreTemplate.outputDim
  );
  coreModel.setWeightsFlat(genome.slice(0, coreGeneCount));

  const narxModels = {};
  let offset = coreGeneCount;
  for (const op of narxOps) {
    const model = new NARX(2, 2, 8); // small NARX: inputLag=2, outputLag=2, hidden=8
    model.setWeightsFlat(genome.slice(offset, offset + narxGeneCount));
    narxModels[op] = model;
    offset += narxGeneCount;
  }

  let totalError = 0;

  for (const prog of testPrograms) {
    const refMem = new Array(MEM_SIZE).fill(0);
    const refState = makeInitialState();
    let steps = 0;
    while (!refState.halted && steps < 16 && refState.pc < prog.length) {
      const instr = prog[refState.pc];
      refStepCPU(refState, refMem, instr);
      steps++;
    }
    const refEnc = encodeState(refState, refMem);

    const neuMem = new Array(MEM_SIZE).fill(0);
    let neuState = makeInitialState();
    steps = 0;

    const narxSteppers = {};
    for (const op of narxOps) {
      narxSteppers[op] = makeNarxStepper(narxModels[op]);
    }

    while (!neuState.halted && steps < 16 && neuState.pc < prog.length) {
      const instr = prog[neuState.pc];

      const narxStep = narxSteppers[instr.opcode];
      if (narxStep) {
        const inputByte = neuState.regs[instr.rs1] & 0xff;
        const inputNorm = inputByte / 255;
        const outNorm = narxStep(inputNorm);
        const outByte = decodeByte(outNorm);
        if (instr.rd !== 0) neuState.regs[instr.rd] = outByte;
        neuState.pc = (neuState.pc + 1) & 0xff;
      } else {
        const stateVec = encodeState(neuState, neuMem);
        const instrVec = encodeInstruction(instr);
        const inputVec = stateVec.concat(instrVec);
        const outVec = coreModel.forward(inputVec);
        const decoded = decodeState(outVec);

        neuState.pc = decoded.pc % prog.length;
        for (let r = 0; r < NUM_REGS; r++) neuState.regs[r] = decoded.regs[r];
        for (let i = 0; i < 4; i++) neuMem[i] = decoded.memHead[i];
        neuState.halted = decoded.halted;
      }

      steps++;
    }

    const neuEnc = encodeState(neuState, neuMem);
    let err = 0;
    for (let i = 0; i < refEnc.length; i++) {
      err += Math.abs(refEnc[i] - neuEnc[i]);
    }
    totalError += err;
  }

  return -totalError;
}

// -----------------------------------------------------------------------------
// GA main
// -----------------------------------------------------------------------------

function runGA() {
  const dummyState = encodeState(
    makeInitialState(),
    new Array(MEM_SIZE).fill(0)
  );
  const dummyInstr = encodeInstruction({
    opcode: OPCODES.ADD,
    rd: 1,
    rs1: 2,
    rs2: 3,
    imm: 0,
  });
  const inputDim = dummyState.length + dummyInstr.length;
  const hiddenDim = 32;
  const outputDim = dummyState.length;

  const coreTemplate = {
    inputDim,
    hiddenDim,
    outputDim,
  };
  const coreProbe = new CPUCoreNet(inputDim, hiddenDim, outputDim);
  const coreGeneCount = coreProbe.getNumWeights();

  const narxProbe = new NARX(2, 2, 8);
  const narxGeneCount = narxProbe.getNumWeights();

  const totalGenes = coreGeneCount + NARX_OPS.length * narxGeneCount;

  const populationSize = 30;
  const numGenerations = 50;
  const mutationRate = 0.05;
  const mutationStd = 0.2;
  const crossoverRate = 0.7;
  const eliteCount = 2;

  let population = [];
  for (let i = 0; i < populationSize; i++) {
    const genes = randomGenome(totalGenes);
    population.push({ genes, fitness: -Infinity });
  }

  let bestEver = null;

  for (let gen = 0; gen < numGenerations; gen++) {
    for (const indiv of population) {
      indiv.fitness = evaluateFitness(
        indiv.genes,
        coreTemplate,
        coreGeneCount,
        narxGeneCount,
        NARX_OPS
      );
    }

    population.sort((a, b) => b.fitness - a.fitness);

    if (!bestEver || population[0].fitness > bestEver.fitness) {
      bestEver = {
        genes: population[0].genes.slice(),
        fitness: population[0].fitness,
      };
    }

    const avgFitness =
      population.reduce((s, x) => s + x.fitness, 0) / population.length;

    console.log(
      `Gen ${gen}: best=${population[0].fitness.toFixed(
        3
      )}, avg=${avgFitness.toFixed(3)}`
    );

    const newPop = population.slice(0, eliteCount);

    const pickParent = () => {
      const i = (Math.random() * population.length) | 0;
      const j = (Math.random() * population.length) | 0;
      return population[i].fitness > population[j].fitness
        ? population[i]
        : population[j];
    };

    while (newPop.length < populationSize) {
      const pA = pickParent();
      const pB = pickParent();
      let childGenes =
        Math.random() < crossoverRate
          ? crossover(pA.genes, pB.genes)
          : pA.genes.slice();
      childGenes = mutate(childGenes, mutationRate, mutationStd);
      newPop.push({ genes: childGenes, fitness: -Infinity });
    }

    population = newPop;
  }

  console.log("Best ever fitness:", bestEver.fitness);

  // Test best genome
  const bestCore = new CPUCoreNet(inputDim, hiddenDim, outputDim);
  bestCore.setWeightsFlat(bestEver.genes.slice(0, coreGeneCount));

  const bestNarxModels = {};
  let narxOffset = coreGeneCount;
  for (const op of NARX_OPS) {
    const model = new NARX(2, 2, 8);
    model.setWeightsFlat(bestEver.genes.slice(narxOffset, narxOffset + narxGeneCount));
    bestNarxModels[op] = model;
    narxOffset += narxGeneCount;
  }

  console.log("Testing best model:");
  for (const [idx, prog] of testPrograms.entries()) {
    const refRes = runRefProgram(prog);
    const refEnc = encodeState(refRes.state, refRes.memory);

    const neuMem = new Array(MEM_SIZE).fill(0);
    let neuState = makeInitialState();
    let steps = 0;

    const narxSteppers = {};
    for (const op of NARX_OPS) {
      narxSteppers[op] = makeNarxStepper(bestNarxModels[op]);
    }

    while (!neuState.halted && steps < 16 && neuState.pc < prog.length) {
      const instr = prog[neuState.pc];

      const narxStep = narxSteppers[instr.opcode];
      if (narxStep) {
        const inputByte = neuState.regs[instr.rs1] & 0xff;
        const inputNorm = inputByte / 255;
        const outNorm = narxStep(inputNorm);
        const outByte = decodeByte(outNorm);
        if (instr.rd !== 0) neuState.regs[instr.rd] = outByte;
        neuState.pc = (neuState.pc + 1) & 0xff;
      } else {
        const stateVec = encodeState(neuState, neuMem);
        const instrVec = encodeInstruction(instr);
        const inputVec = stateVec.concat(instrVec);
        const outVec = bestCore.forward(inputVec);
        const decoded = decodeState(outVec);

        neuState.pc = decoded.pc % prog.length;
        for (let r = 0; r < NUM_REGS; r++) neuState.regs[r] = decoded.regs[r];
        for (let i = 0; i < 4; i++) neuMem[i] = decoded.memHead[i];
        neuState.halted = decoded.halted;
      }

      steps++;
    }

    const neuEnc = encodeState(neuState, neuMem);
    let err = 0;
    for (let i = 0; i < refEnc.length; i++) {
      err += Math.abs(refEnc[i] - neuEnc[i]);
    }

    console.log(`Program ${idx}:`);
    console.log("  Ref:", refRes.state, "mem[0..3]=", refRes.memory.slice(0, 4));
    console.log("  Neu:", neuState, "mem[0..3]=", neuMem.slice(0, 4));
    console.log("  Encoded L1 error:", err.toFixed(4));
  }
}

if (require.main === module) {
  runGA();
}
