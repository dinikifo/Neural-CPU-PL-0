// run_pl0.js
//
// Convenience runner for compiling and executing PL/0 source files without
// editing pl0_cpu_sim.js.
//
// Usage:
//   node run_pl0.js <file.pl0> [--entry=name]
//                  [--fxScale=65536] [--maxSteps=1000000]
//                  [--dump-asm] [--dump-mem=lo:hi]
//                  [--neural ...] [--narx-math ...]
//
// Examples:
//   node run_pl0.js examples/hello_stack.pl0 --entry=helloStack
//   node run_pl0.js examples/fixedpoint_area.pl0 --entry=fixedpointArea --dump-mem=0:32
//   node run_pl0.js examples/trig_chain.pl0 --entry=trigChain --narx-math --train-math

const fs = require('fs');
const path = require('path');

const { PL0CPU, compilePL0, PL0Programs } = require('./pl0_cpu_sim');

function parseNumArg(argv, name, def) {
  const a = argv.find((x) => x.startsWith(`--${name}=`));
  if (!a) return def;
  const v = Number(a.split('=')[1]);
  return Number.isFinite(v) ? v : def;
}

function parseStrArg(argv, name, def) {
  const a = argv.find((x) => x.startsWith(`--${name}=`));
  if (!a) return def;
  return a.split('=')[1] ?? def;
}

function parseRangeArg(argv, name, defLo, defHi) {
  const raw = parseStrArg(argv, name, `${defLo}:${defHi}`);
  const m = /^(-?\d+)\s*:\s*(-?\d+)$/.exec(raw);
  if (!m) return { lo: defLo, hi: defHi };
  return { lo: parseInt(m[1], 10), hi: parseInt(m[2], 10) };
}

function extractPrograms(text) {
  // Non-greedy match for: program <name>; ... end.
  // We terminate on the *program-ending* dot after an 'end', so float literals
  // like 2.5 do not prematurely terminate extraction.
  const re = /\bprogram\s+([A-Za-z][A-Za-z0-9_]*)\s*;[\s\S]*?\bend\s*\./gmi;
  const out = [];
  let m;
  while ((m = re.exec(text)) !== null) {
    out.push({ name: m[1], source: m[0] });
  }
  return out;
}

function main() {
  const argv = process.argv.slice(2);
  const file = argv.find((a) => !a.startsWith('--'));
  if (!file) {
    console.error('Usage: node run_pl0.js <file.pl0> [--entry=name] ...');
    process.exit(1);
  }

  const abs = path.resolve(process.cwd(), file);
  const text = fs.readFileSync(abs, 'utf8');
  const progs = extractPrograms(text);
  if (progs.length === 0) {
    throw new Error(`No PL/0 programs found in '${file}'. Expected 'program name; ... .'`);
  }

  // Fixed-point scale for literals/constants and intrinsic ops.
  const fxScale = parseNumArg(argv, 'fxScale', 65536);

  // Where each program's variables begin in memory.
  // Keeping them separate avoids accidental overlap.
  const baseStep = parseNumArg(argv, 'baseStep', 32);

  // Optional explicit base mapping: --baseMap=setElement:0,getElement:0,matrixTest:20
  const baseMapRaw = parseStrArg(argv, 'baseMap', '');
  const baseMap = new Map();
  if (baseMapRaw) {
    for (const part of baseMapRaw.split(',')) {
      const [k, v] = part.split(':');
      if (!k || v === undefined) continue;
      const n = Number(v);
      if (Number.isFinite(n)) baseMap.set(k, n);
    }
  }

  // Compile all programs found.
  let nextBase = 0;
  for (const p of progs) {
    const base = baseMap.has(p.name) ? baseMap.get(p.name) : nextBase;
    compilePL0(p.source, base, { fxScale });
    nextBase = base + baseStep;
  }

  const entry = parseStrArg(argv, 'entry', parseStrArg(argv, 'program', progs[0].name));
  if (!PL0Programs[entry]) {
    throw new Error(`Entry program '${entry}' was not compiled. Available: ${Object.keys(PL0Programs).join(', ')}`);
  }

  if (argv.includes('--dump-asm')) {
    for (const [name, code] of Object.entries(PL0Programs)) {
      console.log(`\n=== ${name} (assembly) ===`);
      for (let i = 0; i < code.length; i++) {
        console.log(String(i).padStart(4, ' ') + '  ' + code[i]);
      }
    }
    console.log('');
  }

  // Optional neural ALU
  let alu = null;
  if (argv.includes('--neural')) {
    const { NeuralALU } = require('./neural_alu');
    const arch = parseStrArg(argv, 'alu-arch', 'linear');
    const scale = parseNumArg(argv, 'scale', 65536);
    const mix = parseNumArg(argv, 'mix', 1.0);
    const safetyFallback = !argv.includes('--no-fallback');
    const fallbackAbsError = parseNumArg(argv, 'fallbackAbs', 2);

    alu = new NeuralALU({
      architecture: arch,
      hidden: 32,
      scale,
      mix,
      safetyFallback,
      fallbackAbsError,
      divDecode: 'floor',
    });

    // If you pick MLP or explicitly request training, do quick supervised fitting.
    if (arch !== 'linear' || argv.includes('--train')) {
      const epochs = parseNumArg(argv, 'epochs', 10);
      const stepsPerEpoch = parseNumArg(argv, 'steps', 3000);
      const lr = parseNumArg(argv, 'lr', 0.01);
      alu.trainQuick({ epochs, stepsPerEpoch, lr, operandRange: 256, mulRange: 255 });
    }
  }

  // Optional neural NARX math
  let math = null;
  if (argv.includes('--narx-math')) {
    const { NeuralMathNARX } = require('./neural_math_narx');
    const mix = parseNumArg(argv, 'mathMix', 1.0);
    const safetyFallback = !argv.includes('--no-math-fallback');
    const fallbackAbsError = parseNumArg(argv, 'mathFallbackAbs', 0.001);

    math = new NeuralMathNARX({
      scale: fxScale,
      mix,
      safetyFallback,
      fallbackAbsError,
    });

    if (argv.includes('--train-math')) {
      const epochs = parseNumArg(argv, 'mathEpochs', 80);
      const length = parseNumArg(argv, 'mathLen', 6000);
      const learningRate = parseNumArg(argv, 'mathLr', 0.02);
      math.trainQuick({ epochs, length, learningRate });
    }
  }

  const cpu = new PL0CPU(4, 256, 256, { neuralALU: alu, neuralMath: math, fxScale });
  cpu.addInstructions([`PL0CALL ${entry}`, 'HALT']);

  const maxSteps = parseNumArg(argv, 'maxSteps', 1_000_000);
  cpu.execute(maxSteps);

  // Output
  const { lo, hi } = parseRangeArg(argv, 'dump-mem', 30, 42);
  const loC = Math.max(0, Math.min(255, lo));
  const hiC = Math.max(loC, Math.min(256, hi));

  console.log(`\nEntry: ${entry}`);
  console.log(`fxScale: ${fxScale}`);
  console.log(`Memory [${loC}..${hiC}):`, cpu.memory.slice(loC, hiC));
  console.log('Data stack:', cpu.dataStack);

  if (cpu.neuralStats?.enabled) {
    console.log('\nNeural ALU ops executed:', cpu.neuralStats.ops);
    console.log('Neural ALU fallbacks:', cpu.neuralStats.fallbacks);
    const avgAbs = {};
    for (const k of Object.keys(cpu.neuralStats.ops)) {
      const n = cpu.neuralStats.ops[k] || 0;
      avgAbs[k] = n ? (cpu.neuralStats.absErrorSum[k] / n) : 0;
    }
    console.log('Neural ALU avg |pred-exact|:', avgAbs);
  }

  if (cpu.neuralMathStats?.enabled) {
    console.log('\nNeural NARX-math calls:', cpu.neuralMathStats.calls);
    console.log('Neural NARX-math fallbacks:', cpu.neuralMathStats.fallbacks);
    const avgAbs = {};
    for (const k of Object.keys(cpu.neuralMathStats.calls)) {
      const n = cpu.neuralMathStats.calls[k] || 0;
      avgAbs[k] = n ? (cpu.neuralMathStats.absErrorSum[k] / n) : 0;
    }
    console.log('Neural NARX-math avg |predNorm-exactNorm|:', avgAbs);
  }
}

if (require.main === module) {
  main();
}
