# Neural CPU + PL/0 (Hybrid Neural Execution)

A tiny **PL/0 toolchain** (compiler + CPU simulator) with optional **neural execution**:
- a learnable **Neural ALU** for integer ops (`ADD/SUB/MUL/DIV`)
- a learnable **NARX “math coprocessor”** for transcendental intrinsics (`sin/ln/sqrt/...`) in **fixed‑point**

This repo is meant for **experiments**, demos, and education: *what happens when “CPU instructions” are partially learned?*

## Quick start

Requirements: **Node.js 18+** (no native deps).

```bash
# deterministic execution (reference semantics)
node pl0_cpu_sim.js --program=matrixTest

# fixed‑point intrinsics demo (deterministic math)
node pl0_cpu_sim.js --program=mathTest

# enable neural ALU (still uses deterministic control flow + memory)
node pl0_cpu_sim.js --program=matrixTest --neural

# enable NARX neural math coprocessor (optional supervised pretrain)
node pl0_cpu_sim.js --program=mathTest --narx-math --train-math
```

Run your own `.pl0` file:

```bash
node run_pl0.js examples/hello_stack.pl0 --entry=helloStack
node run_pl0.js examples/trig_chain.pl0 --entry=trigChain --narx-math --train-math
```

## What you get

### 1) A PL/0 compiler
A small recursive‑descent compiler that outputs a simple assembly suitable for a stack machine:
- variables, assignment, `if/then`, `while/do`
- procedures (`procedure p; ...;`) + `call p`
- expression arithmetic: `+ - * /` (integer semantics)

### 2) A CPU simulator for that assembly
A compact “counter machine” style VM:
- registers: `r0..r3`
- memory: 256 integer cells
- `dataStack` + `callStack`
- `CALL/RET`, `JMP/JZ/JNZ`, `LOAD/STORE/PEEK/POKE`, `PUSH/POP`, `HALT`

### 3) Fixed‑point intrinsic math (Q16.16 by default)
Math intrinsics operate on **fixed‑point integers** (default scale `fxScale=65536`):

```pl0
a := 0.5;          /* float literal -> fixed point */
b := pi;           /* built-in constant */
c := sin(b);       /* returns fixed point */
d := int(c);       /* floor(c / fxScale) */
e := fx(3);        /* 3 * fxScale */
```

Supported intrinsics:
`sin, cos, tan, tanh, sinh, cosh, ln, log/log10, exp, sqrt`

### 4) Neural execution (optional)
Neural execution is **hybrid by design**:

- **Deterministic:** memory, stacks, dispatch, and control flow  
- **Neural (optional):**
  - NeuralALU: `ADD/SUB/MUL/DIV`
  - NeuralMathNARX: intrinsic math ops

Both neural components support:
- **mixing** (`--mix=...` / `--mathMix=...`) to blend exact + learned behavior
- **safety fallback** (default on) to prevent blow‑ups

## CLI (high level)

### `pl0_cpu_sim.js`
- `--program=matrixTest|mathTest|...`
- `--fxScale=65536`
- `--neural` `[--alu-arch=linear|mlp] [--train] [--epochs=.. --steps=.. --lr=..]`
- `--narx-math` `[--train-math --mathEpochs=.. --mathLen=.. --mathLr=..]`
- `--mix=1.0` / `--mathMix=1.0`
- `--no-fallback` / `--no-math-fallback`

### `run_pl0.js`
- `node run_pl0.js file.pl0 --entry=myProgram`
- `--dump-asm`
- `--dump-mem=lo:hi`
- `--maxSteps=1000000`
- plus the same neural flags as above

For a complete reference, see **docs/MANUAL.md**.

## Repo layout

- `pl0_cpu_sim.js` — CPU simulator + PL/0 compiler + built-in demos
- `run_pl0.js` — compile & run external `.pl0` files
- `neural_alu.js` — NeuralALU implementation (linear + MLP options)
- `neural_math_narx.js` — NARX math coprocessor
- `narx.js` — generic NARX network
- `examples/` — sample PL/0 programs
- `docs/TUTORIAL.md` — hands-on walkthrough
- `docs/MANUAL.md` — full user manual

## Notes / caveats

- This is a **toy research simulator**, not a real CPU emulator.
- Neural modes can change program behavior (especially in loops). Keep fallbacks on if you want stability.
- Fixed‑point intrinsics are clamped to safe input/output ranges (documented in the manual).

## AI-Assisted Creation & Provenance

Some parts of this repository were created or refined with the assistance of large language models (LLMs) at the author’s direction. The author reviewed and integrated the results.

The intent is to place this work as completely as possible into the public domain via CC0-1.0 (see License below).

If you believe any snippet inadvertently reproduces third-party copyrighted code in a way that conflicts with the license, please open an issue with details (file, lines, source link). We will promptly rewrite or remove the material.

Privacy note: Don’t paste sensitive or proprietary material into issues or pull requests; treat prompts/logs as public.

## License (Public Domain)

This project is released into the **public domain** under **CC0‑1.0**.  
See `LICENSE`.
