# User Manual

This manual documents the PL/0 toolchain and the hybrid neural CPU simulator.

> If you want a step-by-step “learn by running” guide, start with **docs/TUTORIAL.md**.

---

## 1. Overview

This repo contains:

- a **PL/0 compiler** that targets a small stack-machine style assembly
- a **CPU simulator** for that assembly
- optional **neural execution**:
  - a learnable **NeuralALU** for integer arithmetic (`ADD/SUB/MUL/DIV`)
  - a learnable **NeuralMathNARX** coprocessor for fixed‑point transcendental math

Design goal: keep *program structure* deterministic (memory/control flow) while making
*some instructions* learnable and swappable.

---

## 2. Installation

Requirements:
- Node.js **18+**

No dependencies are required.

Run the built-in demos:

```bash
node pl0_cpu_sim.js --program=matrixTest
node pl0_cpu_sim.js --program=mathTest
```

Run a `.pl0` file:

```bash
node run_pl0.js examples/hello_stack.pl0 --entry=helloStack
```

---

## 3. The CPU model

### 3.1 State

- **Registers:** `r0`, `r1`, `r2`, `r3` (integer JS numbers)
- **Memory:** 256 integer cells (index `0..255`)
- **Stacks:**
  - `dataStack` (PUSH/POP are for values)
  - `callStack` (CALL/RET return addresses)
- **Instruction pointer:** `pointer` (PC)

The simulator enforces a maximum stack size (default 256).

### 3.2 Integer semantics

- Arithmetic uses JavaScript numbers but is intended to behave like **integer math**.
- `DIV` uses `Math.floor(a / b)` (matching the bundled compiler/runtime expectation).

---

## 4. Instruction set

Assembly is line-based text. Labels are supported (`label:`).

### 4.1 Data movement

- `LOAD rX, #imm`  
  Load immediate into register.
- `LOAD rX, [addr]`  
  Load memory cell at absolute address.
- `STORE rX, [addr]`  
  Store register into memory at absolute address.

- `PEEK rX, [addr]` / `POKE rX, [addr]`  
  Synonyms for load/store (kept for compatibility with some PL/0-style examples).

### 4.2 Stack

- `PUSH rX`  
  Push register value onto data stack.
- `POP rX`  
  Pop data stack into register.

### 4.3 Arithmetic

- `ADD rX, rY` → `rX = rX + rY`
- `SUB rX, rY` → `rX = rX - rY`
- `MUL rX, rY` → `rX = rX * rY`
- `DIV rX, rY` → `rX = floor(rX / rY)`

These ops can be executed by the **NeuralALU** when enabled.

### 4.4 Control flow

- `JMP label|#imm`
- `JZ rX, label|#imm` (jump if `rX === 0`)
- `JNZ rX, label|#imm` (jump if `rX !== 0`)
- `CALL label|#imm`  
  Push return address to `callStack`, then jump.
- `RET`  
  Pop return address from `callStack` and jump.
- `HALT`

### 4.5 Program entry helper

- `PL0CALL programName`  
  A convenience “meta instruction” used by the built-in demos to jump into a compiled PL/0 program block.

---

## 5. PL/0 language support

The compiler is intentionally small. It supports the common PL/0 subset:

- `program Name; ... begin ... end.`
- `const` / `var` blocks
- `procedure p; ...;`
- statements:
  - assignment: `x := expr;`
  - `call p;`
  - `if cond then stmt;`
  - `while cond do stmt;`
  - `begin ... end` blocks
  - `push expr;` / `pop x;` (stack I/O for demos)

Expressions support `+ - * /` with parentheses.

### 5.1 Conditions

Conditions support relations like `= != < <= > >=` (compiled into arithmetic + `JZ/JNZ` style branching).

---

## 6. Fixed‑point math intrinsics

Intrinsic math is **fixed‑point**.

### 6.1 Scale

By default, `fxScale = 65536` (Q16.16 style).

- real value `v` is stored as integer `vFx = round(v * fxScale)`
- to convert back: `v = vFx / fxScale`

You can change it:

```bash
node pl0_cpu_sim.js --program=mathTest --fxScale=65536
node run_pl0.js examples/fixedpoint_area.pl0 --entry=fixedpointArea --fxScale=65536
```

### 6.2 Float literals and constants (compiler sugar)

Because PL/0 is integer-oriented, the compiler adds sugar:

- Float literals: `1.5`, `2e-3` compile to `round(value * fxScale)`
- Built-in constants: `pi`, `tau`, `e` compile to fixed-point
- Helpers:
  - `fx(expr)` / `tofx(expr)` → `expr * fxScale`
  - `int(expr)` / `fromfx(expr)` / `unfx(expr)` → `floor(expr / fxScale)`

Integer literals remain **unscaled** for compatibility with classic integer PL/0 programs.

### 6.3 Intrinsic functions

Unary intrinsics (argument and result are fixed‑point ints):

- `sin(x)`, `cos(x)`, `tan(x)`
- `tanh(x)`, `sinh(x)`, `cosh(x)`
- `ln(x)`
- `log(x)` / `log10(x)`
- `exp(x)`
- `sqrt(x)`

These compile into CPU ops like `FSIN r0`, `FLN r0`, etc.

### 6.4 Clamping / domains

To keep fixed-point outputs sane (and to make deterministic vs neural comparable),
the reference implementation clamps inputs/outputs to safe ranges. For example:

- `sin/cos`: input clamped to `[-π, π]`, output in `[-1, 1]`
- `tan`: input clamped to ~`[-1.3, 1.3]`, output clamped to `[-8, 8]`
- `ln`: input clamped to a small positive minimum
- `sqrt`: input clamped to `>= 0`

(See `refMathFx()` in `pl0_cpu_sim.js` for exact ranges.)

---

## 7. Neural execution

Neural execution is optional and intentionally **hybrid**:

- control flow + memory are deterministic
- arithmetic and/or intrinsic math can be neural

### 7.1 NeuralALU (ADD/SUB/MUL/DIV)

Enable:

```bash
node pl0_cpu_sim.js --program=matrixTest --neural
```

Key flags:

- `--alu-arch=linear|mlp`  
  `linear` is fast and stable; `mlp` is more expressive but usually needs training.
- `--train`  
  Triggers supervised quick training (especially useful for `mlp`).
- `--epochs=10 --steps=3000 --lr=0.01`  
  Training controls.
- `--mix=1.0`  
  Blend factor (1.0 = fully neural, 0.0 = fully exact).
- `--no-fallback`  
  Disable safety fallback.
- `--fallbackAbs=2`  
  Fallback threshold for absolute error (only used when fallback is enabled).

### 7.2 NeuralMathNARX (intrinsic ops)

Enable:

```bash
node pl0_cpu_sim.js --program=mathTest --narx-math
```

Optional supervised pretraining:

```bash
node pl0_cpu_sim.js --program=mathTest --narx-math --train-math --mathEpochs=80 --mathLen=6000 --mathLr=0.02
```

Key flags:

- `--mathMix=1.0`  
  Blend learned vs reference output.
- `--no-math-fallback`
- `--mathFallbackAbs=0.001`  
  Error threshold (in *real units*, not scaled integers).

### 7.3 Practical recommendation

If you want programs to “mostly work” while still being neural:
- keep fallbacks on
- use mixing (`--mix=0.25` or `--mathMix=0.25`) and slowly increase

---

## 8. Runner script (`run_pl0.js`)

`run_pl0.js` compiles a `.pl0` file and executes one selected program.

```bash
node run_pl0.js examples/trig_chain.pl0 --entry=trigChain --narx-math --train-math
```

Useful flags:

- `--entry=name` (required if the file has multiple `program ...; ... end.` blocks)
- `--dump-asm` print compiled assembly
- `--dump-mem=lo:hi` print a memory slice (e.g. `--dump-mem=0:64`)
- `--maxSteps=1000000` guard against infinite loops

---

## 9. Extending the system

### 9.1 Add a new intrinsic math op

1. Add an op name to the intrinsic map in the parser (`this.intrinsics`).
2. Add the op to the CPU’s math dispatch (`case 'FSIN': ...`) list.
3. Implement deterministic reference behavior in `refMathFx(op, xFx, fxScale)`.
4. (Optional) teach the NARX coprocessor about it in `neural_math_narx.js`.

### 9.2 Add new language sugar

The tokenizer/parser lives inside `pl0_cpu_sim.js` (search for `tokenize` and `class PL0Parser`).

---

## 10. FAQ

**Why fixed-point?**  
It lets transcendental math coexist with integer PL/0 semantics, and makes neural error measurable.

**Is it “safe” to run arbitrary code?**  
It’s a toy VM; don’t treat it as a sandbox.

**Why hybrid neural execution?**  
Learning correct memory/control-flow semantics is hard. This repo focuses on
learnable *instructions* while keeping the machine usable for structured programs.
