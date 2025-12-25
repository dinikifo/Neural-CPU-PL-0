# Tutorial: PL/0 on the Neural NARX CPU

This tutorial is a hands-on walkthrough: run demos, compile your own PL/0 files, and turn on neural execution.

For a complete reference (ISA, fixed-point, neural flags), see **MANUAL.md**.

All commands are run from the repository root as `node ...`.

---
## 1) Run the built-in demos

### 1.1 Matrix demo (calls + stack + memory)

```bash
node pl0_cpu_sim.js --program=matrixTest
```

What to look for:

- `matrixTest` pushes 4 values then calls `setElement`
- `setElement` writes into memory starting around address 30
- `matrixTest` then calls `getElement`, which reads back and pushes the value onto the stack

The script prints:

- `Final memory state (addresses 30-42)`
- `Final data stack`

### 1.2 Math demo (fixed-point + intrinsics)

```bash
node pl0_cpu_sim.js --program=mathTest
```

`mathTest` uses:

- float literals (`0.0`, `2.0`)
- constants (`pi`)
- intrinsics (`sin`, `cos`, `ln`, `sqrt`)
- conversion helper `int(...)`

---

## 2) Run your own PL/0 file

Use the helper runner:

```bash
node run_pl0.js examples/hello_stack.pl0 --entry=helloStack
```

Useful flags:

- `--dump-asm` – print compiled assembly
- `--dump-mem=0:64` – print a memory slice
- `--maxSteps=200000` – change the instruction limit

Try the fixed-point area example:

```bash
node run_pl0.js examples/fixedpoint_area.pl0 --entry=fixedpointArea
```

Note: the PL/0 statements `push ...;` and `pop ...;` work on **variables only**. If you want to push an expression result, assign it to a variable first.

---

## 3) Fixed-point math: the mental model

Intrinsic math is fixed-point, default **Q16.16**:

- `fxScale = 65536`
- `1.0` compiles to `65536`
- `pi` compiles to `round(pi * 65536)`

### Multiplying fixed-point values

If `a` and `b` are fixed-point, then:

- `a*b` is scaled by `fxScale^2`
- to get back to fixed-point, you must divide by `fxScale`

In PL/0 you can do that by dividing by `1.0` (because `1.0` compiles to `fxScale`):

```pl0
x := a * b;
x := x / 1.0;  // rescale back to Q16.16
```

### Converting to integer

Use `int(x)` to drop the fixed-point fraction:

```pl0
whole := int(x);
```

---

## 4) Turn on neural execution

### 4.1 Neural ALU (ADD/SUB/MUL/DIV)

```bash
node pl0_cpu_sim.js --program=matrixTest --neural
```

By default the ALU uses `--alu-arch=linear`, which is **analytic-initialized**:

- ADD/SUB/MUL are exact (until feature clamping/saturation)
- DIV starts close and can be trained

Try the MLP ALU:

```bash
node pl0_cpu_sim.js --program=matrixTest --neural --alu-arch=mlp --train --epochs=10 --steps=3000 --lr=0.01
```

Important knobs:

- `--mix=1.0` (pure neural) down to `--mix=0.0` (pure exact)
- `--fallbackAbs=2` and `--no-fallback`
- `--scale=65536` (bigger reduces saturation)

### 4.2 Neural NARX math coprocessor (sin/ln/sqrt/...)

```bash
node pl0_cpu_sim.js --program=mathTest --narx-math
```

Optional pretraining:

```bash
node pl0_cpu_sim.js --program=mathTest --narx-math --train-math --mathEpochs=80 --mathLen=6000 --mathLr=0.02
```

NARX knobs:

- `--mathMix=1.0` / `--mathMix=0.5`
- `--mathFallbackAbs=0.001` (threshold in normalized 0..1 units)
- `--no-math-fallback`

---

## 5) Add a new intrinsic (quick recipe)

1) In `pl0_cpu_sim.js`, add a mapping in `PL0Parser.intrinsics`, e.g.

```js
this.intrinsics.atan = 'FATAN';
```

2) Add an opcode handler in the CPU switch (pattern already used by `FSIN`, `FLN`, etc.)

3) If you want neural NARX support:

- add an entry to `DEFAULT_OP_SPECS` in `neural_math_narx.js`
- add the op name to `ops`

---

## 6) Exercises

- Write a PL/0 program that approximates `sin(x)` for small `x` using `x - x^3/6` and compare to `sin(x)`.
- Run it with `--narx-math --no-math-fallback` and see how it drifts.
- Implement a new intrinsic `abs(x)` deterministically (no NARX) and use it in a loop.
