// pl0_cpu_sim.js
//
// A PL/0-capable CPU simulator, compatible with the instruction set described
// in the provided manual PDF:
//   LOAD/STORE/PUSH/POP/PEEK/POKE/ADD/SUB/MUL/DIV/JMP/JZ/JNZ/CALL/PL0CALL/RET/HALT
//
// This script includes:
//   1) CounterMachine-compatible CPU simulator
//   2) A small PL/0 compiler (tokenize + recursive descent parser) that compiles
//      into the above assembly (same style as in the PDF)
//   3) A demo (matrixTest) that compiles and runs
//
// Run:
//   node pl0_cpu_sim.js


// -----------------------------------------------------------------------------
// CPU simulator (CounterMachine-compatible)
// -----------------------------------------------------------------------------

function clamp(x, lo, hi) {
  return Math.max(lo, Math.min(hi, x));
}

// Deterministic reference for the extended unary math ops.
//
// Fixed-point convention (default Q16.16):
//   - registers/memory hold integers
//   - a real value v is encoded as vFx = round(v * fxScale)
//
// Intrinsics operate on fixed-point values and return fixed-point values.
// This makes them composable with PL/0 integer arithmetic (treating those
// integers as fixed-point).

function fxToFloat(xFx, fxScale) {
  return xFx / fxScale;
}

function floatToFx(x, fxScale) {
  // Keep within a safe-ish 32-bit range by default.
  const MAX_FX = 0x7fffffff;
  const v = Math.round(x * fxScale);
  return clamp(v, -MAX_FX, MAX_FX);
}

function refMathFx(op, xFx, fxScale) {
  const x0 = fxToFloat(xFx, fxScale);
  let x = x0;
  let y;
  switch (op) {
    case 'FSIN': x = clamp(x, -Math.PI, Math.PI); y = Math.sin(x); break;
    case 'FCOS': x = clamp(x, -Math.PI, Math.PI); y = Math.cos(x); break;
    case 'FTAN': {
      // tan can explode; clamp output to keep the fixed-point range sane.
      x = clamp(x, -1.3, 1.3);
      const t = Math.tan(x);
      y = clamp(t, -8, 8);
      break;
    }
    case 'FTANH': x = clamp(x, -3, 3); y = Math.tanh(x); break;
    case 'FSINH': x = clamp(x, -3, 3); y = clamp(Math.sinh(x), -8, 8); break;
    case 'FCOSH': x = clamp(x, -3, 3); y = clamp(Math.cosh(x), 0, 10); break;
    case 'FLN': {
      // ln(x) for x<=0 saturates negative.
      if (x <= 0) return floatToFx(-16, fxScale);
      x = clamp(x, 1e-6, 256);
      y = clamp(Math.log(x), -16, 16);
      break;
    }
    case 'FLOG10': {
      if (x <= 0) return floatToFx(-16, fxScale);
      x = clamp(x, 1e-6, 256);
      y = clamp(Math.log10(x), -16, 16);
      break;
    }
    case 'FEXP': {
      // exp grows quickly; clamp input to avoid huge outputs.
      x = clamp(x, -8, 8);
      y = clamp(Math.exp(x), 0, 256);
      break;
    }
    case 'FSQRT': {
      x = clamp(x, 0, 256);
      y = (x <= 0) ? 0 : Math.sqrt(x);
      y = clamp(y, 0, 16);
      break;
    }
    default:
      throw new Error(`refMathFx: unknown op '${op}'`);
  }
  return floatToFx(y, fxScale);
}

class PL0CPU {
  // options:
  //   neuralALU: instance with compute(op, aInt, bInt) -> int
  //   neuralMath: instance with compute(op, xInt) -> int  (for unary ops like FSIN/FLN/...)
  //   fxScale: fixed-point scaling factor for math intrinsics (default 65536)
  //   trackNeuralStats: boolean (default true if neuralALU provided)
  constructor(numRegs = 4, memorySize = 256, dataStackSize = 256, options = {}) {
    this.numRegs = numRegs;
    this.regs = new Array(numRegs).fill(0);
    this.memory = new Array(memorySize).fill(0);

    this.instructions = [];
    this.pointer = 0;
    this.running = false;

    this.labelMap = {};
    this.callStack = [];
    this.dataStack = [];
    this.dataStackMax = dataStackSize;

    // Fixed-point scaling for intrinsic math.
    this.fxScale = Number.isFinite(options.fxScale) ? options.fxScale : 65536;

    this.neuralALU = options.neuralALU ?? null;
    this.neuralMath = options.neuralMath ?? null;
    this.trackNeuralStats = options.trackNeuralStats ?? (!!this.neuralALU || !!this.neuralMath);
    this.neuralStats = {
      enabled: !!this.neuralALU,
      ops: { ADD: 0, SUB: 0, MUL: 0, DIV: 0 },
      fallbacks: { ADD: 0, SUB: 0, MUL: 0, DIV: 0 },
      absErrorSum: { ADD: 0, SUB: 0, MUL: 0, DIV: 0 },
    };

    this.neuralMathStats = null;
    if (this.neuralMath && this.neuralMath.stats) {
      this.neuralMathStats = this.neuralMath.stats;
    }
  }

  addInstruction(line) {
    this.instructions.push(String(line));
  }

  addInstructions(lines) {
    for (const line of lines) this.addInstruction(line);
  }

  buildLabelMap() {
    this.labelMap = {};
    for (let i = 0; i < this.instructions.length; i++) {
      const line = this.instructions[i].trim();
      if (line.endsWith(':')) {
        this.labelMap[line.slice(0, -1)] = i;
      }
    }
  }

  _parseReg(token) {
    // token like r0 or r1,
    const t = token.endsWith(',') ? token.slice(0, -1) : token;
    if (!/^r\d+$/i.test(t)) throw new Error(`Bad register token: ${token}`);
    const idx = parseInt(t.slice(1), 10);
    if (idx < 0 || idx >= this.numRegs) throw new Error(`Register out of range: ${token}`);
    return idx;
  }

  _parseAddrBracket(token) {
    // token like [50] or [r1]
    if (!token.startsWith('[') || !token.endsWith(']')) {
      throw new Error(`Bad address token: ${token}`);
    }
    const inner = token.slice(1, -1).trim();
    if (/^r\d+$/i.test(inner)) {
      const r = this._parseReg(inner);
      return { mode: 'reg', reg: r };
    }
    if (!/^-?\d+$/.test(inner)) throw new Error(`Bad address literal: ${token}`);
    return { mode: 'imm', addr: parseInt(inner, 10) };
  }

  _loadAddr(addrSpec) {
    const addr = addrSpec.mode === 'reg' ? this.regs[addrSpec.reg] : addrSpec.addr;
    const m = this.memory.length;
    // Keep behavior close to the PDF's JS: clamp into memory range.
    // (The PDF uses JS arrays; out-of-range would yield undefined; we guard.)
    const a = ((addr % m) + m) % m;
    return a;
  }

  execute(maxSteps = 1_000_000) {
    this.buildLabelMap();
    this.pointer = 0;
    this.running = true;

    let steps = 0;
    while (this.running && this.pointer < this.instructions.length) {
      if (steps++ > maxSteps) throw new Error(`Execution aborted: exceeded maxSteps=${maxSteps}`);

      let line = this.instructions[this.pointer].trim();
      if (!line) {
        this.pointer++;
        continue;
      }
      if (line.endsWith(':')) {
        this.pointer++;
        continue;
      }

      // Split op + args. This is intentionally simple; the compiler emits tokens we expect.
      const parts = line.split(/\s+/);
      const op = parts[0].toUpperCase();
      const args = parts.slice(1);

      switch (op) {
        case 'LOAD': {
          // LOAD rX, #imm  |  LOAD rX, [addr]  |  LOAD rX, [rY]
          let [regTok, valTok] = args;
          const rX = this._parseReg(regTok);
          if (valTok.startsWith('#')) {
            const imm = parseInt(valTok.slice(1), 10);
            this.regs[rX] = imm;
          } else {
            const addrSpec = this._parseAddrBracket(valTok);
            const a = this._loadAddr(addrSpec);
            this.regs[rX] = this.memory[a] ?? 0;
          }
          this.pointer++;
          break;
        }
        case 'STORE': {
          // STORE rX, [addr] | STORE rX, [rY]
          let [regTok, addrTok] = args;
          const rX = this._parseReg(regTok);
          const addrSpec = this._parseAddrBracket(addrTok);
          const a = this._loadAddr(addrSpec);
          this.memory[a] = this.regs[rX];
          this.pointer++;
          break;
        }
        case 'PEEK': {
          // Same semantics as LOAD rX, [addr]
          let [regTok, addrTok] = args;
          const rX = this._parseReg(regTok);
          const addrSpec = this._parseAddrBracket(addrTok);
          const a = this._loadAddr(addrSpec);
          this.regs[rX] = this.memory[a] ?? 0;
          this.pointer++;
          break;
        }
        case 'POKE': {
          // Same semantics as STORE rX, [addr]
          let [regTok, addrTok] = args;
          const rX = this._parseReg(regTok);
          const addrSpec = this._parseAddrBracket(addrTok);
          const a = this._loadAddr(addrSpec);
          this.memory[a] = this.regs[rX];
          this.pointer++;
          break;
        }
        case 'PUSH': {
          // PUSH rX
          const rX = this._parseReg(args[0]);
          if (this.dataStack.length >= this.dataStackMax) {
            throw new Error('Data stack overflow');
          }
          this.dataStack.push(this.regs[rX]);
          this.pointer++;
          break;
        }
        case 'POP': {
          // POP rX
          const rX = this._parseReg(args[0]);
          if (this.dataStack.length === 0) {
            throw new Error('Data stack underflow');
          }
          this.regs[rX] = this.dataStack.pop();
          this.pointer++;
          break;
        }
        case 'ADD': {
          // ADD rX, rY => rX = rX + rY
          let [rxTok, ryTok] = args;
          const rX = this._parseReg(rxTok);
          const rY = this._parseReg(ryTok);
          if (this.neuralALU) {
            if (typeof this.neuralALU.computeDetailed === 'function') {
              const d = this.neuralALU.computeDetailed('ADD', this.regs[rX], this.regs[rY]);
              this.regs[rX] = d.result;
              if (this.trackNeuralStats) {
                this.neuralStats.ops.ADD++;
                this.neuralStats.absErrorSum.ADD += Math.abs(d.pred - d.exact);
                if (d.usedFallback) this.neuralStats.fallbacks.ADD++;
              }
            } else {
              this.regs[rX] = this.neuralALU.compute('ADD', this.regs[rX], this.regs[rY]);
              if (this.trackNeuralStats) this.neuralStats.ops.ADD++;
            }
          } else {
            this.regs[rX] = this.regs[rX] + this.regs[rY];
          }
          this.pointer++;
          break;
        }
        case 'SUB': {
          let [rxTok, ryTok] = args;
          const rX = this._parseReg(rxTok);
          const rY = this._parseReg(ryTok);
          if (this.neuralALU) {
            if (typeof this.neuralALU.computeDetailed === 'function') {
              const d = this.neuralALU.computeDetailed('SUB', this.regs[rX], this.regs[rY]);
              this.regs[rX] = d.result;
              if (this.trackNeuralStats) {
                this.neuralStats.ops.SUB++;
                this.neuralStats.absErrorSum.SUB += Math.abs(d.pred - d.exact);
                if (d.usedFallback) this.neuralStats.fallbacks.SUB++;
              }
            } else {
              this.regs[rX] = this.neuralALU.compute('SUB', this.regs[rX], this.regs[rY]);
              if (this.trackNeuralStats) this.neuralStats.ops.SUB++;
            }
          } else {
            this.regs[rX] = this.regs[rX] - this.regs[rY];
          }
          this.pointer++;
          break;
        }
        case 'MUL': {
          let [rxTok, ryTok] = args;
          const rX = this._parseReg(rxTok);
          const rY = this._parseReg(ryTok);
          if (this.neuralALU) {
            if (typeof this.neuralALU.computeDetailed === 'function') {
              const d = this.neuralALU.computeDetailed('MUL', this.regs[rX], this.regs[rY]);
              this.regs[rX] = d.result;
              if (this.trackNeuralStats) {
                this.neuralStats.ops.MUL++;
                this.neuralStats.absErrorSum.MUL += Math.abs(d.pred - d.exact);
                if (d.usedFallback) this.neuralStats.fallbacks.MUL++;
              }
            } else {
              this.regs[rX] = this.neuralALU.compute('MUL', this.regs[rX], this.regs[rY]);
              if (this.trackNeuralStats) this.neuralStats.ops.MUL++;
            }
          } else {
            this.regs[rX] = this.regs[rX] * this.regs[rY];
          }
          this.pointer++;
          break;
        }
        case 'DIV': {
          let [rxTok, ryTok] = args;
          const rX = this._parseReg(rxTok);
          const rY = this._parseReg(ryTok);
          if (this.regs[rY] === 0) throw new Error('Division by zero');
          if (this.neuralALU) {
            if (typeof this.neuralALU.computeDetailed === 'function') {
              const d = this.neuralALU.computeDetailed('DIV', this.regs[rX], this.regs[rY]);
              this.regs[rX] = d.result;
              if (this.trackNeuralStats) {
                this.neuralStats.ops.DIV++;
                this.neuralStats.absErrorSum.DIV += Math.abs(d.pred - d.exact);
                if (d.usedFallback) this.neuralStats.fallbacks.DIV++;
              }
            } else {
              this.regs[rX] = this.neuralALU.compute('DIV', this.regs[rX], this.regs[rY]);
              if (this.trackNeuralStats) this.neuralStats.ops.DIV++;
            }
          } else {
            this.regs[rX] = Math.floor(this.regs[rX] / this.regs[rY]);
          }
          this.pointer++;
          break;
        }

        // --- Neural/NARX unary math ops (optional extensions) ---
        // These are not part of the original PDF instruction set, but let the
        // PL/0 compiler emit "intrinsics" that map to neural coprocessor ops.
        case 'FSIN':
        case 'FCOS':
        case 'FTAN':
        case 'FTANH':
        case 'FSINH':
        case 'FCOSH':
        case 'FLN':
        case 'FLOG10':
        case 'FEXP':
        case 'FSQRT': {
          const rX = this._parseReg(args[0]);
          const opName = op; // already uppercased
          if (this.neuralMath) {
            if (typeof this.neuralMath.computeDetailed === 'function') {
              const d = this.neuralMath.computeDetailed(opName, this.regs[rX]);
              this.regs[rX] = d.result;
            } else {
              this.regs[rX] = this.neuralMath.compute(opName, this.regs[rX]);
            }
          } else {
            // Deterministic fallback
            this.regs[rX] = refMathFx(opName, this.regs[rX], this.fxScale);
          }
          this.pointer++;
          break;
        }
        case 'JMP': {
          const label = args[0];
          if (!(label in this.labelMap)) throw new Error(`Unknown label: ${label}`);
          this.pointer = this.labelMap[label];
          break;
        }
        case 'JZ': {
          // JZ rX, label
          let [rxTok, label] = args;
          const rX = this._parseReg(rxTok);
          if (!(label in this.labelMap)) throw new Error(`Unknown label: ${label}`);
          this.pointer = (this.regs[rX] === 0) ? this.labelMap[label] : (this.pointer + 1);
          break;
        }
        case 'JNZ': {
          // JNZ rX, label
          let [rxTok, label] = args;
          const rX = this._parseReg(rxTok);
          if (!(label in this.labelMap)) throw new Error(`Unknown label: ${label}`);
          this.pointer = (this.regs[rX] !== 0) ? this.labelMap[label] : (this.pointer + 1);
          break;
        }
        case 'CALL': {
          // CALL label
          const label = args[0];
          if (!(label in this.labelMap)) throw new Error(`Unknown label: ${label}`);
          this.callStack.push(this.pointer + 1);
          this.pointer = this.labelMap[label];
          break;
        }
        case 'PL0CALL': {
          // PL0CALL programName
          let programName = args[0];
          if (programName.endsWith(',')) programName = programName.slice(0, -1);
          const newInstrs = PL0Programs[programName];
          if (!newInstrs) throw new Error(`No compiled PL/0 program named: ${programName}`);
          // Save current context
          this.callStack.push({
            instructions: this.instructions,
            labelMap: this.labelMap,
            returnPointer: this.pointer + 1,
          });
          // Switch to callee
          this.instructions = newInstrs;
          this.buildLabelMap();
          this.pointer = 0;
          break;
        }
        case 'RET': {
          if (this.callStack.length === 0) {
            this.running = false;
            break;
          }
          const top = this.callStack.pop();
          if (typeof top === 'number') {
            this.pointer = top;
          } else {
            this.instructions = top.instructions;
            this.labelMap = top.labelMap;
            this.pointer = top.returnPointer;
          }
          break;
        }
        case 'HALT': {
          this.running = false;
          this.pointer++;
          break;
        }
        default:
          throw new Error(`Unknown instruction '${op}' at line ${this.pointer}: ${line}`);
      }
    }
  }
}


// -----------------------------------------------------------------------------
// PL/0 tokenizer + compiler (taken from the PDF listing, with tiny cleanups)
// -----------------------------------------------------------------------------

function tokenize(input) {
  // Remove single-line comments (// ...)
  input = input.replace(/\/\/.*$/gm, '');

  const tokens = [];
  const keywords = new Set([
    'program', 'var', 'begin', 'end', 'call', 'if', 'then', 'while', 'do', 'odd',
    'push', 'pop', 'peek', 'poke',
  ]);

  // Symbols supported by the PDF-style PL/0 subset (+ our extensions).
  const oneCharSymbols = new Set(['+', '-', '*', '/', '(', ')', ',', ';', '.', '=',]);
  // Two-character symbols:
  //   ':=' assignment
  const isAlpha = (c) => /[A-Za-z]/.test(c);
  const isDigit = (c) => /[0-9]/.test(c);
  const isAlnum = (c) => /[A-Za-z0-9_]/.test(c);

  let i = 0;
  while (i < input.length) {
    const c = input[i];

    // whitespace
    if (/\s/.test(c)) { i++; continue; }

    // ':='
    if (c === ':' && input[i + 1] === '=') {
      tokens.push({ type: 'symbol', value: ':=' });
      i += 2;
      continue;
    }

    // number literal: int | float | scientific (e.g. 1.5, 2e-3)
    if (isDigit(c)) {
      let j = i;
      while (j < input.length && isDigit(input[j])) j++;

      let isFloat = false;

      // decimal fraction
      if (input[j] === '.' && isDigit(input[j + 1])) {
        isFloat = true;
        j++; // consume '.'
        while (j < input.length && isDigit(input[j])) j++;
      }

      // exponent
      if (input[j] === 'e' || input[j] === 'E') {
        const s = input[j + 1];
        const s2 = input[j + 2];
        if (isDigit(s) || ((s === '+' || s === '-') && isDigit(s2))) {
          isFloat = true;
          j++; // consume e/E
          if (input[j] === '+' || input[j] === '-') j++;
          while (j < input.length && isDigit(input[j])) j++;
        }
      }

      const raw = input.slice(i, j);
      if (isFloat) {
        const value = Number(raw);
        if (!Number.isFinite(value)) throw new Error(`Bad float literal: ${raw}`);
        tokens.push({ type: 'float', value, raw });
      } else {
        tokens.push({ type: 'number', value: parseInt(raw, 10), raw });
      }
      i = j;
      continue;
    }

    // identifier / keyword
    if (isAlpha(c)) {
      let j = i + 1;
      while (j < input.length && isAlnum(input[j])) j++;
      const raw = input.slice(i, j);
      const low = raw.toLowerCase();
      if (keywords.has(low)) tokens.push({ type: 'keyword', value: low });
      else tokens.push({ type: 'ident', value: raw });
      i = j;
      continue;
    }

    // one-char symbol
    if (oneCharSymbols.has(c)) {
      tokens.push({ type: 'symbol', value: c });
      i++;
      continue;
    }

    throw new Error(`Unknown character '${c}' at index ${i}`);
  }

  return tokens;
}

class PL0Parser {
  constructor(tokens, options = {}) {
    this.tokens = tokens;
    this.pos = 0;
    this.varTable = new Map();
    this.nextVarAddr = 0;
    this.labelCounter = 100;
    this.tempAddr = 254;

    // Fixed-point scale used for sugar literals (e.g. 1.5, pi)
    // and helper conversions fx(...)/int(...).
    this.fxScale = Number.isFinite(options.fxScale) ? options.fxScale : 65536;

    // Built-in real constants (encoded as fixed-point when used as factors)
    // NOTE: variables with the same name override these.
    this.consts = {
      pi: Math.PI,
      tau: Math.PI * 2,
      e: Math.E,
    };

    // Unary math intrinsics (extension): ident '(' expr ')'
    // These compile into extended CPU ops (FSIN/FLN/...)
    this.intrinsics = {
      sin: 'FSIN',
      cos: 'FCOS',
      tan: 'FTAN',
      tanh: 'FTANH',
      sinh: 'FSINH',
      cosh: 'FCOSH',
      ln: 'FLN',
      log: 'FLOG10',
      log10: 'FLOG10',
      exp: 'FEXP',
      sqrt: 'FSQRT',
    };
  }

  currentToken() {
    return this.tokens[this.pos] || { type: 'EOF', value: '' };
  }

  eat(expected) {
    const token = this.currentToken();
    if (token.value === expected || token.type === expected) {
      this.pos++;
      return token;
    }
    throw new Error(`Parse error: expected ${expected}, got ${token.value} at pos=${this.pos}`);
  }

  newLabel() {
    const label = `label_${this.labelCounter}`;
    this.labelCounter++;
    return label;
  }

  newTemp() {
    const t = this.tempAddr;
    this.tempAddr--;
    return t;
  }

  declareVar(ident) {
    if (this.varTable.has(ident)) throw new Error(`Variable '${ident}' already declared`);
    const addr = this.nextVarAddr;
    this.nextVarAddr++;
    this.varTable.set(ident, addr);
    return addr;
  }

  getVarAddr(ident) {
    if (!this.varTable.has(ident)) throw new Error(`Unknown variable '${ident}'`);
    return this.varTable.get(ident);
  }

  // program -> "program" ident ";" block "."
  parseProgram() {
    this.eat('program');
    const progNameToken = this.currentToken();
    this.eat('ident');
    this.eat(';');
    const [blockAST, blockCode] = this.parseBlock();
    this.eat('.');
    const programAST = { type: 'program', name: progNameToken.value, block: blockAST };
    return [programAST, blockCode];
  }

  // block -> varDecl? statement
  parseBlock() {
    let varDecls = [];
    let codeVars = [];
    if (this.currentToken().value === 'var') {
      [varDecls, codeVars] = this.parseVarDecl();
    }
    const [stmtAST, stmtCode] = this.parseStatement();
    const blockAST = { type: 'block', varDecls, statement: stmtAST };
    const blockCode = codeVars.concat(stmtCode);
    return [blockAST, blockCode];
  }

  // varDecl -> "var" ident {"," ident} ";"
  parseVarDecl() {
    this.eat('var');
    const decls = [];
    while (true) {
      const idToken = this.currentToken();
      this.eat('ident');
      const addr = this.declareVar(idToken.value);
      decls.push({ type: 'varDecl', ident: idToken.value, addr });
      if (this.currentToken().value === ',') {
        this.eat(',');
      } else {
        break;
      }
    }
    this.eat(';');
    return [decls, []];
  }

  // statement -> assignment | callStmt | ifStmt | whileStmt | compoundStmt | pushStmt | popStmt |
  //              peekStmt | pokeStmt | (empty)
  parseStatement() {
    const tk = this.currentToken();
    if (tk.value === 'call') return this.parseCallStatement();
    if (tk.value === 'if') return this.parseIfStatement();
    if (tk.value === 'while') return this.parseWhileStatement();
    if (tk.value === 'begin') return this.parseCompoundStatement();
    if (tk.value === 'push') return this.parsePushStatement();
    if (tk.value === 'pop') return this.parsePopStatement();
    if (tk.value === 'peek') return this.parsePeekStatement();
    if (tk.value === 'poke') return this.parsePokeStatement();
    if (tk.type === 'ident') return this.parseAssignment();
    return [{ type: 'noop' }, []];
  }

  // callStmt -> "call" ident ";"
  parseCallStatement() {
    this.eat('call');
    const idToken = this.currentToken();
    this.eat('ident');
    this.eat(';');
    const ast = { type: 'call', ident: idToken.value };
    const code = [`PL0CALL ${idToken.value}`];
    return [ast, code];
  }

  // assignment -> ident ":=" expression ";"
  parseAssignment() {
    const idToken = this.currentToken();
    this.eat('ident');
    this.eat(':=');
    const [exprAST, exprCode] = this.parseExpression();
    this.eat(';');
    const addr = this.getVarAddr(idToken.value);
    const code = [...exprCode, `STORE r0, [${addr}]`];
    const ast = { type: 'assign', ident: idToken.value, expr: exprAST };
    return [ast, code];
  }

  // pushStmt -> "push" ident ";"
  parsePushStatement() {
    this.eat('push');
    const idToken = this.currentToken();
    this.eat('ident');
    this.eat(';');
    const addr = this.getVarAddr(idToken.value);
    const code = [`LOAD r0, [${addr}]`, 'PUSH r0'];
    const ast = { type: 'push', ident: idToken.value };
    return [ast, code];
  }

  // popStmt -> "pop" ident ";"
  parsePopStatement() {
    this.eat('pop');
    const idToken = this.currentToken();
    this.eat('ident');
    this.eat(';');
    const addr = this.getVarAddr(idToken.value);
    const code = ['POP r0', `STORE r0, [${addr}]`];
    const ast = { type: 'pop', ident: idToken.value };
    return [ast, code];
  }

  // peekStmt -> "peek" "(" ident "," ident ")" ";"
  parsePeekStatement() {
    this.eat('peek');
    this.eat('(');
    const destToken = this.currentToken();
    this.eat('ident');
    this.eat(',');
    const addrToken = this.currentToken();
    this.eat('ident');
    this.eat(')');
    this.eat(';');
    const destAddr = this.getVarAddr(destToken.value);
    const addrVarAddr = this.getVarAddr(addrToken.value);
    const code = [
      `LOAD r0, [${addrVarAddr}]`,
      'PEEK r1, [r0]',
      `STORE r1, [${destAddr}]`,
    ];
    const ast = { type: 'peek', dest: destToken.value, addr: addrToken.value };
    return [ast, code];
  }

  // pokeStmt -> "poke" "(" ident "," ident ")" ";"
  parsePokeStatement() {
    this.eat('poke');
    this.eat('(');
    const addrToken = this.currentToken();
    this.eat('ident');
    this.eat(',');
    const valToken = this.currentToken();
    this.eat('ident');
    this.eat(')');
    this.eat(';');
    const addrVarAddr = this.getVarAddr(addrToken.value);
    const valAddr = this.getVarAddr(valToken.value);
    const code = [
      `LOAD r0, [${addrVarAddr}]`,
      `LOAD r1, [${valAddr}]`,
      'POKE r1, [r0]',
    ];
    const ast = { type: 'poke', addr: addrToken.value, val: valToken.value };
    return [ast, code];
  }

  // ifStmt -> "if" expression "then" statement
  parseIfStatement() {
    this.eat('if');
    const [condAST, condCode] = this.parseExpression();
    this.eat('then');
    const [thenAST, thenCode] = this.parseStatement();
    const skipLabel = this.newLabel();
    const code = [...condCode, `JZ r0, ${skipLabel}`, ...thenCode, `${skipLabel}:`];
    const ast = { type: 'if', condition: condAST, thenPart: thenAST };
    return [ast, code];
  }

  // whileStmt -> "while" expression "do" statement
  parseWhileStatement() {
    this.eat('while');
    const startLabel = this.newLabel();
    const exitLabel = this.newLabel();
    const loopStart = `${startLabel}:`;
    const [condAST, condCode] = this.parseExpression();
    this.eat('do');
    const [bodyAST, bodyCode] = this.parseStatement();
    const code = [
      loopStart,
      ...condCode,
      `JZ r0, ${exitLabel}`,
      ...bodyCode,
      `JMP ${startLabel}`,
      `${exitLabel}:`,
    ];
    const ast = { type: 'while', condition: condAST, body: bodyAST };
    return [ast, code];
  }

  // compoundStmt -> "begin" statement { ";" statement } "end"
  parseCompoundStatement() {
    this.eat('begin');
    const stmts = [];
    const codeAll = [];
    while (this.currentToken().value !== 'end') {
      const [stmtAST, stmtCode] = this.parseStatement();
      stmts.push(stmtAST);
      codeAll.push(...stmtCode);
      if (this.currentToken().value === ';') this.eat(';');
    }
    this.eat('end');
    const ast = { type: 'compound', statements: stmts };
    return [ast, codeAll];
  }

  // expression -> term { (+|-) term }
  parseExpression() {
    let [leftAST, leftCode] = this.parseTerm();
    while (this.currentToken().value === '+' || this.currentToken().value === '-') {
      const opToken = this.currentToken().value;
      this.eat(opToken);
      const [rightAST, rightCode] = this.parseTerm();
      const binAST = { type: 'binop', op: opToken, left: leftAST, right: rightAST };

      const tempAddr = this.newTemp();
      const storeLeft = `STORE r0, [${tempAddr}]`;
      const loadLeftIntoR1 = `LOAD r1, [${tempAddr}]`;
      // NOTE: r0 holds the *right* value after rightCode, r1 holds the *left*.
      // ADD is commutative, so `ADD r0, r1` is fine.
      // SUB is not, so compute left-right into r1, then move back into r0.
      let code;
      if (opToken === '+') {
        code = [...leftCode, storeLeft, ...rightCode, loadLeftIntoR1, 'ADD r0, r1'];
      } else {
        code = [
          ...leftCode,
          storeLeft,
          ...rightCode,
          loadLeftIntoR1,
          'SUB r1, r0',
          `STORE r1, [${tempAddr}]`,
          `LOAD r0, [${tempAddr}]`,
        ];
      }

      leftAST = binAST;
      leftCode = code;
    }
    return [leftAST, leftCode];
  }

  // term -> factor { (*|/) factor }
  parseTerm() {
    let [leftAST, leftCode] = this.parseFactor();
    while (this.currentToken().value === '*' || this.currentToken().value === '/') {
      const opToken = this.currentToken().value;
      this.eat(opToken);
      const [rightAST, rightCode] = this.parseFactor();
      const binAST = { type: 'binop', op: opToken, left: leftAST, right: rightAST };

      const tempAddr = this.newTemp();
      const storeLeft = `STORE r0, [${tempAddr}]`;
      const loadLeftIntoR1 = `LOAD r1, [${tempAddr}]`;
      // NOTE: r0 holds the *right* value after rightCode, r1 holds the *left*.
      // MUL is commutative, so `MUL r0, r1` is fine.
      // DIV is not, so compute left/right into r1, then move back into r0.
      let code;
      if (opToken === '*') {
        code = [...leftCode, storeLeft, ...rightCode, loadLeftIntoR1, 'MUL r0, r1'];
      } else {
        code = [
          ...leftCode,
          storeLeft,
          ...rightCode,
          loadLeftIntoR1,
          'DIV r1, r0',
          `STORE r1, [${tempAddr}]`,
          `LOAD r0, [${tempAddr}]`,
        ];
      }

      leftAST = binAST;
      leftCode = code;
    }
    return [leftAST, leftCode];
  }

  // factor -> number | float | ident | "(" expression ")"
//
// Fixed-point sugar:
//   - Float literals (e.g. 1.5, 0.25, 2e-3) compile to round(value * fxScale)
//   - Built-in constants: pi, tau, e
//   - Helper conversions:
//       fx(expr)  => expr * fxScale
//       int(expr) => floor(expr / fxScale)
//
// Note: integer literals remain unscaled for backwards-compatibility with the
// PDF-style integer PL/0 programs (e.g. matrixTest).
parseFactor() {
  const tk = this.currentToken();

  if (tk.type === 'number') {
    this.eat('number');
    const code = [`LOAD r0, #${tk.value}`];
    return [{ type: 'num', value: tk.value }, code];
  }

  if (tk.type === 'float') {
    this.eat('float');
    const scaled = floatToFx(tk.value, this.fxScale);
    const code = [`LOAD r0, #${scaled}`];
    return [{ type: 'num', value: scaled, raw: tk.raw, kind: 'fixed' }, code];
  }

  if (tk.type === 'ident') {
    // Either a variable, a constant, or a call-like intrinsic: name '(' expression ')'
    const name = tk.value;
    const nameLower = String(name).toLowerCase();
    const next = this.tokens[this.pos + 1] || { type: 'EOF', value: '' };

    if (next.value === '(') {
      this.eat('ident');
      this.eat('(');
      const [exprAST, exprCode] = this.parseExpression();
      this.eat(')');

      // Unary math intrinsics
      const op = this.intrinsics[nameLower];
      if (op) {
        const code = [
          ...exprCode,
          `${op} r0`,
        ];
        return [{ type: 'intrinsic', name, arg: exprAST }, code];
      }

      // Fixed-point helpers
      if (nameLower === 'fx' || nameLower === 'tofx') {
        const tempAddr = this.newTemp();
        const code = [
          ...exprCode,
          `STORE r0, [${tempAddr}]`,
          `LOAD r0, #${this.fxScale}`,
          `LOAD r1, [${tempAddr}]`,
          `MUL r0, r1`,
        ];
        return [{ type: 'fx', arg: exprAST }, code];
      }

      if (nameLower === 'int' || nameLower === 'fromfx' || nameLower === 'unfx') {
        const code = [
          ...exprCode,
          `LOAD r1, #${this.fxScale}`,
          `DIV r0, r1`,
        ];
        return [{ type: 'int', arg: exprAST }, code];
      }

      throw new Error(`Unknown intrinsic '${name}(... )'. Supported: ${Object.keys(this.intrinsics).concat(['fx','int']).join(', ')}`);
    }

    // variable or constant
    this.eat('ident');

    if (this.varTable.has(name)) {
      const addr = this.getVarAddr(name);
      const code = [`LOAD r0, [${addr}]`];
      return [{ type: 'var', name }, code];
    }

    if (Object.prototype.hasOwnProperty.call(this.consts, nameLower)) {
      const scaled = floatToFx(this.consts[nameLower], this.fxScale);
      const code = [`LOAD r0, #${scaled}`];
      return [{ type: 'const', name: nameLower, value: scaled }, code];
    }

    throw new Error(`Unknown variable '${name}'`);
  }

  if (tk.value === '(') {
    this.eat('(');
    const [exprAST, exprCode] = this.parseExpression();
    this.eat(')');
    return [exprAST, exprCode];
  }
  throw new Error('Unexpected token in factor: ' + JSON.stringify(tk));
}
}


// Global storage for compiled PL/0 programs
const PL0Programs = {};

function compilePL0(programText, baseAddr = 0, options = {}) {
  const tokens = tokenize(programText);
  const parser = new PL0Parser(tokens, { fxScale: options.fxScale });
  if (baseAddr !== 0) parser.nextVarAddr = baseAddr;
  const [ast, code] = parser.parseProgram();
  code.push('RET');
  PL0Programs[ast.name] = code;
  return code;
}


// -----------------------------------------------------------------------------
// Demo program (same structure as the PDF listing)
// -----------------------------------------------------------------------------

const setElementSource = `
program setElement;
var row, col, width, val, offset;
begin
 pop val;
 pop width;
 pop col;
 pop row;
 offset := row * width + col;
 offset := offset + 30;
 poke(offset, val);
end.
`;

const getElementSource = `
program getElement;
var row, col, width, offset, value;
begin
 pop width;
 pop col;
 pop row;
 offset := row * width + col;
 offset := offset + 30;
 peek(value, offset);
 push value;
end.
`;

const matrixTestSource = `
program matrixTest;
var width, row, col, val;
begin
 width := 4;
 row := 2;
 col := 3;
 val := 99;

 push row;
 push col;
 push width;
 push val;
 call setElement;

 push row;
 push col;
 push width;
 call getElement;
end.
`;

// Demo that exercises the intrinsic unary math extension.
// NOTE: values are fixed-point integers (default Q16.16, scale=65536).
const mathTestSource = `
program mathTest;
var a, b, c, d, e, f;
begin
  // Float literals are fixed-point (scaled by fxScale).
  // Built-in constants: pi, tau, e
  // Helpers: fx(x) => x*fxScale, int(x) => floor(x/fxScale)

  a := pi;        // pi in radians (fixed-point)
  b := sin(a);    // ~0.0

  c := cos(0.0);  // 1.0

  d := ln(1.0);   // 0.0
  e := sqrt(2.0); // ~1.4142

  f := int(c);    // 1

  push b;
  push c;
  push d;
  push e;
  push f;
end.
`;

function main() {
  const argv = process.argv.slice(2);

  // Fixed-point scale for intrinsic math AND for compile-time float literals/constants.
  const fxArg = argv.find((a) => a.startsWith('--fxScale='));
  const fxScale = fxArg ? Number(fxArg.split('=')[1]) : 65536;

  // Compile demo programs using the same fxScale the CPU will use.
  compilePL0(setElementSource, 0, { fxScale });
  compilePL0(getElementSource, 0, { fxScale });
  compilePL0(matrixTestSource, 20, { fxScale });
  compilePL0(mathTestSource, 40, { fxScale });
  const useNeural = argv.includes('--neural');
  const useNarxMath = argv.includes('--narx-math');
  const progArg = argv.find((a) => a.startsWith('--program='));
  const programName = progArg ? progArg.split('=')[1] : 'matrixTest';
  const mixArg = argv.find((a) => a.startsWith('--mix='));
  const mix = mixArg ? Number(mixArg.split('=')[1]) : 1.0;


  let alu = null;
  let math = null;

  if (useNeural) {
    const { NeuralALU } = require('./neural_alu');
    const useFallback = !argv.includes('--no-fallback');
    const fbArg = argv.find((a) => a.startsWith('--fallbackAbs='));
    const fallbackAbsError = fbArg ? Number(fbArg.split('=')[1]) : 2;
    const archArg = argv.find((a) => a.startsWith('--alu-arch='));
    const arch = archArg ? archArg.split('=')[1] : 'linear';
    const scaleArg = argv.find((a) => a.startsWith('--scale='));
    const scale = scaleArg ? Number(scaleArg.split('=')[1]) : 65536;

    alu = new NeuralALU({
      architecture: arch,
      hidden: 32,
      scale: Number.isFinite(scale) ? scale : 65536,
      mix: Number.isFinite(mix) ? mix : 1.0,
      safetyFallback: useFallback,
      fallbackAbsError: Number.isFinite(fallbackAbsError) ? fallbackAbsError : 2,
      // Match the manual PDF semantics for DIV.
      divDecode: 'floor',
    });

    // If you use the MLP architecture (or want to refine DIV for the linear one), train.
    if (arch !== 'linear' || argv.includes('--train')) {
      const epochsArg = argv.find((a) => a.startsWith('--epochs='));
      const stepsArg = argv.find((a) => a.startsWith('--steps='));
      const lrArg = argv.find((a) => a.startsWith('--lr='));
      const epochs = epochsArg ? Number(epochsArg.split('=')[1]) : 10;
      const stepsPerEpoch = stepsArg ? Number(stepsArg.split('=')[1]) : 3000;
      const lr = lrArg ? Number(lrArg.split('=')[1]) : 0.01;
      alu.trainQuick({ epochs, stepsPerEpoch, lr, operandRange: 256, mulRange: 255 });
    }
  }

  if (useNarxMath) {
    const { NeuralMathNARX } = require('./neural_math_narx');
    const mmArg = argv.find((a) => a.startsWith('--mathMix='));
    const mathMix = mmArg ? Number(mmArg.split('=')[1]) : 1.0;
    const useMathFallback = !argv.includes('--no-math-fallback');
    const thArg = argv.find((a) => a.startsWith('--mathFallbackAbs='));
    const thr = thArg ? Number(thArg.split('=')[1]) : 0.001;
    math = new NeuralMathNARX({
      scale: Number.isFinite(fxScale) ? fxScale : 65536,
      mix: Number.isFinite(mathMix) ? mathMix : 1.0,
      safetyFallback: useMathFallback,
      fallbackAbsError: Number.isFinite(thr) ? thr : 0.001,
    });

    // Optional supervised pre-training to make "pure neural" less random.
    // Usage: --train-math --mathEpochs=80 --mathLen=6000 --mathLr=0.02
    if (argv.includes('--train-math')) {
      const meArg = argv.find((a) => a.startsWith('--mathEpochs='));
      const mlArg = argv.find((a) => a.startsWith('--mathLen='));
      const mLrArg = argv.find((a) => a.startsWith('--mathLr='));
      const mathEpochs = meArg ? Number(meArg.split('=')[1]) : 80;
      const mathLen = mlArg ? Number(mlArg.split('=')[1]) : 6000;
      const mathLr = mLrArg ? Number(mLrArg.split('=')[1]) : 0.02;
      math.trainQuick({ epochs: mathEpochs, length: mathLen, learningRate: mathLr });
    }
  }

  const cpu = new PL0CPU(4, 256, 256, {
    neuralALU: alu,
    neuralMath: math,
    fxScale: Number.isFinite(fxScale) ? fxScale : 65536,
  });
  cpu.addInstructions([
    `PL0CALL ${programName}`,
    'HALT',
  ]);
  cpu.execute();

  console.log('Final memory state (addresses 30-42):', cpu.memory.slice(30, 42));
  console.log('Final data stack:', cpu.dataStack);
  if (cpu.neuralStats?.enabled) {
    console.log('Neural ALU ops executed:', cpu.neuralStats.ops);
    console.log('Neural ALU fallbacks:', cpu.neuralStats.fallbacks);
    const avgAbs = {};
    for (const k of Object.keys(cpu.neuralStats.ops)) {
      const n = cpu.neuralStats.ops[k] || 0;
      avgAbs[k] = n ? (cpu.neuralStats.absErrorSum[k] / n) : 0;
    }
    console.log('Neural ALU avg |pred-exact|:', avgAbs);
  }

  if (cpu.neuralMathStats?.enabled) {
    console.log('Neural NARX-math calls:', cpu.neuralMathStats.calls);
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

// Allow reuse as a module (e.g. for tests/experiments).
module.exports = { PL0CPU, tokenize, PL0Parser, compilePL0, PL0Programs };
