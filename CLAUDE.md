# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational ISA (Instruction Set Architecture) project for a university course on Computer Systems. Implements a register-based 2-operand ISA with a sequential VM interpreter and a 5-stage pipelined simulator with forwarding (variant B).

## Commands

```bash
# Run pipeline simulator with all 8 test cases (sequential + pipelined, verification)
python3 pipeline.py

# Run with per-cycle debug trace of pipeline stages
python3 pipeline.py --debug

# Run sequential VM demo (array sum)
python3 ISA.py
```

No external dependencies — pure Python 3.7+ stdlib only.

## Architecture

**ISA.py** — Reference sequential VM. Contains all shared types:
- `VM` class: architectural state (PC, R0-R7, Z/N flags, MEM[4096]), two-pass assembler (`load_program`), step-by-step execution (`step`/`run`)
- `Instr`/`Operand` dataclasses: decoded instruction representation
- `to_int32`, `idiv_trunc0`: arithmetic helpers
- 14 opcodes: HALT, NOP, MOV, LD, ST, ADD, SUB, MUL, DIV, CMP, JMP, JZ, JNZ, JN, JNN
- 5 addressing modes: REG, IMM, MEM_DIR, MEM_REG, MEM_REG_OFF

**pipeline.py** — 5-stage pipelined simulator (IF→ID→EX→MEM→WB). Imports from ISA.py:
- `InstrMeta`: per-instruction metadata (R(I)/W(I) sets, result_ready stage) computed by `compute_meta()`
- `PipeReg`: inter-stage pipeline register carrying intermediate values
- `PipelinedVM`: ticked simulator with forwarding (EX→ID, MEM→ID), load-use stall detection, branch flush (predict not-taken, resolve at EX)
- Main loop processes stages in reverse order (WB→MEM→EX→ID→IF) so forwarding values are available
- Test program: count positive elements in array (task 8)

**README.md** — Full ISA specification (Part A: architecture, Part B: interpreter, Part C: pipeline model).

## Key Design Details

- Harvard architecture: program memory (`prog[]`) separate from data memory (`MEM[]`) — no structural hazards between IF and MEM
- Forwarding uses freshly computed stage outputs (`new_ex_mem`, `new_mem_wb`), not stale pipeline registers
- Instructions with memory-source operands (e.g., `ADD R1, [R2+5]`) have `result_ready="MEM"` — address computed in EX, memory read + ALU in MEM
- WB commits first each cycle, so register file reads in ID see committed values; forwarding overrides when newer values exist in EX/MEM
