import re
from dataclasses import dataclass
from typing import List, Optional


def to_int32(x: int) -> int:
    x &= 0xFFFFFFFF
    return x - 0x100000000 if x & 0x80000000 else x


def idiv_trunc0(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError
    sign = -1 if (a < 0) ^ (b < 0) else 1
    return sign * (abs(a) // abs(b))


class VMError(Exception):
    def __init__(self, code: str, pc: int, msg: str):
        super().__init__(f"{code} at PC={pc}: {msg}")
        self.code, self.pc, self.msg = code, pc, msg


@dataclass
class Operand:
    kind: str  # REG, IMM, MEM_DIR, MEM_REG, MEM_REG_OFF
    reg: Optional[int] = None
    imm: Optional[int] = None
    addr: Optional[int] = None


@dataclass
class Instr:
    op: str
    a: Optional[Operand] = None
    b: Optional[Operand] = None
    raw: str = ""


class VM:
    def __init__(self, mem_size: int = 4096, debug: bool = False):
        self.MEM_SIZE = mem_size
        self.debug = debug
        self.reset()

    def reset(self):
        self.PC = 0
        self.R = [0] * 8
        self.Z = 0
        self.N = 0
        self.MEM = [0] * self.MEM_SIZE
        self.prog: List[Instr] = []

    # ---------- parsing ----------

    def _parse_int(self, s: str) -> int:
        s = s.strip()
        neg = False
        if s.startswith("-"):
            neg, s = True, s[1:]
        base = 16 if s.lower().startswith("0x") else 10
        if base == 16:
            s = s[2:]
        if not s:
            raise ValueError("empty int literal")
        v = int(s, base)
        return -v if neg else v

    def _parse_operand(self, tok: str, pc_for_err: int) -> Operand:
        tok = tok.strip()
        # REG
        tok_low = tok.lower()
        if tok_low.startswith("r") and tok_low[1:].isdigit():
            r = int(tok_low[1:])
            if not (0 <= r <= 7):
                raise VMError("E_BAD_REG", pc_for_err, f"register {tok}")
            return Operand("REG", reg=r)
        # IMM
        if tok.startswith("#"):
            return Operand("IMM", imm=to_int32(self._parse_int(tok[1:])))
        # MEM forms: [addr]  [Rk]  [Rk+imm]  [Rk-imm]
        if tok.startswith("[") and tok.endswith("]"):
            inner = tok[1:-1].strip()
            low = inner.lower()
            # [Rk+imm] / [Rk-imm]
            m = re.match(r"^(r[0-7])\s*([+-])\s*(.+)$", low)
            if m:
                r = int(m.group(1)[1:])
                sign = m.group(2)
                off = self._parse_int(m.group(3))
                if sign == "-":
                    off = -off
                return Operand("MEM_REG_OFF", reg=r, imm=to_int32(off))
            # [Rk]
            if low.startswith("r") and low[1:].isdigit():
                r = int(low[1:])
                if not (0 <= r <= 7):
                    raise VMError("E_BAD_REG", pc_for_err, f"register {inner}")
                return Operand("MEM_REG", reg=r)
            # [addr]
            return Operand("MEM_DIR", addr=self._parse_int(inner))
        raise ValueError(f"unknown operand '{tok}'")

    def load_program(self, text: str):
        lines = text.splitlines()
        cleaned = []
        labels = {}
        pc = 0

        # Pass 1: strip comments, collect labels, build cleaned instruction lines
        for line in lines:
            if ";" in line:
                line = line.split(";", 1)[0]
            line = line.strip()
            if not line:
                continue
            # Allow "label: instr" or "label:" alone; multiple labels on one line
            while True:
                if ":" in line:
                    before, after = line.split(":", 1)
                    lab = before.strip()
                    if lab and (lab[0].isalpha() or lab[0] == "_") and all(
                        ch.isalnum() or ch == "_" for ch in lab
                    ):
                        if lab in labels:
                            raise VMError("E_DECODE", pc, f"duplicate label {lab}")
                        labels[lab] = pc
                        line = after.strip()
                        if not line:
                            break
                        continue
                break
            if not line:
                continue
            cleaned.append(line)
            pc += 1

        # Pass 2: parse instructions
        prog: List[Instr] = []
        for i, line in enumerate(cleaned):
            parts = line.split(None, 1)
            op = parts[0].upper()
            rest = parts[1].strip() if len(parts) > 1 else ""
            a = b = None

            if op in ("HALT", "NOP"):
                if rest:
                    raise VMError("E_DECODE", i, f"unexpected operands: {rest}")

            elif op in ("JMP", "JZ", "JNZ", "JN", "JNN"):
                if not rest:
                    raise VMError("E_DECODE", i, "missing label")
                if rest not in labels:
                    raise VMError("E_BAD_JUMP", i, f"unknown label {rest}")
                a = Operand("IMM", imm=labels[rest])  # target stored as instruction index

            else:
                if not rest:
                    raise VMError("E_DECODE", i, "missing operands")
                toks = [t.strip() for t in rest.split(",")]
                if len(toks) != 2:
                    raise VMError("E_DECODE", i, f"expected 2 operands, got {len(toks)}")
                try:
                    a = self._parse_operand(toks[0], i)
                    b = self._parse_operand(toks[1], i)
                except VMError:
                    raise
                except Exception as e:
                    raise VMError("E_DECODE", i, str(e))

            prog.append(Instr(op=op, a=a, b=b, raw=line))

        self.prog = prog
        self.PC = 0

    # ---------- execution ----------

    def _addr(self, op: Operand, pc: int) -> int:
        if op.kind == "MEM_DIR":
            addr = op.addr
        elif op.kind == "MEM_REG":
            addr = self.R[op.reg]
        elif op.kind == "MEM_REG_OFF":
            addr = to_int32(self.R[op.reg] + op.imm)
        else:
            raise VMError("E_DECODE", pc, "operand is not memory")
        if not (0 <= addr < self.MEM_SIZE):
            raise VMError("E_MEM_OOB", pc, f"address {addr}")
        return addr

    def _get(self, op: Operand, pc: int) -> int:
        if op.kind == "REG":
            return self.R[op.reg]
        if op.kind == "IMM":
            return op.imm
        if op.kind.startswith("MEM_"):
            return self.MEM[self._addr(op, pc)]
        raise VMError("E_DECODE", pc, f"bad operand kind {op.kind}")

    def _set_reg(self, r: int, v: int):
        self.R[r] = to_int32(v)
        self.Z = 1 if self.R[r] == 0 else 0
        self.N = 1 if self.R[r] < 0 else 0

    def step(self) -> bool:
        pc = self.PC
        if not (0 <= pc < len(self.prog)):
            raise VMError("E_BAD_JUMP", pc, "PC out of program")
        ins = self.prog[pc]
        op = ins.op

        if self.debug:
            print(f"PC={pc:04d}  {ins.raw:<25}  R={self.R}  Z={self.Z} N={self.N}")

        def require_reg(o: Operand, name: str):
            if o.kind != "REG":
                raise VMError("E_DECODE", pc, f"{name} must be REG")

        if op == "HALT":
            return False

        if op == "NOP":
            self.PC += 1
            return True

        if op in ("JMP", "JZ", "JNZ", "JN", "JNN"):
            target = ins.a.imm
            if op == "JMP":
                self.PC = target
            elif op == "JZ":
                self.PC = target if self.Z == 1 else pc + 1
            elif op == "JNZ":
                self.PC = target if self.Z == 0 else pc + 1
            elif op == "JN":
                self.PC = target if self.N == 1 else pc + 1
            else:  # JNN
                self.PC = target if self.N == 0 else pc + 1
            return True

        a, b = ins.a, ins.b

        if op == "MOV":
            require_reg(a, "dst")
            if b.kind not in ("REG", "IMM"):
                raise VMError("E_DECODE", pc, "MOV src must be REG or IMM")
            self._set_reg(a.reg, self._get(b, pc))
            self.PC += 1
            return True

        if op == "LD":
            require_reg(a, "dst")
            if not b.kind.startswith("MEM_"):
                raise VMError("E_DECODE", pc, "LD src must be memory")
            self._set_reg(a.reg, self._get(b, pc))
            self.PC += 1
            return True

        if op == "ST":
            if not a.kind.startswith("MEM_"):
                raise VMError("E_DECODE", pc, "ST dst must be memory")
            require_reg(b, "src")
            self.MEM[self._addr(a, pc)] = self.R[b.reg]
            self.PC += 1
            return True

        if op in ("ADD", "SUB", "MUL", "DIV"):
            require_reg(a, "dst")
            lhs = self.R[a.reg]
            rhs = self._get(b, pc)
            if op == "ADD":
                res = lhs + rhs
            elif op == "SUB":
                res = lhs - rhs
            elif op == "MUL":
                res = lhs * rhs
            else:
                try:
                    res = idiv_trunc0(lhs, rhs)
                except ZeroDivisionError:
                    raise VMError("E_DIV0", pc, "division by zero")
            self._set_reg(a.reg, res)
            self.PC += 1
            return True

        if op == "CMP":
            va = self._get(a, pc)
            vb = self._get(b, pc)
            self.Z = 1 if va == vb else 0
            self.N = 1 if va < vb else 0
            self.PC += 1
            return True

        raise VMError("E_DECODE", pc, f"unknown opcode {op}")

    def run(self, max_steps: int = 1_000_000):
        steps = 0
        while steps < max_steps:
            steps += 1
            if not self.step():
                return {"status": "OK", "steps": steps, "pc": self.PC}
        raise VMError("E_TIMEOUT", self.PC, f"max_steps={max_steps} exceeded")


if __name__ == "__main__":
    # Демонстрационная задача: сумма элементов массива
    # MEM[0]   = N        — длина массива
    # MEM[1..N] = A[0..N-1] — элементы массива
    # MEM[N+1] = sum      — результат (после HALT)
    prog = """
            LD   R0, [0]        ; R0 = N (счётчик)
            MOV  R7, R0         ; R7 = N (сохранить для адреса результата)
            MOV  R1, #1         ; R1 = ptr (индекс текущего элемента)
            MOV  R2, #0         ; R2 = sum = 0
    loop:
            CMP  R0, #0
            JZ   end
            LD   R3, [R1]       ; R3 = A[i]
            ADD  R2, R3         ; sum += A[i]
            ADD  R1, #1         ; ptr++
            SUB  R0, #1         ; счётчик--
            JMP  loop
    end:
            MOV  R4, R7
            ADD  R4, #1         ; R4 = N+1
            ST   [R4], R2       ; MEM[N+1] = sum
            HALT
    """

    vm = VM(debug=True)
    vm.load_program(prog)

    N = 4
    arr = [10, -3, 7, 100]
    vm.MEM[0] = to_int32(N)
    for i, v in enumerate(arr):
        vm.MEM[1 + i] = to_int32(v)

    result = vm.run()
    print(result)
    print("sum =", vm.MEM[N + 1])  # ожидается 114
