"""
Конвейерный симулятор ISA (5 стадий: IF → ID → EX → MEM → WB)
Вариант B: с пересылкой результатов (forwarding)
Задача 8: подсчёт положительных элементов массива
"""

import sys
import copy
from dataclasses import dataclass, field
from typing import Optional

from ISA import VM, Instr, Operand, to_int32, idiv_trunc0, VMError


# ---------------------------------------------------------------------------
#  Метаданные инструкций для конвейера
# ---------------------------------------------------------------------------

@dataclass
class InstrMeta:
    reads_regs: set = field(default_factory=set)
    reads_flags: set = field(default_factory=set)
    writes_reg: Optional[int] = None
    writes_flags: set = field(default_factory=set)
    writes_mem: bool = False
    reads_mem: bool = False
    result_ready: str = "EX"        # "EX" или "MEM"
    is_branch: bool = False
    is_halt: bool = False
    is_nop: bool = False
    branch_target: Optional[int] = None


def _operand_regs(op: Optional[Operand]) -> set:
    if op is None:
        return set()
    if op.kind == "REG":
        return {op.reg}
    if op.kind in ("MEM_REG", "MEM_REG_OFF"):
        return {op.reg}
    return set()


def compute_meta(instr: Instr) -> InstrMeta:
    op = instr.op
    a, b = instr.a, instr.b

    if op == "HALT":
        return InstrMeta(is_halt=True)

    if op == "NOP":
        return InstrMeta(is_nop=True)

    if op in ("JMP", "JZ", "JNZ", "JN", "JNN"):
        rf = set()
        if op in ("JZ", "JNZ"):
            rf = {"Z"}
        elif op in ("JN", "JNN"):
            rf = {"N"}
        return InstrMeta(
            reads_flags=rf,
            is_branch=True,
            branch_target=a.imm,
        )

    if op == "MOV":
        return InstrMeta(
            reads_regs=_operand_regs(b),
            writes_reg=a.reg,
            writes_flags={"Z", "N"},
            result_ready="EX",
        )

    if op == "LD":
        return InstrMeta(
            reads_regs=_operand_regs(b),
            writes_reg=a.reg,
            writes_flags={"Z", "N"},
            reads_mem=True,
            result_ready="MEM",
        )

    if op == "ST":
        reads = _operand_regs(a) | _operand_regs(b)
        return InstrMeta(
            reads_regs=reads,
            writes_mem=True,
            result_ready="EX",
        )

    if op in ("ADD", "SUB", "MUL", "DIV"):
        reads = {a.reg} | _operand_regs(b)
        has_mem = b.kind.startswith("MEM_") if b else False
        return InstrMeta(
            reads_regs=reads,
            writes_reg=a.reg,
            writes_flags={"Z", "N"},
            reads_mem=has_mem,
            result_ready="MEM" if has_mem else "EX",
        )

    if op == "CMP":
        reads = _operand_regs(a) | _operand_regs(b)
        has_mem = False
        if a and a.kind.startswith("MEM_"):
            has_mem = True
        if b and b.kind.startswith("MEM_"):
            has_mem = True
        return InstrMeta(
            reads_regs=reads,
            writes_flags={"Z", "N"},
            reads_mem=has_mem,
            result_ready="MEM" if has_mem else "EX",
        )

    return InstrMeta()


# ---------------------------------------------------------------------------
#  Pipeline Register
# ---------------------------------------------------------------------------

@dataclass
class PipeReg:
    valid: bool = False
    pc: int = 0
    instr: Optional[Instr] = None
    meta: Optional[InstrMeta] = None

    # ID stage outputs
    val_a: int = 0
    val_b: int = 0
    store_val: int = 0
    addr_base: int = 0
    addr_offset: int = 0

    # EX stage outputs
    alu_result: int = 0
    mem_addr: int = 0
    branch_taken: bool = False
    branch_target: int = 0
    flags_z: int = 0
    flags_n: int = 0

    # MEM stage outputs
    mem_data: int = 0
    result: int = 0

    # tracking
    dst_reg: Optional[int] = None


def empty_reg() -> PipeReg:
    return PipeReg()


# ---------------------------------------------------------------------------
#  PipelinedVM
# ---------------------------------------------------------------------------

class PipelinedVM:
    def __init__(self, mem_size: int = 4096, debug: bool = False):
        self.mem_size = mem_size
        self.debug = debug

    def load_and_init(self, prog_text: str, mem_init: dict):
        self.prog_text = prog_text
        self.mem_init = mem_init

    def run_sequential(self) -> dict:
        vm = VM(mem_size=self.mem_size, debug=self.debug)
        vm.load_program(self.prog_text)
        for addr, val in self.mem_init.items():
            vm.MEM[addr] = to_int32(val)
        result = vm.run()
        return {
            "steps": result["steps"],
            "R": list(vm.R),
            "Z": vm.Z,
            "N": vm.N,
            "MEM": list(vm.MEM),
            "PC": vm.PC,
        }

    def run_pipelined(self, max_cycles: int = 100_000) -> dict:
        vm = VM(mem_size=self.mem_size, debug=False)
        vm.load_program(self.prog_text)
        for addr, val in self.mem_init.items():
            vm.MEM[addr] = to_int32(val)

        self.R = list(vm.R)
        self.Z = vm.Z
        self.N = vm.N
        self.MEM = list(vm.MEM)
        self.prog = vm.prog
        self.fetch_pc = 0
        self.halted = False
        self.finished = False

        self.if_id = empty_reg()
        self.id_ex = empty_reg()
        self.ex_mem = empty_reg()
        self.mem_wb = empty_reg()

        self.cycles = 0
        self.instructions_completed = 0
        self.data_stalls = 0
        self.structural_stalls = 0
        self.control_flushes = 0

        while self.cycles < max_cycles:
            self.cycles += 1

            stall = self._check_data_hazard()

            if self.debug:
                self._print_state(stall)

            # === WB: фиксация mem_wb в архитектурном состоянии ===
            self._do_wb()

            if self.finished:
                break

            # === MEM: обработка ex_mem → new_mem_wb ===
            new_mem_wb = self._do_mem()

            # === EX: обработка id_ex → new_ex_mem ===
            # Для переходов: флаги берутся с forwarding от new_mem_wb и arch state
            new_ex_mem = self._do_ex(new_mem_wb)

            flush = new_ex_mem.valid and new_ex_mem.branch_taken

            # === ID: обработка if_id → new_id_ex (с forwarding от new_ex_mem, new_mem_wb) ===
            if stall:
                new_id_ex = empty_reg()  # bubble
                self.data_stalls += 1
                new_if_id = self.if_id   # замораживаем
            else:
                new_id_ex = self._do_id(new_ex_mem, new_mem_wb)
                if flush:
                    new_if_id = empty_reg()
                else:
                    new_if_id = self._do_if()

            # === Flush при taken branch ===
            if flush:
                flushed_count = 0
                if self.if_id.valid:
                    flushed_count += 1
                if new_id_ex.valid:
                    flushed_count += 1
                self.control_flushes += flushed_count
                new_id_ex = empty_reg()
                new_if_id = empty_reg()
                self.fetch_pc = new_ex_mem.branch_target
                self.halted = False  # могли зафетчить HALT спекулятивно

            # === Advance pipeline ===
            self.mem_wb = new_mem_wb
            self.ex_mem = new_ex_mem
            self.id_ex = new_id_ex
            if not stall:
                self.if_id = new_if_id

        final_pc = self._find_halt_pc()
        return {
            "cycles": self.cycles,
            "instructions": self.instructions_completed,
            "cpi": self.cycles / max(self.instructions_completed, 1),
            "data_stalls": self.data_stalls,
            "structural_stalls": self.structural_stalls,
            "control_flushes": self.control_flushes,
            "R": list(self.R),
            "Z": self.Z,
            "N": self.N,
            "MEM": list(self.MEM),
            "PC": final_pc,
        }

    def _find_halt_pc(self) -> int:
        for i, ins in enumerate(self.prog):
            if ins.op == "HALT":
                return i
        return len(self.prog)

    # ---- Hazard detection ----

    def _check_data_hazard(self) -> bool:
        """Load-use hazard: инструкция в EX (id_ex) с result_ready="MEM"
        пишет регистр/флаги, которые читает инструкция в ID (if_id)."""
        if not self.id_ex.valid or not self.if_id.valid:
            return False
        ex_meta = self.id_ex.meta
        id_meta = self.if_id.meta
        if ex_meta is None or id_meta is None:
            return False
        if ex_meta.result_ready == "MEM" and self.id_ex.dst_reg is not None:
            if self.id_ex.dst_reg in id_meta.reads_regs:
                return True
        if ex_meta.result_ready == "MEM" and (ex_meta.writes_flags & id_meta.reads_flags):
            return True
        return False

    # ---- Forwarding ----

    def _forward_reg(self, reg: int, from_ex: PipeReg, from_mem: PipeReg) -> int:
        """Получить значение регистра с forwarding.
        from_ex = результат текущего цикла EX (new_ex_mem)
        from_mem = результат текущего цикла MEM (new_mem_wb)
        Регистровый файл уже обновлён WB в начале цикла.
        """
        # EX → ID: результат ALU только что вычислен в EX
        if from_ex.valid and from_ex.dst_reg == reg:
            if from_ex.meta and from_ex.meta.result_ready == "EX":
                return from_ex.alu_result
        # MEM → ID: результат только что получен в MEM
        if from_mem.valid and from_mem.dst_reg == reg:
            return from_mem.result
        # Из регистрового файла (обновлён WB)
        return self.R[reg]

    def _forward_flag(self, flag: str, from_ex: PipeReg, from_mem: PipeReg) -> int:
        """Получить значение флага с forwarding."""
        # EX → : результат EX стадии
        if from_ex.valid and from_ex.meta and flag in from_ex.meta.writes_flags:
            if from_ex.meta.result_ready == "EX":
                return from_ex.flags_z if flag == "Z" else from_ex.flags_n
        # MEM → : результат MEM стадии
        if from_mem.valid and from_mem.meta and flag in from_mem.meta.writes_flags:
            return from_mem.flags_z if flag == "Z" else from_mem.flags_n
        # Архитектурное состояние (обновлено WB)
        return self.Z if flag == "Z" else self.N

    # ---- Pipeline stages ----

    def _do_if(self) -> PipeReg:
        if self.halted or self.fetch_pc >= len(self.prog):
            return empty_reg()

        instr = self.prog[self.fetch_pc]
        meta = compute_meta(instr)
        reg = PipeReg(
            valid=True,
            pc=self.fetch_pc,
            instr=instr,
            meta=meta,
            dst_reg=meta.writes_reg,
        )
        if meta.is_halt:
            self.halted = True
        self.fetch_pc += 1
        return reg

    def _do_id(self, from_ex: PipeReg, from_mem: PipeReg) -> PipeReg:
        if not self.if_id.valid:
            return empty_reg()

        r = copy.copy(self.if_id)
        instr = r.instr
        meta = r.meta
        op = instr.op

        if meta.is_halt or meta.is_nop:
            return r

        if meta.is_branch:
            r.branch_target = meta.branch_target
            return r

        a, b = instr.a, instr.b

        def fwd(reg_idx):
            return self._forward_reg(reg_idx, from_ex, from_mem)

        def read_addr(mem_op):
            if mem_op.kind == "MEM_DIR":
                return 0, mem_op.addr
            elif mem_op.kind == "MEM_REG":
                return fwd(mem_op.reg), 0
            elif mem_op.kind == "MEM_REG_OFF":
                return fwd(mem_op.reg), mem_op.imm
            return 0, 0

        if op == "MOV":
            if b.kind == "REG":
                r.val_b = fwd(b.reg)
            elif b.kind == "IMM":
                r.val_b = b.imm

        elif op == "LD":
            r.addr_base, r.addr_offset = read_addr(b)

        elif op == "ST":
            r.store_val = fwd(b.reg)
            r.addr_base, r.addr_offset = read_addr(a)

        elif op in ("ADD", "SUB", "MUL", "DIV"):
            r.val_a = fwd(a.reg)
            if b.kind == "REG":
                r.val_b = fwd(b.reg)
            elif b.kind == "IMM":
                r.val_b = b.imm
            elif b.kind.startswith("MEM_"):
                r.addr_base, r.addr_offset = read_addr(b)

        elif op == "CMP":
            if a.kind == "REG":
                r.val_a = fwd(a.reg)
            elif a.kind == "IMM":
                r.val_a = a.imm
            elif a.kind.startswith("MEM_"):
                r.addr_base, r.addr_offset = read_addr(a)

            if b.kind == "REG":
                r.val_b = fwd(b.reg)
            elif b.kind == "IMM":
                r.val_b = b.imm

        return r

    def _do_ex(self, from_mem: PipeReg) -> PipeReg:
        """EX: ALU / адреса / разрешение переходов.
        from_mem = результат MEM стадии этого же цикла (для forwarding флагов)."""
        if not self.id_ex.valid:
            return empty_reg()

        r = copy.copy(self.id_ex)
        instr = r.instr
        meta = r.meta
        op = instr.op

        if meta.is_halt or meta.is_nop:
            return r

        if meta.is_branch:
            # Forwarding флагов: от инструкции в MEM (from_mem) и от arch state
            # from_mem = new_mem_wb, инструкция которая только что прошла MEM
            # Используем from_mem как "from_mem" и empty как "from_ex" (нет более ранней)
            z = self._forward_flag("Z", empty_reg(), from_mem)
            n = self._forward_flag("N", empty_reg(), from_mem)
            taken = False
            if op == "JMP":
                taken = True
            elif op == "JZ":
                taken = (z == 1)
            elif op == "JNZ":
                taken = (z == 0)
            elif op == "JN":
                taken = (n == 1)
            elif op == "JNN":
                taken = (n == 0)
            r.branch_taken = taken
            r.branch_target = meta.branch_target
            return r

        if op == "MOV":
            r.alu_result = to_int32(r.val_b)
            r.flags_z = 1 if r.alu_result == 0 else 0
            r.flags_n = 1 if r.alu_result < 0 else 0
            r.result = r.alu_result

        elif op == "LD":
            r.mem_addr = to_int32(r.addr_base + r.addr_offset)

        elif op == "ST":
            r.mem_addr = to_int32(r.addr_base + r.addr_offset)

        elif op in ("ADD", "SUB", "MUL", "DIV"):
            if meta.reads_mem:
                r.mem_addr = to_int32(r.addr_base + r.addr_offset)
            else:
                r.alu_result = self._alu(op, r.val_a, r.val_b)
                r.flags_z = 1 if r.alu_result == 0 else 0
                r.flags_n = 1 if r.alu_result < 0 else 0
                r.result = r.alu_result

        elif op == "CMP":
            if not meta.reads_mem:
                r.flags_z = 1 if r.val_a == r.val_b else 0
                r.flags_n = 1 if r.val_a < r.val_b else 0
            else:
                r.mem_addr = to_int32(r.addr_base + r.addr_offset)

        return r

    def _do_mem(self) -> PipeReg:
        if not self.ex_mem.valid:
            return empty_reg()

        r = copy.copy(self.ex_mem)
        instr = r.instr
        meta = r.meta
        op = instr.op

        if meta.is_halt or meta.is_nop or meta.is_branch:
            return r

        if op == "LD":
            addr = r.mem_addr
            if 0 <= addr < len(self.MEM):
                r.mem_data = self.MEM[addr]
            else:
                r.mem_data = 0
            r.result = to_int32(r.mem_data)
            r.flags_z = 1 if r.result == 0 else 0
            r.flags_n = 1 if r.result < 0 else 0

        elif op == "ST":
            addr = r.mem_addr
            if 0 <= addr < len(self.MEM):
                self.MEM[addr] = r.store_val

        elif op in ("ADD", "SUB", "MUL", "DIV") and meta.reads_mem:
            addr = r.mem_addr
            if 0 <= addr < len(self.MEM):
                r.mem_data = self.MEM[addr]
            else:
                r.mem_data = 0
            r.alu_result = self._alu(op, r.val_a, r.mem_data)
            r.flags_z = 1 if r.alu_result == 0 else 0
            r.flags_n = 1 if r.alu_result < 0 else 0
            r.result = r.alu_result

        elif op == "CMP" and meta.reads_mem:
            addr = r.mem_addr
            if 0 <= addr < len(self.MEM):
                r.mem_data = self.MEM[addr]
            else:
                r.mem_data = 0
            a_is_mem = instr.a and instr.a.kind.startswith("MEM_")
            if a_is_mem:
                va, vb = r.mem_data, r.val_b
            else:
                va, vb = r.val_a, r.mem_data
            r.flags_z = 1 if va == vb else 0
            r.flags_n = 1 if va < vb else 0

        else:
            r.result = r.alu_result

        return r

    def _do_wb(self):
        r = self.mem_wb
        if not r.valid:
            return

        meta = r.meta

        if meta.is_halt:
            self.finished = True
            self.instructions_completed += 1
            return

        if meta.is_nop or meta.is_branch:
            self.instructions_completed += 1
            return

        if r.dst_reg is not None:
            self.R[r.dst_reg] = r.result

        if meta.writes_flags:
            self.Z = r.flags_z
            self.N = r.flags_n

        self.instructions_completed += 1

    # ---- ALU ----

    def _alu(self, op: str, lhs: int, rhs: int) -> int:
        if op == "ADD":
            return to_int32(lhs + rhs)
        elif op == "SUB":
            return to_int32(lhs - rhs)
        elif op == "MUL":
            return to_int32(lhs * rhs)
        elif op == "DIV":
            if rhs == 0:
                return 0
            return to_int32(idiv_trunc0(lhs, rhs))
        return 0

    # ---- Debug ----

    def _stage_str(self, name: str, reg: PipeReg) -> str:
        if not reg.valid:
            return f"{name}: {'---':^20}"
        raw = reg.instr.raw if reg.instr else "?"
        return f"{name}: PC={reg.pc:<3} {raw:<15}"

    def _print_state(self, stall: bool):
        parts = [
            self._stage_str("WB ", self.mem_wb),
            self._stage_str("MEM", self.ex_mem),
            self._stage_str("EX ", self.id_ex),
            self._stage_str("ID ", self.if_id),
        ]
        line = f"Cycle {self.cycles:4d}: " + " | ".join(parts)
        if stall:
            line += "  [STALL]"
        print(line)


# ---------------------------------------------------------------------------
#  Тестовая программа: задача 8 — подсчёт положительных элементов массива
# ---------------------------------------------------------------------------

PROG_COUNT_POSITIVE = """\
        LD   R0, [0]        ; R0 = N (счётчик)
        MOV  R7, R0         ; R7 = N (сохранить для адреса результата)
        MOV  R1, #1         ; R1 = ptr (индекс текущего элемента)
        MOV  R2, #0         ; R2 = count = 0
loop:
        CMP  R0, #0         ; проверить счётчик
        JZ   done           ; если 0 — выход
        LD   R3, [R1]       ; R3 = A[i]
        CMP  R3, #0         ; сравнить элемент с 0
        JN   skip           ; R3 < 0 — пропустить
        JZ   skip           ; R3 == 0 — пропустить
        ADD  R2, #1         ; count++ (элемент положительный)
skip:
        ADD  R1, #1         ; ptr++
        SUB  R0, #1         ; counter--
        JMP  loop
done:
        MOV  R4, R7         ; R4 = N
        ADD  R4, #1         ; R4 = N+1
        ST   [R4], R2       ; MEM[N+1] = count
        HALT
"""


def make_mem_init(arr: list) -> dict:
    mem = {0: len(arr)}
    for i, v in enumerate(arr):
        mem[i + 1] = v
    return mem


def run_test(name: str, arr: list, expected: int, debug: bool = False):
    mem_init = make_mem_init(arr)
    n = len(arr)

    pvm = PipelinedVM(debug=debug)
    pvm.load_and_init(PROG_COUNT_POSITIVE, mem_init)

    seq = pvm.run_sequential()
    pipe = pvm.run_pipelined()

    seq_result = seq["MEM"][n + 1] if n + 1 < len(seq["MEM"]) else None
    pipe_result = pipe["MEM"][n + 1] if n + 1 < len(pipe["MEM"]) else None

    regs_ok = seq["R"] == pipe["R"]
    z_ok = seq["Z"] == pipe["Z"]
    n_ok = seq["N"] == pipe["N"]
    mem_ok = seq["MEM"][:n + 2] == pipe["MEM"][:n + 2]
    all_ok = regs_ok and z_ok and n_ok and mem_ok

    print(f"\n{'='*60}")
    print(f"Тест: {name}")
    print(f"Массив: {arr}")
    print(f"Ожидаемый результат: {expected}")
    print(f"{'='*60}")
    print(f"\n--- Последовательное выполнение ---")
    print(f"  Инструкций: {seq['steps']}")
    print(f"  Результат:  MEM[{n+1}] = {seq_result}")
    print(f"\n--- Конвейерное выполнение ---")
    print(f"  Тактов:               {pipe['cycles']}")
    print(f"  Инструкций:           {pipe['instructions']}")
    print(f"  CPI:                  {pipe['cpi']:.2f}")
    print(f"  Data stalls:          {pipe['data_stalls']}")
    print(f"  Structural stalls:    {pipe['structural_stalls']}")
    print(f"  Control flushes:      {pipe['control_flushes']}")
    print(f"  Результат:            MEM[{n+1}] = {pipe_result}")
    print(f"\n--- Верификация ---")
    print(f"  Регистры:  {'OK' if regs_ok else 'FAIL'}")
    print(f"  Флаг Z:    {'OK' if z_ok else 'FAIL'}")
    print(f"  Флаг N:    {'OK' if n_ok else 'FAIL'}")
    print(f"  Память:    {'OK' if mem_ok else 'FAIL'}")
    print(f"  Результат: {'OK' if seq_result == expected else 'FAIL'} (seq), {'OK' if pipe_result == expected else 'FAIL'} (pipe)")
    print(f"  Эквивалентность: {'OK' if all_ok else 'FAIL'}")

    if not regs_ok:
        print(f"  SEQ R:  {seq['R']}")
        print(f"  PIPE R: {pipe['R']}")
    if not mem_ok:
        for i in range(n + 2):
            if seq["MEM"][i] != pipe["MEM"][i]:
                print(f"  MEM[{i}]: seq={seq['MEM'][i]}, pipe={pipe['MEM'][i]}")

    return all_ok, pipe


def main():
    debug = "--debug" in sys.argv

    tests = [
        ("Смешанный массив",    [3, -1, 0, 7, -5],     2),
        ("Все положительные",   [1, 2, 3, 4, 5],       5),
        ("Все отрицательные",   [-1, -2, -3],           0),
        ("С нулями",            [0, 0, 0],              0),
        ("Один положительный",  [42],                   1),
        ("Один отрицательный",  [-1],                   0),
        ("Пустой массив",       [],                     0),
        ("Большой массив",      [10, -3, 7, 0, -1, 5, 100, -50, 1, 0], 5),
    ]

    all_passed = True
    stats_summary = []

    for name, arr, expected in tests:
        ok, pipe = run_test(name, arr, expected, debug=(debug and name == tests[0][0]))
        if not ok:
            all_passed = False
        stats_summary.append((name, pipe))

    print(f"\n{'='*60}")
    print("ИТОГОВАЯ ТАБЛИЦА")
    print(f"{'='*60}")
    print(f"{'Тест':<30} {'Такты':>6} {'Инстр':>6} {'CPI':>6} {'D-stl':>6} {'Flush':>6}")
    print("-" * 66)
    for name, pipe in stats_summary:
        print(f"{name:<30} {pipe['cycles']:>6} {pipe['instructions']:>6} {pipe['cpi']:>6.2f} {pipe['data_stalls']:>6} {pipe['control_flushes']:>6}")

    print(f"\nВсе тесты: {'PASSED' if all_passed else 'FAILED'}")


if __name__ == "__main__":
    main()
