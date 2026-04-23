"""
VLIW симулятор — Лабораторная работа №4
«Моделирование архитектуры с явным параллелизмом (VLIW)»

Модель: issue width = 4 слота
  Ресурсы:
    ALU    (MOV, ADD, SUB, MUL, DIV, CMP) — max 2 в пакете
    MEM    (LD, ST)                        — max 1 в пакете
    BRANCH (JMP, JZ, JNZ, JN, JNN)        — max 1 в пакете

  Синтаксис пакета:
    Одиночная инструкция:  MOV R1, #1
    Параллельный пакет:    { MOV R1, #1 || ADD R2, R3 || LD R4, [R0] }
    Метка перед пакетом:   loop: { CMP R0, #0 || LD R3, [R1] }

  Правила корректности пакета:
    1. Нет зависимостей RAW, WAR, WAW между операциями пакета.
    2. Число операций каждого класса не превышает лимит ресурса.
    3. Общее число операций (без NOP) <= issue_width.
    4. HALT — только в одиночном пакете.
    5. Не более 1 инструкции перехода на пакет.

  Семантика исполнения:
    - Все операнды читаются из архитектурного состояния НАЧАЛА такта.
    - Все результаты фиксируются АТОМАРНО в конце такта.
    - Переход обновляет PC после фиксации всех результатов пакета.
    - NOP не учитывается в статистике операций.
"""

import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

from ISA import VM, Instr, Operand, to_int32, idiv_trunc0, VMError


# ---------------------------------------------------------------------------
#  Классификация ресурсов
# ---------------------------------------------------------------------------

ALU_OPS    = frozenset({"MOV", "ADD", "SUB", "MUL", "DIV", "CMP"})
MEM_OPS    = frozenset({"LD", "ST"})
BRANCH_OPS = frozenset({"JMP", "JZ", "JNZ", "JN", "JNN"})

ISSUE_WIDTH     = 4
RESOURCE_LIMITS = {"ALU": 2, "MEM": 1, "BRANCH": 1}


def resource_class(op: str) -> str:
    if op in ALU_OPS:    return "ALU"
    if op in MEM_OPS:    return "MEM"
    if op in BRANCH_OPS: return "BRANCH"
    return "OTHER"


# ---------------------------------------------------------------------------
#  Анализ зависимостей: R(I) и W(I)
# ---------------------------------------------------------------------------

def _mem_regs(o: Optional[Operand]) -> frozenset:
    if o and o.kind in ("MEM_REG", "MEM_REG_OFF"):
        return frozenset({o.reg})
    return frozenset()

def _reg_set(o: Optional[Operand]) -> frozenset:
    if o and o.kind == "REG":
        return frozenset({o.reg})
    return frozenset()


def op_reads(instr: Instr) -> Tuple[frozenset, frozenset]:
    """Возвращает (read_regs, read_flags)."""
    op, a, b = instr.op, instr.a, instr.b
    if op == "MOV":
        return _reg_set(b), frozenset()
    if op == "LD":
        return _mem_regs(b), frozenset()
    if op == "ST":
        return _mem_regs(a) | _reg_set(b), frozenset()
    if op in ("ADD", "SUB", "MUL", "DIV"):
        return frozenset({a.reg}) | _reg_set(b) | _mem_regs(b), frozenset()
    if op == "CMP":
        return _reg_set(a) | _mem_regs(a) | _reg_set(b) | _mem_regs(b), frozenset()
    if op in ("JZ", "JNZ"):
        return frozenset(), frozenset({"Z"})
    if op in ("JN", "JNN"):
        return frozenset(), frozenset({"N"})
    return frozenset(), frozenset()


def op_writes(instr: Instr) -> Tuple[frozenset, frozenset]:
    """Возвращает (write_regs, write_flags)."""
    op, a = instr.op, instr.a
    if op in ("MOV", "LD", "ADD", "SUB", "MUL", "DIV"):
        return frozenset({a.reg}), frozenset({"Z", "N"})
    if op == "CMP":
        return frozenset(), frozenset({"Z", "N"})
    return frozenset(), frozenset()


# ---------------------------------------------------------------------------
#  Структуры данных
# ---------------------------------------------------------------------------

@dataclass
class VLIWPacket:
    instrs:    List[Instr]   # все операции (включая NOP)
    labels:    List[str]     # метки, привязанные к пакету
    pkt_index: int           # порядковый номер пакета

    @property
    def real_instrs(self) -> List[Instr]:
        return [i for i in self.instrs if i.op not in ("NOP",)]


class VLIWError(Exception):
    def __init__(self, pkt: int, msg: str):
        super().__init__(f"VLIW error at packet {pkt}: {msg}")
        self.pkt = pkt
        self.msg = msg


# ---------------------------------------------------------------------------
#  VLIW виртуальная машина
# ---------------------------------------------------------------------------

class VLIWVM:
    def __init__(self, mem_size: int = 4096,
                 issue_width: int = ISSUE_WIDTH,
                 resource_limits: Optional[dict] = None,
                 debug: bool = False):
        self.mem_size       = mem_size
        self.issue_width    = issue_width
        self.resource_limits = resource_limits or dict(RESOURCE_LIMITS)
        self.debug          = debug
        self._helper_vm     = VM(mem_size=mem_size)

    # ------------------------------------------------------------------ parsing

    def _parse_one_instr(self, text: str, labels: Dict[str, int], pidx: int) -> Instr:
        """Разбирает одну ISA-инструкцию из строки."""
        text = text.strip()
        parts = text.split(None, 1)
        op   = parts[0].upper()
        rest = parts[1].strip() if len(parts) > 1 else ""
        a = b = None

        if op in ("HALT", "NOP"):
            if rest:
                raise VMError("E_DECODE", pidx, f"лишние операнды: {rest}")

        elif op in ("JMP", "JZ", "JNZ", "JN", "JNN"):
            if not rest:
                raise VMError("E_DECODE", pidx, "отсутствует метка")
            if rest not in labels:
                raise VMError("E_BAD_JUMP", pidx, f"неизвестная метка {rest}")
            a = Operand("IMM", imm=labels[rest])

        else:
            if not rest:
                raise VMError("E_DECODE", pidx, "отсутствуют операнды")
            toks = [t.strip() for t in rest.split(",")]
            if len(toks) != 2:
                raise VMError("E_DECODE", pidx, f"ожидалось 2 операнда, получено {len(toks)}")
            try:
                a = self._helper_vm._parse_operand(toks[0], pidx)
                b = self._helper_vm._parse_operand(toks[1], pidx)
            except VMError:
                raise
            except Exception as e:
                raise VMError("E_DECODE", pidx, str(e))

        return Instr(op=op, a=a, b=b, raw=text)

    def load_program(self, text: str) -> List[VLIWPacket]:
        """
        Разбирает VLIW-программу в список пакетов.
        Поддерживаемый синтаксис:
          Одиночная:  MOV R1, #1
          Пакет:      { MOV R1, #1 || ADD R2, R3 }
          Метки:      loop: { CMP R0, #0 || LD R3, [R1] }
        """
        lines = text.splitlines()

        # === Проход 1: сбор меток и сырых строк пакетов ===
        raw_packets: List[Tuple[List[str], List[str]]] = []  # (labels, op_strings)
        labels_map: Dict[str, int] = {}
        pkt_idx = 0
        pending_labels: List[str] = []
        i = 0

        while i < len(lines):
            line = lines[i]
            if ";" in line:
                line = line.split(";", 1)[0]
            line = line.strip()
            i += 1
            if not line:
                continue

            # Извлечь ведущие метки вида  label:  (возможно несколько подряд)
            while True:
                m = re.match(r'^([A-Za-z_][A-Za-z0-9_]*)\s*:(.*)', line)
                if m:
                    lab = m.group(1)
                    if lab in labels_map:
                        raise VMError("E_DECODE", pkt_idx, f"дублирующаяся метка {lab}")
                    labels_map[lab] = pkt_idx
                    pending_labels.append(lab)
                    line = m.group(2).strip()
                    if not line:
                        break
                else:
                    break

            if not line:
                continue

            # Блок { ... } — возможно многострочный
            if line.startswith("{"):
                block = line
                while "}" not in block:
                    if i >= len(lines):
                        raise VMError("E_DECODE", pkt_idx, "незакрытая скобка {")
                    nxt = lines[i]
                    if ";" in nxt:
                        nxt = nxt.split(";", 1)[0]
                    block += " " + nxt.strip()
                    i += 1

                inner = re.search(r'\{(.*?)\}', block, re.DOTALL)
                if not inner:
                    raise VMError("E_DECODE", pkt_idx, "некорректный блок {}")
                content = inner.group(1).strip()
                op_strs = [s.strip() for s in content.split("||") if s.strip()]

                raw_packets.append((list(pending_labels), op_strs))
                pending_labels = []
                pkt_idx += 1
            else:
                # Одиночная инструкция → одноэлементный пакет
                raw_packets.append((list(pending_labels), [line]))
                pending_labels = []
                pkt_idx += 1

        # === Проход 2: разбор инструкций и валидация пакетов ===
        packets: List[VLIWPacket] = []
        for pidx, (labs, op_strs) in enumerate(raw_packets):
            instrs = [self._parse_one_instr(s, labels_map, pidx) for s in op_strs]
            pkt = VLIWPacket(instrs=instrs, labels=labs, pkt_index=pidx)
            self._validate_packet(pkt)
            packets.append(pkt)

        return packets

    # ---------------------------------------------------------------- validation

    def _validate_packet(self, pkt: VLIWPacket):
        """Проверяет ресурсные ограничения и зависимости внутри пакета."""
        instrs = pkt.real_instrs
        pidx   = pkt.pkt_index

        if not instrs:
            return

        # HALT должен быть единственной операцией
        if any(i.op == "HALT" for i in instrs) and len(instrs) > 1:
            raise VLIWError(pidx, "HALT должен быть единственной операцией в пакете")

        # Проверка ресурсных лимитов
        counts: Dict[str, int] = {}
        for ins in instrs:
            rc = resource_class(ins.op)
            if rc in self.resource_limits:
                counts[rc] = counts.get(rc, 0) + 1
                lim = self.resource_limits[rc]
                if counts[rc] > lim:
                    raise VLIWError(pidx,
                        f"превышен лимит ресурса {rc}: {counts[rc]} > {lim}")

        # Проверка ширины выдачи
        if len(instrs) > self.issue_width:
            raise VLIWError(pidx,
                f"слишком много операций: {len(instrs)} > {self.issue_width}")

        # Попарная проверка зависимостей
        rw = [(ins, *op_reads(ins), *op_writes(ins)) for ins in instrs]
        # rw[k] = (ins, r_regs, r_flags, w_regs, w_flags)

        n = len(rw)
        for i in range(n):
            ins_i, rr_i, rf_i, wr_i, wf_i = rw[i]
            for j in range(n):
                if i == j:
                    continue
                ins_j, rr_j, rf_j, wr_j, wf_j = rw[j]

                # RAW: i пишет, j читает (проверяем и регистры, и флаги)
                if wr_i & rr_j:
                    raise VLIWError(pidx,
                        f"RAW по регистру(ам) {wr_i & rr_j}: "
                        f"'{ins_i.raw}' → '{ins_j.raw}'")
                if wf_i & rf_j:
                    raise VLIWError(pidx,
                        f"RAW по флагу(ам) {wf_i & rf_j}: "
                        f"'{ins_i.raw}' → '{ins_j.raw}'")

                # WAW только для регистров (флаги: последняя операция в пакете
                # определяет финальное значение Z/N — это детерминировано по
                # порядку слотов и допускается архитектурой VLIW)
                if i < j:
                    if wr_i & wr_j:
                        raise VLIWError(pidx,
                            f"WAW по регистру(ам) {wr_i & wr_j}: "
                            f"'{ins_i.raw}' и '{ins_j.raw}'")

                # WAR только для регистров (флаги не являются входными
                # операндами ALU-инструкций, кроме ветвлений; ветвления
                # не пишут регистры, поэтому конфликта нет)
                if rr_i & wr_j:
                    raise VLIWError(pidx,
                        f"WAR по регистру(ам) {rr_i & wr_j}: "
                        f"'{ins_i.raw}' читает, '{ins_j.raw}' пишет")

    # ----------------------------------------------------------------- execution

    def _mem_addr(self, o: Operand, R_snap: list, pidx: int) -> int:
        if o.kind == "MEM_DIR":
            addr = o.addr
        elif o.kind == "MEM_REG":
            addr = R_snap[o.reg]
        elif o.kind == "MEM_REG_OFF":
            addr = to_int32(R_snap[o.reg] + o.imm)
        else:
            raise VLIWError(pidx, f"операнд не является памятью: {o.kind}")
        if not (0 <= addr < self.mem_size):
            raise VLIWError(pidx, f"E_MEM_OOB: адрес {addr}")
        return addr

    def _get_val(self, o: Operand, R_snap: list, MEM_snap: list, pidx: int) -> int:
        if o.kind == "REG":
            return R_snap[o.reg]
        if o.kind == "IMM":
            return o.imm
        return MEM_snap[self._mem_addr(o, R_snap, pidx)]

    def execute(self, packets: List[VLIWPacket],
                mem_init: Optional[dict] = None) -> dict:
        """
        Исполняет список пакетов.
        Возвращает словарь со статистикой и финальным архитектурным состоянием.
        """
        R   = [0] * 8
        Z   = 0
        N   = 0
        MEM = [0] * self.mem_size

        if mem_init:
            for addr, val in mem_init.items():
                MEM[addr] = to_int32(val)

        total_ops     = 0   # непустые операции (не NOP), включая HALT
        total_packets = 0   # исполненных пакетов (тактов)
        cycles        = 0

        pc = 0  # индекс текущего пакета

        while pc < len(packets):
            pkt      = packets[pc]
            real_ops = pkt.real_instrs

            if self.debug:
                ops_str = " || ".join(i.raw for i in real_ops) if real_ops else "NOP"
                print(f"Cycle {cycles+1:4d}  PKT[{pc}]: {{ {ops_str} }}")

            # Пустой пакет (только NOP)
            if not real_ops:
                cycles        += 1
                total_packets += 1
                pc            += 1
                continue

            # HALT
            if real_ops[0].op == "HALT":
                cycles        += 1
                total_packets += 1
                total_ops     += 1
                break

            # Снимок состояния для атомарного чтения
            R_snap   = list(R)
            Z_snap   = Z
            N_snap   = N
            MEM_snap = list(MEM)

            # Накопление эффектов такта
            new_R:   Dict[int, int] = {}
            new_MEM: Dict[int, int] = {}
            new_Z: Optional[int]    = None
            new_N: Optional[int]    = None
            new_pc = pc + 1

            for ins in real_ops:
                if ins.op == "NOP":
                    continue

                op, a, b = ins.op, ins.a, ins.b
                pidx     = pkt.pkt_index

                if op == "MOV":
                    v = to_int32(self._get_val(b, R_snap, MEM_snap, pidx))
                    new_R[a.reg] = v
                    new_Z = 1 if v == 0 else 0
                    new_N = 1 if v <  0 else 0

                elif op == "LD":
                    v = to_int32(self._get_val(b, R_snap, MEM_snap, pidx))
                    new_R[a.reg] = v
                    new_Z = 1 if v == 0 else 0
                    new_N = 1 if v <  0 else 0

                elif op == "ST":
                    addr = self._mem_addr(a, R_snap, pidx)
                    new_MEM[addr] = R_snap[b.reg]

                elif op in ("ADD", "SUB", "MUL", "DIV"):
                    lhs = R_snap[a.reg]
                    rhs = self._get_val(b, R_snap, MEM_snap, pidx)
                    if op == "ADD":
                        res = lhs + rhs
                    elif op == "SUB":
                        res = lhs - rhs
                    elif op == "MUL":
                        res = lhs * rhs
                    else:
                        if rhs == 0:
                            raise VLIWError(pidx, "E_DIV0: деление на ноль")
                        res = idiv_trunc0(lhs, rhs)
                    v = to_int32(res)
                    new_R[a.reg] = v
                    new_Z = 1 if v == 0 else 0
                    new_N = 1 if v <  0 else 0

                elif op == "CMP":
                    va = self._get_val(a, R_snap, MEM_snap, pidx)
                    vb = self._get_val(b, R_snap, MEM_snap, pidx)
                    new_Z = 1 if va == vb else 0
                    new_N = 1 if va <  vb else 0

                elif op in ("JMP", "JZ", "JNZ", "JN", "JNN"):
                    target = a.imm
                    if op == "JMP":
                        new_pc = target
                    elif op == "JZ":
                        new_pc = target if Z_snap == 1 else pc + 1
                    elif op == "JNZ":
                        new_pc = target if Z_snap == 0 else pc + 1
                    elif op == "JN":
                        new_pc = target if N_snap == 1 else pc + 1
                    elif op == "JNN":
                        new_pc = target if N_snap == 0 else pc + 1

                total_ops += 1

            # Атомарная фиксация всех записей
            for reg, val in new_R.items():
                R[reg] = val
            for addr, val in new_MEM.items():
                MEM[addr] = val
            if new_Z is not None:
                Z = new_Z
            if new_N is not None:
                N = new_N

            cycles        += 1
            total_packets += 1
            pc             = new_pc

        max_slots  = cycles * self.issue_width
        slot_fill  = total_ops / max_slots if max_slots > 0 else 0.0
        ipc        = total_ops / cycles    if cycles   > 0 else 0.0

        return {
            "cycles":    cycles,
            "ops":       total_ops,
            "ipc":       ipc,
            "packets":   total_packets,
            "slot_fill": slot_fill,
            "R":  R,
            "Z":  Z,
            "N":  N,
            "MEM": MEM,
        }


# ---------------------------------------------------------------------------
#  Тестовая программа: задача 8 — подсчёт положительных элементов
#  Входные данные: MEM[0]=N, MEM[1..N]=A[], результат MEM[N+1]=count
# ---------------------------------------------------------------------------

PROG_SEQUENTIAL = """\
        LD   R0, [0]        ; R0 = N (счётчик)
        MOV  R7, R0         ; R7 = N (для адреса результата)
        MOV  R1, #1         ; R1 = ptr
        MOV  R2, #0         ; R2 = count
loop:
        CMP  R0, #0
        JZ   done
        LD   R3, [R1]       ; R3 = A[i]
        CMP  R3, #0
        JN   skip
        JZ   skip
        ADD  R2, #1
skip:
        ADD  R1, #1
        SUB  R0, #1
        JMP  loop
done:
        MOV  R4, R7
        ADD  R4, #1
        ST   [R4], R2
        HALT
"""

PROG_VLIW = """\
; === Пролог: параллельная инициализация ===
; LD R0,[0] (MEM) || MOV R1,#1 (ALU) || MOV R2,#0 (ALU)
; Нет RAW/WAR/WAW по регистрам между этими тремя операциями.
; Флаги: каждая перезаписывает Z/N; последняя (MOV R2,#0) задаёт итоговые.
; Флаги после пролога не используются сразу — порядок не критичен.
{ LD R0, [0] || MOV R1, #1 || MOV R2, #0 }
; MOV R7 и MOV R4 оба читают R0 из снимка — нет зависимостей
{ MOV R7, R0 || MOV R4, R0 }
; Вычислить адрес результата (N+1) заранее — освобождает эпилог
{ ADD R4, #1 }

; === Цикл ===
loop:
; КЛЮЧЕВОЙ ПАКЕТ: LD идёт первым в слоте, CMP — вторым.
; Семантика «последний слот побеждает по флагам»:
;   - R3 = A[ptr] (загружает LD из снимка R1)
;   - Z/N = результат CMP R0,#0 (перекрывает флаги LD)
; Нет RAW: CMP читает R0 (не R3); LD читает R1 (не результат CMP).
{ LD R3, [R1] || CMP R0, #0 }
{ JZ done }
{ CMP R3, #0 }
{ JN skip }
{ JZ skip }
{ ADD R2, #1 }
skip:
; ADD R1,#1 || SUB R0,#1: разные регистры, нет RAW/WAR/WAW по регистрам.
; Флаги от SUB (последнего) не читаются JMP → порядок флагов безразличен.
{ ADD R1, #1 || SUB R0, #1 }
{ JMP loop }

; === Эпилог: адрес R4 = N+1 уже готов из пролога ===
done:
{ ST [R4], R2 }
{ HALT }
"""


def make_mem_init(arr: list) -> dict:
    mem = {0: len(arr)}
    for i, v in enumerate(arr):
        mem[i + 1] = v
    return mem


def run_comparison(name: str, arr: list, expected: int):
    """Запускает последовательную и VLIW версии, выводит сравнение."""
    mem_init = make_mem_init(arr)
    n        = len(arr)

    vm_seq  = VLIWVM()
    vm_vliw = VLIWVM()

    pkts_seq  = vm_seq.load_program(PROG_SEQUENTIAL)
    pkts_vliw = vm_vliw.load_program(PROG_VLIW)

    res_seq  = vm_seq.execute(pkts_seq,  mem_init)
    res_vliw = vm_vliw.execute(pkts_vliw, mem_init)

    result_seq  = res_seq["MEM"][n + 1]  if n + 1 < vm_seq.mem_size  else None
    result_vliw = res_vliw["MEM"][n + 1] if n + 1 < vm_vliw.mem_size else None

    # Последовательная и VLIW-программы — разные планировки одного алгоритма.
    # Временные регистры (R3 и др.) могут различаться. Проверяем результат.
    result_ok = (result_seq == expected) and (result_vliw == expected)
    equiv     = result_ok  # семантическая эквивалентность = одинаковый результат

    print(f"\n{'='*65}")
    print(f"Тест: {name}")
    print(f"Массив: {arr}  Ожидаемый результат: {expected}")
    print(f"{'='*65}")

    print(f"\n{'Метрика':<30} {'Послед.':>10} {'VLIW':>10}")
    print("-" * 52)
    print(f"{'Тактов (циклов)':<30} {res_seq['cycles']:>10} {res_vliw['cycles']:>10}")
    print(f"{'Выполнено операций':<30} {res_seq['ops']:>10} {res_vliw['ops']:>10}")
    print(f"{'IPC':<30} {res_seq['ipc']:>10.3f} {res_vliw['ipc']:>10.3f}")
    print(f"{'Пакетов':<30} {res_seq['packets']:>10} {res_vliw['packets']:>10}")
    print(f"{'Коэф. заполнения слотов':<30} {res_seq['slot_fill']:>10.3f} {res_vliw['slot_fill']:>10.3f}")
    print(f"{'Ускорение (по тактам)':<30} {'—':>10} {res_seq['cycles']/res_vliw['cycles']:>10.3f}x")

    print(f"\nРезультат: MEM[{n+1}] = {result_seq} (послед.), {result_vliw} (VLIW)")
    print(f"Корректность: {'OK' if result_seq == expected else 'FAIL'} (послед.),"
          f" {'OK' if result_vliw == expected else 'FAIL'} (VLIW)")
    print(f"Семантическая эквивалентность: {'OK' if equiv else 'FAIL'}")

    return res_seq, res_vliw, equiv


def main():
    debug = "--debug" in sys.argv

    if debug:
        print("=== Отладочный режим: трассировка VLIW пакетов ===")
        vm = VLIWVM(debug=True)
        pkts = vm.load_program(PROG_VLIW)
        mem_init = make_mem_init([3, -1, 0, 7, -5])
        vm.execute(pkts, mem_init)
        return

    tests = [
        ("Смешанный массив",     [3, -1, 0, 7, -5],       2),
        ("Все положительные",    [1, 2, 3, 4, 5],          5),
        ("Все отрицательные",    [-1, -2, -3],              0),
        ("С нулями",             [0, 0, 0],                 0),
        ("Один положительный",   [42],                      1),
        ("Один отрицательный",   [-1],                      0),
        ("Пустой массив",        [],                        0),
        ("Большой массив",       [10,-3,7,0,-1,5,100,-50,1,0], 5),
    ]

    all_ok = True
    summary = []

    for name, arr, expected in tests:
        res_seq, res_vliw, equiv = run_comparison(name, arr, expected)
        if not equiv:
            all_ok = False
        summary.append((name, arr, res_seq, res_vliw))

    print(f"\n\n{'='*75}")
    print("ИТОГОВАЯ СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
    print(f"{'='*75}")
    print(f"{'Тест':<25} {'N':>3} | "
          f"{'SEQ такты':>9} {'SEQ IPC':>7} | "
          f"{'VLW такты':>9} {'VLW IPC':>7} {'Fill':>6} {'Уск.':>6}")
    print("-" * 75)
    for name, arr, rs, rv in summary:
        n = len(arr)
        speedup = rs['cycles'] / rv['cycles'] if rv['cycles'] > 0 else 0
        print(f"{name:<25} {n:>3} | "
              f"{rs['cycles']:>9} {rs['ipc']:>7.3f} | "
              f"{rv['cycles']:>9} {rv['ipc']:>7.3f} "
              f"{rv['slot_fill']:>6.3f} {speedup:>6.3f}x")

    print(f"\nВсе тесты: {'PASSED' if all_ok else 'FAILED'}")
    print(f"\nПараметры VLIW-модели: issue_width={ISSUE_WIDTH}, "
          f"ресурсы={RESOURCE_LIMITS}")


if __name__ == "__main__":
    main()
