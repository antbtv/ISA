"""
Модель кэш-памяти для вычислительной системы (лабораторная работа №3).

Поддерживает:
- Произвольный размер кэша, строки и ассоциативность
- Политики замещения: LRU, FIFO, RANDOM
- Политики записи: write-back (по умолчанию), write-through
- Параметр штрафа за промах (miss_penalty)
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CacheLine:
    valid: bool = False
    dirty: bool = False
    tag: int = 0
    data: List[int] = field(default_factory=list)
    # Для политик замещения
    lru_timestamp: int = 0   # LRU: время последнего доступа
    fifo_order: int = 0      # FIFO: порядок загрузки в набор


class Cache:
    """
    Множественно-ассоциативная кэш-память.

    Параметры
    ---------
    cache_size   : общий объём кэша в словах (int32)
    line_size    : размер строки в словах
    associativity: ассоциативность (число строк в наборе)
    replacement  : политика замещения — 'LRU', 'FIFO', 'RANDOM'
    write_policy : политика записи — 'write-back', 'write-through'
    miss_penalty : штраф за промах в тактах конвейера
    """

    VALID_REPLACEMENTS = ('LRU', 'FIFO', 'RANDOM')
    VALID_WRITE_POLICIES = ('write-back', 'write-through')

    def __init__(
        self,
        cache_size: int = 256,
        line_size: int = 4,
        associativity: int = 2,
        replacement: str = 'LRU',
        write_policy: str = 'write-back',
        miss_penalty: int = 10,
    ):
        if replacement not in self.VALID_REPLACEMENTS:
            raise ValueError(f"replacement must be one of {self.VALID_REPLACEMENTS}")
        if write_policy not in self.VALID_WRITE_POLICIES:
            raise ValueError(f"write_policy must be one of {self.VALID_WRITE_POLICIES}")
        if line_size < 1 or (line_size & (line_size - 1)) != 0:
            raise ValueError("line_size must be a power of 2 >= 1")

        self.cache_size = cache_size
        self.line_size = line_size
        self.associativity = associativity
        self.replacement = replacement
        self.write_policy = write_policy
        self.miss_penalty = miss_penalty

        # Параметры адресации
        total_lines = cache_size // line_size
        self.num_sets = max(1, total_lines // associativity)
        self.offset_bits = int(math.log2(line_size))
        self.index_bits = int(math.log2(self.num_sets)) if self.num_sets > 1 else 0
        self.tag_bits = 32 - self.offset_bits - self.index_bits

        # Структура кэша: num_sets наборов по associativity строк
        self.sets: List[List[CacheLine]] = [
            [CacheLine(data=[0] * line_size) for _ in range(associativity)]
            for _ in range(self.num_sets)
        ]

        # Счётчик для LRU и FIFO
        self._clock: int = 0
        # Счётчик загрузок в каждый набор (для FIFO)
        self._fifo_counter: List[int] = [0] * self.num_sets

        # Статистика
        self.total_accesses: int = 0
        self.reads: int = 0
        self.writes: int = 0
        self.hits: int = 0
        self.misses: int = 0
        self.writebacks: int = 0

    # ------------------------------------------------------------------
    # Разложение адреса
    # ------------------------------------------------------------------

    def _decompose(self, addr: int):
        """Возвращает (tag, set_index, offset) для заданного адреса."""
        offset = addr & ((1 << self.offset_bits) - 1)
        if self.index_bits > 0:
            index = (addr >> self.offset_bits) & ((1 << self.index_bits) - 1)
        else:
            index = 0
        tag = addr >> (self.offset_bits + self.index_bits)
        return tag, index, offset

    # ------------------------------------------------------------------
    # Поиск строки в наборе
    # ------------------------------------------------------------------

    def _find_way(self, set_lines: List[CacheLine], tag: int) -> int:
        """Возвращает индекс способа (way) с совпадающим тегом, или -1."""
        for i, line in enumerate(set_lines):
            if line.valid and line.tag == tag:
                return i
        return -1

    # ------------------------------------------------------------------
    # Выбор жертвы для замещения
    # ------------------------------------------------------------------

    def _select_victim(self, set_idx: int) -> int:
        """Выбирает способ (way) для вытеснения."""
        set_lines = self.sets[set_idx]

        # Приоритет 1: невалидная строка (холодный старт)
        for i, line in enumerate(set_lines):
            if not line.valid:
                return i

        # Приоритет 2: политика замещения
        if self.replacement == 'RANDOM':
            return random.randrange(self.associativity)

        if self.replacement == 'LRU':
            return min(range(self.associativity),
                       key=lambda i: set_lines[i].lru_timestamp)

        if self.replacement == 'FIFO':
            return min(range(self.associativity),
                       key=lambda i: set_lines[i].fifo_order)

        return 0  # fallback

    # ------------------------------------------------------------------
    # Сброс грязной строки в основную память
    # ------------------------------------------------------------------

    def _writeback(self, line: CacheLine, set_idx: int, mem: list):
        """Записывает содержимое грязной строки обратно в основную память."""
        base_addr = (line.tag << (self.offset_bits + self.index_bits)) | (set_idx << self.offset_bits)
        for i, val in enumerate(line.data):
            a = base_addr + i
            if 0 <= a < len(mem):
                mem[a] = val
        self.writebacks += 1

    # ------------------------------------------------------------------
    # Загрузка блока из основной памяти в строку кэша
    # ------------------------------------------------------------------

    def _load_block(self, line: CacheLine, set_idx: int, tag: int,
                    addr: int, mem: list):
        """Загружает блок из основной памяти в строку кэша."""
        base_addr = addr - (addr & ((1 << self.offset_bits) - 1))  # выравнивание
        line.data = []
        for i in range(self.line_size):
            a = base_addr + i
            line.data.append(mem[a] if 0 <= a < len(mem) else 0)
        line.valid = True
        line.dirty = False
        line.tag = tag
        # Обновить FIFO order
        line.fifo_order = self._fifo_counter[set_idx]
        self._fifo_counter[set_idx] += 1

    # ------------------------------------------------------------------
    # Обновление LRU-метки
    # ------------------------------------------------------------------

    def _touch(self, line: CacheLine):
        self._clock += 1
        line.lru_timestamp = self._clock

    # ------------------------------------------------------------------
    # Публичный интерфейс: чтение
    # ------------------------------------------------------------------

    def read(self, addr: int, mem: list):
        """
        Читает слово по адресу addr через кэш.

        Возвращает (value: int, stall_cycles: int).
        stall_cycles == 0 при попадании, miss_penalty при промахе.
        """
        self.total_accesses += 1
        self.reads += 1

        tag, set_idx, offset = self._decompose(addr)
        set_lines = self.sets[set_idx]
        way = self._find_way(set_lines, tag)

        if way >= 0:
            # Попадание (hit)
            self.hits += 1
            line = set_lines[way]
            if self.replacement == 'LRU':
                self._touch(line)
            return line.data[offset], 0

        # Промах (miss)
        self.misses += 1
        victim_way = self._select_victim(set_idx)
        victim = set_lines[victim_way]

        # Сброс грязной строки
        if victim.valid and victim.dirty:
            self._writeback(victim, set_idx, mem)

        # Загрузка нового блока
        self._load_block(victim, set_idx, tag, addr, mem)

        if self.replacement == 'LRU':
            self._touch(victim)

        return victim.data[offset], self.miss_penalty

    # ------------------------------------------------------------------
    # Публичный интерфейс: запись
    # ------------------------------------------------------------------

    def write(self, addr: int, value: int, mem: list) -> int:
        """
        Записывает value по адресу addr через кэш.

        Возвращает stall_cycles (0 при попадании, miss_penalty при промахе).
        """
        self.total_accesses += 1
        self.writes += 1

        tag, set_idx, offset = self._decompose(addr)
        set_lines = self.sets[set_idx]
        way = self._find_way(set_lines, tag)

        stall = 0

        if way >= 0:
            # Попадание
            self.hits += 1
            line = set_lines[way]
            line.data[offset] = value
            if self.write_policy == 'write-back':
                line.dirty = True
            else:
                mem[addr] = value  # write-through: сразу в память
            if self.replacement == 'LRU':
                self._touch(line)
        else:
            # Промах
            self.misses += 1
            stall = self.miss_penalty

            if self.write_policy == 'write-through':
                # Write-through + write-no-allocate: писать прямо в память, не загружать блок
                mem[addr] = value
            else:
                # Write-back + write-allocate: загрузить блок, затем обновить
                victim_way = self._select_victim(set_idx)
                victim = set_lines[victim_way]
                if victim.valid and victim.dirty:
                    self._writeback(victim, set_idx, mem)
                self._load_block(victim, set_idx, tag, addr, mem)
                victim.data[offset] = value
                victim.dirty = True
                if self.replacement == 'LRU':
                    self._touch(victim)

        return stall

    # ------------------------------------------------------------------
    # Статистика
    # ------------------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total_accesses if self.total_accesses > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        return self.misses / self.total_accesses if self.total_accesses > 0 else 0.0

    def amat(self, hit_time: int = 1) -> float:
        """Average Memory Access Time = HitTime + MissRate * MissPenalty."""
        return hit_time + self.miss_rate * self.miss_penalty

    def get_stats(self) -> dict:
        return {
            'total_accesses': self.total_accesses,
            'reads': self.reads,
            'writes': self.writes,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hit_rate,
            'miss_rate': self.miss_rate,
            'writebacks': self.writebacks,
            'amat': self.amat(),
        }

    def describe(self) -> str:
        """Возвращает строку с конфигурацией кэша."""
        return (
            f"Cache(size={self.cache_size}w, line={self.line_size}w, "
            f"assoc={self.associativity}, sets={self.num_sets}, "
            f"repl={self.replacement}, policy={self.write_policy}, "
            f"penalty={self.miss_penalty})"
        )
