"""Поэтапное решение прямоугольных пятнашек с направленными ценами ходов.

Дейкстра оптимизирует каждый локальный этап в абстракции выбранных плиток.
Глобальная оптимальность следует только для задачи, целиком решённой IDA*.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from pathlib import Path


MOVES = ((-1, 0, "U"), (1, 0, "D"), (0, -1, "L"), (0, 1, "R"))
MOVE_INDEX = {move[2]: index for index, move in enumerate(MOVES)}


class UnsolvableError(ValueError):
    """Конфигурация не может быть собрана законными ходами."""


@dataclass(frozen=True)
class Task:
    """Поле и цены перемещения плитки вверх, вниз, влево и вправо."""

    rows: int
    cols: int
    board: tuple[int, ...]
    costs: tuple[int, int, int, int]


@dataclass(frozen=True)
class Result:
    """Проверяемый итог с полной последовательностью ходов."""

    order: tuple[tuple[int, str], ...]
    cost: int
    final_board: tuple[int, ...]


def parse_task(text: str) -> Task:
    """Прочитать формат DATA/COST/ORDER; готовый ORDER здесь игнорируется."""

    lines = [line.split("--", 1)[0].strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    blocks: dict[str, list[str]] = {}
    index = 0

    while index < len(lines):
        name = lines[index]
        if name not in {"DATA", "COST", "ORDER"} or name in blocks:
            raise ValueError(f"Неизвестный или повторный блок: {name}")
        index += 1
        body: list[str] = []

        while index < len(lines):
            part, separator, tail = lines[index].partition("/")
            if part.strip():
                body.append(part.strip())
            index += 1
            if separator:
                if tail.strip():
                    raise ValueError("Данные после закрывающего /")
                break
        else:
            raise ValueError(f"Блок {name} не закрыт")

        blocks[name] = body

    if "DATA" not in blocks:
        raise ValueError("Отсутствует блок DATA")

    data = blocks["DATA"]
    header = data[0].split()
    if len(header) != 3 or header[0] != "SIZE":
        raise ValueError("Ожидалось SIZE N M")
    rows, cols = map(int, header[1:])
    if not (2 <= rows <= 100 and 2 <= cols <= 100):
        raise ValueError("Размеры должны лежать в диапазоне 2..100")
    if len(data) != rows + 1:
        raise ValueError("Неверное число строк поля")

    board: list[int] = []
    for row in data[1:]:
        tokens = row.split()
        if len(tokens) != cols:
            raise ValueError("Неверное число клеток в строке")
        board.extend(0 if token == "=" else int(token) for token in tokens)
    if set(board) != set(range(rows * cols)):
        raise ValueError("Фишки должны встречаться ровно один раз; пустота — знак =")

    costs = tuple(map(int, " ".join(blocks.get("COST", ["1 1 1 1"])).split()))
    if len(costs) != 4 or min(costs) < 0:
        raise ValueError("COST требует четыре неотрицательные цены U D L R")
    return Task(rows, cols, tuple(board), costs)


def load_task(path: str | Path) -> Task:
    """Прочитать задачу из текстового файла UTF-8."""

    return parse_task(Path(path).read_text(encoding="utf-8-sig"))


def is_solvable(task: Task) -> bool:
    """Проверить инвариант чётности относительно стандартной цели."""

    values = [tile for tile in task.board if tile]
    inversions = sum(a > b for index, a in enumerate(values) for b in values[index + 1 :])
    if task.cols % 2:
        return inversions % 2 == 0
    blank_from_bottom = task.rows - task.board.index(0) // task.cols
    return (inversions + blank_from_bottom) % 2 == 1


class Solver:
    """Сохраняет поле, фиксированные клетки и последовательность законных ходов."""

    def __init__(self, task: Task) -> None:
        """Подготовить изменяемое состояние и соседство клеток."""

        self.task = task
        self.board = list(task.board)
        self.blank = self.board.index(0)
        self.locked: set[int] = set()
        self.order: list[tuple[int, str]] = []
        self.cost = 0
        self.neighbors: list[list[tuple[int, str, int]]] = []

        for position in range(len(self.board)):
            row, col = divmod(position, task.cols)
            adjacent = []
            for dr, dc, direction in MOVES:
                nr, nc = row + dr, col + dc
                if 0 <= nr < task.rows and 0 <= nc < task.cols:
                    tile_position = nr * task.cols + nc
                    # Пустота идёт в (dr, dc), плитка — в обратном направлении.
                    tile_direction = MOVES[MOVE_INDEX[direction] ^ 1][2]
                    price = task.costs[MOVE_INDEX[tile_direction]]
                    adjacent.append((tile_position, tile_direction, price))
            self.neighbors.append(adjacent)

    def move(self, tile_position: int) -> None:
        """Выполнить один ход и записать его фактическую стоимость."""

        for neighbor, direction, price in self.neighbors[self.blank]:
            if neighbor == tile_position and neighbor not in self.locked:
                tile = self.board[neighbor]
                self.board[self.blank], self.board[neighbor] = tile, 0
                self.blank = neighbor
                self.order.append((tile, direction))
                self.cost += price
                return
        raise ValueError("Недопустимый ход или попытка сдвинуть зафиксированную плитку")

    def place(self, tiles: tuple[int, ...], targets: tuple[int, ...]) -> None:
        """Найти Дейкстрой минимальную цену этапа в абстракции выбранных плиток.

        Остальные незакреплённые плитки эквивалентны: их позиции не входят в ключ.
        Каждый найденный переход всё равно исполняется на настоящем поле.
        """

        positions = {tile: position for position, tile in enumerate(self.board)}
        start = tuple(positions[tile] for tile in tiles) + (self.blank,)
        if start[:-1] == targets:
            return

        best = {start: 0}
        parent: dict[tuple[int, ...], tuple[tuple[int, ...], int]] = {}
        queue = [(0, 0, start)]
        counter = 0

        while queue:
            distance, _, state = heapq.heappop(queue)
            if distance != best[state]:
                continue
            if state[:-1] == targets:
                path: list[int] = []
                while state != start:
                    state, tile_position = parent[state]
                    path.append(tile_position)
                for tile_position in reversed(path):
                    self.move(tile_position)
                return

            blank = state[-1]
            for neighbor, _, price in self.neighbors[blank]:
                if neighbor in self.locked:
                    continue
                tracked = list(state[:-1])
                if neighbor in tracked:
                    tracked[tracked.index(neighbor)] = blank
                next_state = tuple(tracked) + (neighbor,)
                candidate = distance + price
                if candidate < best.get(next_state, float("inf")):
                    best[next_state] = candidate
                    parent[next_state] = (state, neighbor)
                    counter += 1
                    heapq.heappush(queue, (candidate, counter, next_state))

        raise ValueError("Этап недостижим при текущих зафиксированных клетках")

    def remainder_task(self) -> Task:
        """Проверить границы остатка и перенумеровать плитки по локальной цели.

        Например, номера 25, 26, 27, 28 из первой строки углового блока
        становятся 1, 2, 3, 4. Критерий чётности применяется к этим номерам
        и ширине блока, а не к исходным номерам и ширине большого поля.
        """

        size = len(self.board)
        active = [position for position in range(size) if position not in self.locked]
        if not active or self.blank not in active or self.board[self.blank] != 0:
            raise RuntimeError("Пустая клетка должна находиться в рабочем остатке")

        top, left = divmod(active[0], self.task.cols)
        expected = [
            row * self.task.cols + col
            for row in range(top, self.task.rows)
            for col in range(left, self.task.cols)
        ]
        if active != expected:
            raise RuntimeError("Рабочий остаток должен быть прямоугольником в правом нижнем углу")

        for position in self.locked:
            target = position + 1 if position < size - 1 else 0
            if self.board[position] != target:
                raise RuntimeError("Зафиксированная плитка находится не на своём месте")

        goal = [position + 1 if position < size - 1 else 0 for position in active]
        values = [self.board[position] for position in active]
        if sorted(values) != sorted(goal):
            raise RuntimeError("Набор плиток остатка не соответствует его целевым клеткам")

        labels = {tile: index + 1 for index, tile in enumerate(goal) if tile}
        labels[0] = 0
        return Task(
            self.task.rows - top,
            self.task.cols - left,
            tuple(labels[tile] for tile in values),
            self.task.costs,
        )

    def solve_remainder(self) -> None:
        """Точно решить оставшуюся область методом IDA* с допустимой эвристикой."""

        active = tuple(position for position in range(len(self.board)) if position not in self.locked)
        if len(active) > 16:
            raise ValueError("IDA* рассчитан на остаток не более 4×4")

        remainder = self.remainder_task()
        if not is_solvable(remainder):
            raise UnsolvableError(
                "Остаток неразрешим по чётности. При правильно собранной внешней "
                "области всё поле также неразрешимо; временное перемещение "
                "собранных плиток не поможет."
            )

        goal = tuple(position + 1 if position < len(self.board) - 1 else 0 for position in active)
        local = {global_position: index for index, global_position in enumerate(active)}
        board = [self.board[position] for position in active]
        blank = local[self.blank]
        neighbors = [[(local[n], price) for n, _, price in self.neighbors[p] if n in local] for p in active]

        def heuristic() -> int:
            """Нижняя граница цены обязательных горизонтальных и вертикальных ходов."""

            total = 0
            for index, tile in enumerate(board):
                if not tile:
                    continue
                row, col = divmod(active[index], self.task.cols)
                target_row, target_col = divmod(tile - 1, self.task.cols)
                total += max(0, row - target_row) * self.task.costs[0]
                total += max(0, target_row - row) * self.task.costs[1]
                total += max(0, col - target_col) * self.task.costs[2]
                total += max(0, target_col - col) * self.task.costs[3]
            return total

        bound = heuristic()
        path: list[int] = []
        visited = {tuple(board)}

        def search(empty: int, depth_cost: int) -> int | bool:
            """Обойти очередной порог IDA*, избегая циклов текущего пути."""

            estimate = depth_cost + heuristic()
            if estimate > bound:
                return estimate
            if tuple(board) == goal:
                return True
            minimum = float("inf")
            for neighbor, price in neighbors[empty]:
                board[empty], board[neighbor] = board[neighbor], 0
                key = tuple(board)
                if key not in visited:
                    visited.add(key)
                    path.append(active[neighbor])
                    answer = search(neighbor, depth_cost + price)
                    if answer is True:
                        return True
                    minimum = min(minimum, answer)
                    path.pop()
                    visited.remove(key)
                board[neighbor], board[empty] = board[empty], board[neighbor]
            return minimum

        while True:
            answer = search(blank, 0)
            if answer is True:
                for tile_position in path:
                    self.move(tile_position)
                return
            if answer == float("inf"):
                raise ValueError("Оставшаяся область неразрешима")
            bound = answer

    def solve(self) -> Result:
        """Закрыть строки и столбцы парами на конце, затем решить остаток."""

        if not is_solvable(self.task):
            raise UnsolvableError(
                "Исходная конфигурация неразрешима по чётности: "
                "ни одна последовательность законных ходов не приведёт к цели."
            )

        top = left = 0
        target_side = 3 if min(self.task.rows, self.task.cols) == 3 else 4
        while self.task.rows - top > target_side or self.task.cols - left > target_side:
            height = self.task.rows - top
            width = self.task.cols - left

            if height > target_side and (width <= target_side or height >= width):
                cells = [top * self.task.cols + col for col in range(left, self.task.cols)]
                top += 1
            else:
                cells = [row * self.task.cols + left for row in range(top, self.task.rows)]
                left += 1

            for target in cells[:-2]:
                self.place((target + 1,), (target,))
                self.locked.add(target)
            pair = tuple(target + 1 for target in cells[-2:])
            self.place(pair, tuple(cells[-2:]))
            self.locked.update(cells[-2:])

        self.solve_remainder()
        goal = tuple(range(1, len(self.board))) + (0,)
        if tuple(self.board) != goal:
            raise AssertionError("Решатель не достиг целевой конфигурации")
        return Result(tuple(self.order), self.cost, tuple(self.board))


def solve(task: Task) -> Result:
    """Создать независимый решатель и вернуть проверенный итог."""

    return Solver(task).solve()


def format_board(board: tuple[int, ...], cols: int) -> str:
    """Представить поле строками с обозначением пустой клетки знаком =."""

    width = len(str(len(board) - 1))
    return "\n".join(" ".join(f"{tile or '=':>{width}}" for tile in board[i:i + cols])
                     for i in range(0, len(board), cols))


def verify(task: Task, result: Result) -> None:
    """Повторить ORDER и независимо проверить стоимость и конечное поле."""

    replay = Solver(task)
    for tile, direction in result.order:
        position = replay.board.index(tile)
        expected = next((d for n, d, _ in replay.neighbors[replay.blank] if n == position), None)
        if expected != direction:
            raise AssertionError("ORDER содержит недопустимый ход")
        replay.move(position)
    if replay.cost != result.cost or tuple(replay.board) != result.final_board:
        raise AssertionError("Стоимость или конечное поле не совпадают с ORDER")
