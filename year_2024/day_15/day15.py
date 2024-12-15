from collections import deque

import numpy as np
from numpy.typing import NDArray

from tools import loader, parsers
from tools.common import Point

MOVES = {
    '<': Point(col=-1, row=0), '>': Point(col=1, row=0),
    '^': Point(col=0, row=-1), 'v': Point(col=0, row=1),
}


def get_double_boxes(arr: NDArray, boxes: list[Point], direction: Point) -> list[Point]:
    added = set(boxes)
    queue = deque(boxes)
    while queue:
        current = queue.popleft()
        behind = current + direction
        char = arr[behind]
        if char == '[':
            box_behind = (behind, behind + MOVES['>'])
            if box_behind not in added:
                added.add(box_behind)
                boxes.extend(box_behind)
                queue.extend(box_behind)
        elif char == ']':
            box_behind = (behind + MOVES['<'], behind)
            if box_behind not in added:
                added.add(box_behind)
                boxes.extend(box_behind)
                queue.extend(box_behind)
        elif char == '#':
            raise ValueError
    return boxes


def get_stack_of_boxes(arr: NDArray, box: Point, direction: Point) -> list[Point]:
    boxes = [box]
    if direction in {MOVES['^'], MOVES['v']}:
        if arr[box] == ']':
            boxes.append(box + MOVES['<'])
        elif arr[box] == '[':
            boxes.append(box + MOVES['>'])
        if len(boxes) == 2:
            return get_double_boxes(arr, boxes, direction)

    while True:
        tile_behind_box = box + direction
        if arr[tile_behind_box] == '#':
            raise ValueError
        if arr[tile_behind_box] in 'O[]':
            boxes.append(tile_behind_box)
            box = tile_behind_box
        else:
            return boxes


def warehouse(data: list[list[str]], part2: bool) -> int:
    """
    >>> print(warehouse(parsers.blocks('test.txt'), part2=False))
    10092
    >>> print(warehouse(parsers.blocks('test.txt'), part2=True))
    9021"""
    map_, moves_ = data
    moves = iter(''.join(moves_))
    arr = np.array([list(i) for i in map_], dtype=str)
    if part2:
        arr = np.repeat(arr, 2, axis=1)
        for row, line in enumerate(arr.copy()):
            counter = 0
            for col, char in enumerate(line):
                if char == 'O':
                    counter += 1
                    arr[row, col] = ']' if counter % 2 == 0 else '['

    robot = Point(*np.argwhere(arr == '@')[0])
    arr[arr == '@'] = '.'

    for move in moves:
        next_move = MOVES[move]
        next_tile = robot + next_move
        if arr[next_tile] == '#':
            continue
        if arr[next_tile] in 'O[]':
            try:
                box_stack = get_stack_of_boxes(arr, next_tile, next_move)
            except ValueError:
                continue
            for box in reversed(box_stack):
                char = arr[box]
                arr[box] = '.'
                arr[box + next_move] = char
        robot = next_tile

    char = 'O' if not part2 else '['
    boxes = np.where(arr == char)
    return sum(boxes[0]) * 100 + sum(boxes[1])


print(warehouse(parsers.blocks(loader.get()), part2=False))  # 1371036
print(warehouse(parsers.blocks(loader.get()), part2=True))  # 1392847
