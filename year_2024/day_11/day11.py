from collections import Counter

from tools import loader, parsers


def count_stones(data: str, limit: int) -> int:
    """
    >>> print(count_stones('125 17', 25))
    55312
    """
    stones = Counter(map(int, data.split()))
    for _ in range(limit):
        current_stones = Counter()
        for stone, amount in stones.items():
            string = str(stone)
            if stone == 0:
                current_stones[1] += amount
            elif len(string) % 2 == 0:
                mid = len(string) // 2
                left, right = string[:mid], string[mid:]
                current_stones[int(left)] += amount
                current_stones[int(right)] += amount
            else:
                current_stones[stone * 2024] += amount
        stones = current_stones
    return sum(stones.values())


print(count_stones(parsers.string(loader.get()), 25))  # 186203
print(count_stones(parsers.string(loader.get()), 75))  # 221291560078593
