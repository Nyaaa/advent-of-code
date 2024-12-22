from collections import defaultdict, deque
from collections.abc import Generator
from itertools import islice
from math import floor

from tools import loader, parsers


def mix_and_prune(secret: int, number: int) -> int:
    return (secret ^ number) % 16777216


def pseudorandom_generator(secret: int) -> Generator[int]:
    while True:
        secret = mix_and_prune(secret, secret * 64)
        secret = mix_and_prune(secret, floor(secret / 32))
        secret = mix_and_prune(secret, secret * 2048)
        yield secret


def part_1(data: list[str]) -> int:
    """
    >>> print(part_1(parsers.lines('test.txt')))
    37327623
    """
    return sum(next(islice(pseudorandom_generator(int(number)), 1999, None)) for number in data)


def part_2(data: list[str]) -> int:
    """
    >>> print(part_2(parsers.lines('test2.txt')))
    23"""
    prices = defaultdict(int)
    for number in data:
        seen = set()
        sequence = deque([], maxlen=4)
        buyer = pseudorandom_generator(int(number))
        price = int(number[-1])
        for _ in range(2000):
            num = next(buyer)
            new_price = num % 10
            sequence.append(new_price - price)
            seq = tuple(sequence)
            if seq not in seen:
                seen.add(seq)
                prices[seq] += new_price
            price = new_price
    return max(prices.values())


print(part_1(parsers.lines(loader.get())))  # 15335183969
print(part_2(parsers.lines(loader.get())))  # 1696
