from tools import parsers, loader


def part_2() -> int:
    seen = set()
    out = 0
    while True:
        for line in data:
            out += int(line)
            if out in seen:
                return out
            seen.add(out)


data = parsers.lines(loader.get())
print(f'Part 1: {sum(int(i) for i in data)}')  # 543
print(f'Part 2: {part_2()}')  # 621
