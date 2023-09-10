from tools import parsers, loader


class Firewall:
    def __init__(self, data: list[str]) -> None:
        self.layers = {int(k): int(v) for k, v in (line.split(': ') for line in data)}

    def traverse(self, step: int = 0) -> list[int]:
        hits = []
        for layer in range(max(self.layers.keys()) + 1):
            value = self.layers.get(layer, 0) - 1
            if value >= 0:
                offset = step % (value * 2)
                value = value * 2 - offset if offset > value else offset
                if value == 0:
                    hits.append(layer)
            step += 1
        return hits

    def part_1(self) -> int:
        """
        >>> print(Firewall(parsers.lines('test.txt')).part_1())
        24"""
        return sum(i * self.layers[i] for i in self.traverse())

    def part_2(self) -> int:
        """
        >>> print(Firewall(parsers.lines('test.txt')).part_2())
        10"""
        step_offset = 0
        while True:
            if not self.traverse(step_offset):
                return step_offset
            step_offset += 1


print(Firewall(parsers.lines(loader.get())).part_1())  # 1316
print(Firewall(parsers.lines(loader.get())).part_2())  # 3840052
