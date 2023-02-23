from tools import parsers, loader


class Droplet:
    def __init__(self, data):
        self.exposed = []
        self.cubes = [tuple(map(int, cube.split(','))) for cube in data]

    def part_1(self):
        """test part 1:
        >>> print(Droplet(parsers.lines('test.txt')).part_1())
        64"""
        exposed = 0
        for cube in self.cubes:
            x, y, z = cube
            adjacent = {(x + 1, y, z), (x - 1, y, z),
                        (x, y + 1, z), (x, y - 1, z),
                        (x, y, z + 1), (x, y, z - 1)}
            covered = adjacent.intersection(self.cubes)
            exposed += 6 - len(covered)
        return exposed


print(Droplet(parsers.lines(loader.get())).part_1())  # 3498
