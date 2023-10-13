from tools import loader, parsers


class Droplet:
    def __init__(self, data: list[str]) -> None:
        self.cubes = [tuple(map(int, cube.split(','))) for cube in data]

    def part_1(self) -> int:
        """test part 1:
        >>> print(Droplet(parsers.lines('test.txt')).part_1())
        64"""
        exposed = 0
        for cube in self.cubes:
            adjacent = self.adjacent(cube)
            covered = adjacent.intersection(self.cubes)
            exposed += 6 - len(covered)
        return exposed

    def part_2(self) -> int:
        """test part 2:
        >>> print(Droplet(parsers.lines('test.txt')).part_2())
        58"""
        min_x, min_y, min_z = (min(cube[i] - 1 for cube in self.cubes) for i in range(3))
        max_x, max_y, max_z = (max(cube[i] + 1 for cube in self.cubes) for i in range(3))
        check = [(min_x, min_y, min_z)]
        checked = []
        exposed = 0

        while check:
            cube = check.pop(0)
            if cube in self.cubes:
                exposed += 1
                continue
            if cube not in checked:
                checked.append(cube)
                check += [(x, y, z) for x, y, z in self.adjacent(cube) if
                          min_x <= x <= max_x and
                          min_y <= y <= max_y and
                          min_z <= z <= max_z]
        return exposed

    @staticmethod
    def adjacent(cube: tuple[int, int, int]) -> set[tuple[int, int, int]]:
        x, y, z = cube
        return {(x + 1, y, z), (x - 1, y, z),
                (x, y + 1, z), (x, y - 1, z),
                (x, y, z + 1), (x, y, z - 1)}


print(Droplet(parsers.lines(loader.get())).part_1())  # 3498
print(Droplet(parsers.lines(loader.get())).part_2())  # 2008
