from tools import loader, parsers


class FileSystem:
    def __init__(self, data: list[str]) -> None:
        self.files: dict[str:[list[dict]]] = {}  # { path : [{ path : size }] }
        self.folders: dict[str, int] = {}  # { path : size }

        path = ''
        for line in data:
            _line = line.split()
            name = f'{path}/{_line[1]}'

            match _line:
                case ['$', 'ls']:
                    continue
                case ['$', 'cd', '..']:
                    path = path.rsplit('/', 1)[0]
                case ['$', 'cd', folder]:
                    path += f'/{folder}'
                    self.files[path] = []
                    self.folders[path] = 0
                case ['dir', _]:
                    self.files[path].append({name: 0})
                case _:
                    self.files[path].append({name: int(_line[0])})

        while True:
            for folder in self.folders:
                size = self.get_size(folder)
                self.folders.update({folder: size})
            if self.folders['//'] != 0:
                break

    def get_size(self, folder: str) -> int:
        size = 0
        for i in self.files[folder]:
            name, file_size = next(iter(i.items()))
            if file_size != 0:
                size += file_size
            elif self.folders[name] != 0:
                size += self.folders[name]
            else:
                return 0
        return size

    def part_1(self) -> int:
        """test part 1:
        >>> print(FileSystem(parsers.lines('test07.txt')).part_1())
        95437"""

        return sum(
            [self.folders[folder] for folder in self.folders if self.folders[folder] <= 100000]
        )

    def part_2(self) -> int:
        """test part 2:
        >>> print(FileSystem(parsers.lines('test07.txt')).part_2())
        24933642"""

        total = 70000000
        needed = 30000000
        free = total - self.folders['//']
        target = needed - free

        return min(
            [self.folders[folder] for folder in self.folders if self.folders[folder] >= target]
        )


print(FileSystem(parsers.lines(loader.get())).part_1())  # 1490523
print(FileSystem(parsers.lines(loader.get())).part_2())  # 12390492
