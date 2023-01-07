from tools import parsers


class FileSystem:
    def __init__(self, data):
        self.files: dict[str:[list[str, int]]] = {}  # { folder/file: [ path, size ] }
        self.folders: dict[str, int] = {}  # { folder : size }

        path = ''
        for line in data:
            line = line.split()
            name = f'{path}/{line[1]}'

            match line:
                case ['$', 'ls']:
                    continue
                case ['$', 'cd', '..']:
                    path = path.rsplit('/', 1)[0]
                case ['$', 'cd', folder]:
                    path += f'/{folder}'
                    self.files[path] = []
                    self.folders[path] = 0
                case ['dir', _]:
                    self.files[path].append({'dir': [name, 0]})
                case _:
                    self.files[path].append({'file': [name, int(line[0])]})

        while True:
            for folder in self.folders:
                size = self.get_size(folder)
                self.folders.update({folder: size})
            if self.folders['//'] != 0:
                break

    def get_size(self, folder) -> int:
        size = 0
        for i in self.files[folder]:
            name = list(i.values())[0][0]
            file_size = int(list(i.values())[0][1])
            if file_size != 0:
                size += file_size
            else:
                if self.folders[name] != 0:
                    size += self.folders[name]
                else:
                    return 0
        return size

    def part_1(self):
        """test part 1:
        >>> print(FileSystem(parsers.lines('test07.txt')).part_1())
        95437"""

        total = 0
        for folder in self.folders:
            if self.folders[folder] <= 100000:
                total += self.folders[folder]

        return total

    def part_2(self):
        """test part 2:
        >>> print(FileSystem(parsers.lines('test07.txt')).part_2())
        24933642"""

        total = 70000000
        needed = 30000000
        free = total - self.folders['//']
        target = needed - free

        candidates = []
        for folder in self.folders:
            if self.folders[folder] >= target:
                candidates.append(self.folders[folder])

        return min(candidates)


print(FileSystem(parsers.lines('input07.txt')).part_1())  # 1490523
print(FileSystem(parsers.lines('input07.txt')).part_2())  # 12390492
