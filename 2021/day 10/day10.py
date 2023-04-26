from tools import parsers, loader

MATCHES = ('()', '[]', '<>', '{}')
SCORE = {
    ')': 3,
    ']': 57,
    '}': 1197,
    '>': 25137
}


class Brackets:
    def remove_closed(self, string: str) -> str:
        new_string = string
        for character in MATCHES:
            new_string = new_string.replace(character, '')
        if new_string == string:
            return new_string
        return self.remove_closed(new_string)

    def part_1(self, data) -> int:
        """
        >>> print(Brackets().part_1(parsers.lines('test.txt')))
        26397"""
        result = 0
        for line in data:
            compressed = self.remove_closed(line)
            for i in compressed:
                if i in SCORE.keys():
                    result += SCORE[i]
                    break
        return result


print(Brackets().part_1(parsers.lines(loader.get())))  # 367227
