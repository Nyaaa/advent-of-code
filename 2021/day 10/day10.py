from tools import loader, parsers

MATCHES = ('()', '[]', '<>', '{}')
SCORE = {
    ')': 3,
    ']': 57,
    '}': 1197,
    '>': 25137
}
CLOSE = {
    '(': 1,
    '[': 2,
    '{': 3,
    '<': 4
}


class Brackets:
    def remove_closed(self, string: str) -> str:
        new_string = string
        for character in MATCHES:
            new_string = new_string.replace(character, '')
        if new_string == string:
            return new_string
        return self.remove_closed(new_string)

    def part_1(self, data: list[str]) -> int:
        """
        >>> print(Brackets().part_1(parsers.lines('test.txt')))
        26397"""
        result = 0
        for line in data:
            compressed = self.remove_closed(line)
            for i in compressed:
                if i in SCORE:
                    result += SCORE[i]
                    break
        return result

    def part_2(self, data: list[str]) -> int:
        """
        >>> print(Brackets().part_2(parsers.lines('test.txt')))
        288957"""
        scores = []
        for line in data:
            compressed = self.remove_closed(line)
            if all(i not in SCORE for i in compressed):
                line_score = 0
                for i in compressed[::-1]:
                    line_score = line_score * 5 + CLOSE[i]
                scores.append(line_score)

        scores.sort()
        mid = len(scores) // 2
        return scores[mid]


print(Brackets().part_1(parsers.lines(loader.get())))  # 367227
print(Brackets().part_2(parsers.lines(loader.get())))  # 3583341858
