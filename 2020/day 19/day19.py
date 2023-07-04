import regex as re

from tools import parsers, loader


class Message:
    def __init__(self, data: list):
        self.rules = dict()
        self.messages = data[1]
        for line in data[0]:
            name, sets = line.split(': ')
            sets = re.sub(r'[\'\"]', '', sets)
            if '|' in sets:
                sets = f'(?: {sets} )'
            self.rules[name] = sets
        self.pattern = self.compose_pattern()

    def compose_pattern(self):
        pattern = self.rules['0']
        while match := re.search(r'\d+', pattern):
            val = self.rules[match.group()]
            pattern = f'{pattern[:match.start()]} {val} {pattern[match.end():]}'
        return re.compile(f'^{pattern}$'.replace(' ', ''))

    def part_1(self):
        """
        >>> print(Message(parsers.blocks('test.txt')).part_1())
        3"""
        return sum((1 for message in self.messages if self.pattern.match(message)))

    def part_2(self):
        """
        >>> print(Message(parsers.blocks('test.txt')).part_2())
        12"""
        self.rules['8'] = '(?: 42 )+'
        self.rules['11'] = '(?P <R> 42 (?&R)? 31 )'  # recursive regexp
        self.pattern = self.compose_pattern()
        return sum((1 for message in self.messages if self.pattern.match(message)))


print(Message(parsers.blocks(loader.get())).part_1())  # 230
print(Message(parsers.blocks(loader.get())).part_2())  # 341
