from tools import parsers, loader


def parse(data: list) -> list[dict]:
    passports = []
    for block in data:
        strings = [val for row in block for val in row.split()]  # flatten list & split
        passports.append({i: j for i, j in (s.split(':') for s in strings)})
    return passports


def validate(passport: dict) -> bool:
    valid_full = {'byr', 'iyr', 'eyr', 'hgt', 'hcl', 'ecl', 'pid', 'cid'}
    keys = set(passport.keys())
    diff = valid_full.difference(keys)
    if not diff or diff == {'cid'}:
        return True
    return False


def start(data: list) -> int:
    passports = parse(data)
    valid = [validate(passport) for passport in passports]
    return sum(valid)


print(start(parsers.blocks('test.txt')))
print(start(parsers.blocks(loader.get())))  # 206
