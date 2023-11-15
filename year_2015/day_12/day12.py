import json

from tools import loader, parsers


def accounting(datum: dict | list | int | str, part2: bool) -> int:
    match datum:
        case dict() if not part2 or ('red' not in datum.values() and part2):
            return sum(accounting(i, part2) for i in datum.values())
        case list():
            return sum(accounting(i, part2) for i in datum)
        case int():
            return datum
        case _:
            return 0


print(accounting(json.loads(parsers.string(loader.get())), part2=False))  # 111754
print(accounting(json.loads(parsers.string(loader.get())), part2=True))  # 65402
