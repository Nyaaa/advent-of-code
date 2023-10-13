from tools import loader, parsers


def firewall(data: list[str]) -> tuple[int, int]:
    blacklist = sorted([tuple(map(int, line.split('-'))) for line in data])
    potential = [i[1] + 1 for i in blacklist]
    # Apparently, my input has no open gaps larger than one ip, so no additional loop is needed
    unblocked = [i for i in potential if
                 not any(bottom <= i <= top for bottom, top in blacklist)
                 and i < 4294967295]
    return unblocked[0], len(unblocked)


print(firewall(parsers.lines(loader.get())))  # 31053880, 117
