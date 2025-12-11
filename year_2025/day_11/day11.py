from functools import cache

from tools import loader, parsers


def network(data: list[str], start: str) -> int:
    """
    >>> print(network(parsers.lines('test.txt'), start='you'))
    5
    >>> print(network(parsers.lines('test2.txt'), start='svr'))
    2
    """
    graph = {left: list(right.split(' ')) for (left, right) in (line.split(': ') for line in data)}

    @cache
    def count_paths(node: str, visits: int) -> int:
        match node:
            case 'dac': visits += 1
            case 'fft': visits += 1
            case 'out': return 1 if visits == 2 else 0
        return sum(count_paths(i, visits) for i in graph[node])

    return count_paths(node=start, visits=0 if start == 'svr' else 2)


print(network(parsers.lines(loader.get()), start='you'))  # 634
print(network(parsers.lines(loader.get()), start='svr'))  # 377452269415704
