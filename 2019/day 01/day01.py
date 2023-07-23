from tools import parsers, loader


def calculate_fuel(module: int) -> int:
    return module // 3 - 2


def calculate_mass(data: list) -> int:
    """
    >>> print(calculate_mass(['100756']))
    33583"""
    return sum(calculate_fuel(int(line)) for line in data)


def calculate_mass_with_fuel(data: list) -> int:
    """
    >>> print(calculate_mass_with_fuel(['100756']))
    50346"""
    total_fuel = 0
    for fuel in data:
        module_fuel = 0
        while True:
            fuel = calculate_fuel(int(fuel))
            if fuel <= 0:
                break
            module_fuel += fuel
        total_fuel += module_fuel
    return total_fuel


print(calculate_mass(parsers.lines(loader.get())))  # 3268951
print(calculate_mass_with_fuel(parsers.lines(loader.get())))  # 4900568
