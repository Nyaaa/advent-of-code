import re
from collections.abc import Generator
from string import ascii_lowercase

from more_itertools import windowed

from tools import loader, parsers


def password_generator(password: str) -> Generator[str]:
    password = list(password)
    while True:
        for i in range(len(password) - 1, 0, -1):
            password[i] = chr((ord(password[i]) + 1 - 97) % 26 + 97)
            if password[i] != 'a':
                break
        yield ''.join(password)


def get_new_password(password: str) -> str:
    straight = {''.join(i) for i in windowed(ascii_lowercase, 3)}
    is_valid = False
    pw_gen = password_generator(password)
    while not is_valid:
        password = next(pw_gen)
        is_valid = (any(i in password for i in straight)
                    and not re.search(r'[iol]', password)
                    and len(re.findall(r'(.)\1', password)) >= 2)
    return password


part1 = get_new_password(parsers.string(loader.get()))
print(part1)  # vzbxxyzz
print(get_new_password(part1))  # vzcaabcc
