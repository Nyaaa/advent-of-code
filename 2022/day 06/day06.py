with open('input06.txt') as f:
    data = f.read().strip()


def get_chunk(_i: int, length: int):
    try:
        return [data[_i + n] for n in range(length)]
    except IndexError:
        pass

# part 1


for i in range(len(data)):
    chunk = get_chunk(i, 4)
    dupes = set(chunk)
    if len(dupes) == len(chunk):
        offset = i + 4
        print(offset)  # 1779
        break

# part 2


for i in range(len(data)):
    chunk = get_chunk(i, 14)
    dupes = set(chunk)
    if len(dupes) == len(chunk):
        offset = i + 14
        print(offset)  # 2635
        break
