with open('input06.txt') as f:
    data = f.read().strip()


def get_chunk(i: int, length: int):
    chunk = []
    try:
        for n in range(length):
            j = i + n
            chunk.append(data[j])
    except IndexError:
        pass
    else:
        return chunk

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
