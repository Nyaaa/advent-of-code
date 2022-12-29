# TODO refactor this mess!

# with open('test07.txt') as f:
with open('input07.txt') as f:
    data = f.read().splitlines()

files = {}
folders = {}

path = ''
for line in data:
    line = line.split()

    if line[0] == '$':
        type = 'cmd'
    elif line[0] == 'dir':
        type = 'dir'
    else:
        type = 'file'

    if type == 'cmd':
        if line[1] == 'cd' and line[2] == '..':
            path = path.rsplit('/', 1)[0]
        elif  line[1] == 'cd':
            dir = line[2]
            if dir != '/':
                path += f'/{dir}'
            else:
                path = 'root'
            files[path] = []
            folders[path] = 0
        elif  line[1] == 'ls':
            continue
    else:
        name = f'{path}/{line[1]}'
        if type == 'file':
            size = int(line[0])
        else:
            size = 0

        files[path].append({type: [name, size]})


def get_size(folder):
    size = 0
    for i in files[folder]:
        name = list(i.values())[0][0]
        file_size = int(list(i.values())[0][1])
        if file_size != 0:
            size += file_size
        else:
            if folders[name] != 0:
                size += folders[name]
            else:
                return 0
    return size


while True:
    for folder in folders:
        size = get_size(folder)
        folders.update({folder: size})
    if folders['root'] != 0:
        break

# part 1

sum = 0

for folder in folders:
    if folders[folder] <= 100000:
        sum += folders[folder]

print(sum)  # 1490523

# part 2

total = 70000000
needed = 30000000
free = total - folders['root']
target = needed - free

candidates = []
for folder in folders:
    if folders[folder] >= target:
        candidates.append(folders[folder])

print(min(candidates))  # 12390492