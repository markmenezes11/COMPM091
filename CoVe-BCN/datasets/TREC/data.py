import io

with io.open("train-dev.txt", 'r', encoding='utf-8') as f:
    classes = {}
    num = 0
    lines = f.readlines()
    for line in lines[:int(0.2*len(lines))]:
        sample = line.strip().split(' ', 1)
        classs = sample[0]
        if classs not in classes:
            classes[classs] = num
            num += 1
        print([x.lower() for x in sample[1].split()])
        print(sample[1].split())

print(classes)
print(num)
