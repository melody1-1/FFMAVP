f = open("./data/test dataset/firstStage/firstStage_test.faa")
line = f.readline()
list1 = []
while line:
    line = line.strip('\n')
    list1.append(line)
    line = f.readline()
f.close()
print(list1)

longest_num = max(list1, key=len)

len_longest_num = 121

list2 = [i.ljust(len_longest_num, 'O') if i[0] != '>' else i for i in list1]
with open("./data/test dataset/firstStage/firstStage_test_121.faa", 'w') as f:
    for i in list2:
        f.write(i+'\n')