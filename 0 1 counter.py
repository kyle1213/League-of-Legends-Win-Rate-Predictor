import json
a = 0
b = 0
c = 0
with open('.\data\output2.json') as json_file:
    json_data = json.load(json_file)
    print(len(json_data))
    for item in json_data:
        if(item['array'][10] == 0):
            a = a + 1
        if(item['array'][10] == 1):
            b = b + 1
print(a)
print(b)
print(a+b)