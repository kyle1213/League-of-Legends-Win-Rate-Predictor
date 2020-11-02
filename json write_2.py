import json
value = 0
arr = [[]*10]*41000
with open('.\data\output2.json', "r") as json_file:
    json_data = json.load(json_file)
    for j in range(len(json_data)):
        arr[j] = json_data[j]['array']
        for i in range(5, 10):
            arr[j][i] = -json_data[j]['array'][i]


flag = 0
flag2 = 0
j = 0
with open('.\data\datas.txt', "r") as txt:
    lines = txt.readlines()
    with open('.\data\datas2.txt', "w") as txt2:
        for line in lines:
            if(line[0] != '['):
                if(line[0] != ']'):
                    if (line[4] != '{'):
                        if(line[4] != '}'):
                            if (line[9] == 'a'):
                                for i in range(len(line)):
                                    if(flag >= 5):
                                        if(line[i] == ','):
                                            if(line[i+1] == '0'):
                                                continue
                                            line = line[:flag2+1] + ' -' + line[flag2 + 2:]
                                            flag2 = i + 1
                                            flag = flag + 1
                                    elif(line[i] == ','):
                                        flag2 = i
                                        flag = flag + 1
                                    
                                    
                                flag2 = 0
                                flag = 0
                        
            txt2.write(line)
    

