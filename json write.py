
with open('.\data\datas.txt', "r") as txt:
    lines = txt.readlines()
    with open('.\data\datas2.txt', "w") as txt2:
        for line in lines:
            if(line[0] != '['):
                if(line[0] != ']'):
                    if (line[4] == '}'):
                        line = "    },\n"
                        
            txt2.write(line)
    

    
