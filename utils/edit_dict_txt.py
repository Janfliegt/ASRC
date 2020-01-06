
dict_file = "D:\Materials\deep_learning\ASRC\dataset/am/dict.txt"
dict_obj =  open(dict_file, 'r+', encoding='utf-8')
file_data = ""
dic_lines = dict_obj.readlines()
for line in dic_lines:
    pinyin, characs = line.split()
    if pinyin.endswith('5'):
        line = pinyin[:-1] + '\t' + characs + '\n'
    file_data += line
dict_obj.close()
with open(dict_file, "w", encoding="utf-8") as f:
    f.write(file_data)




