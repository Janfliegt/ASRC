import os
from tqdm import tqdm

pinyin_dict = {}
charac_dict = {}
trn_dir = "D:\Materials\deep_learning\ASRC\dataset\lm/train"
def get_trn(trn_file, pinyin_dict, charac_dict):
        with open(trn_file, encoding='utf-8') as trn_obj:
            trn_lines = trn_obj.readlines()
            for line in tqdm(trn_lines):
                if (line != ''):
                    _, trn_pinyin, characs = line.split('\t')
                    trn_pinyin = trn_pinyin.split(' ')
                    characs = characs.strip('\n')
                    for pinyin in trn_pinyin:
                        if pinyin not in pinyin_dict:
                            pinyin_dict[pinyin] = 1
                    for charac in characs:
                        if charac not in charac_dict:
                            charac_dict[charac] = 1
        return pinyin_dict, charac_dict

trn_files = [os.path.join(trn_dir, file) for file in os.listdir(trn_dir) if
                         file.endswith('.trn') or file.endswith('.txt')]
for trn_file in trn_files:
    pinyin_dict, charac_dict = get_trn(trn_file, pinyin_dict, charac_dict)
dict_file = "pinyin.txt"
pinyin = list(pinyin_dict.keys())
pinyin = sorted(pinyin)
pinyin = "\n".join(pinyin)
with open(dict_file, "w", encoding="utf-8") as f:
    f.write(pinyin)

dict_file = "charac.txt"
characs = list(charac_dict.keys())
characs = sorted(characs)
characs = "\n".join(characs)
with open(dict_file, "w", encoding="utf-8") as f:
    f.write(characs)
