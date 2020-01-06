import os
from tqdm import tqdm

data_path = "D:\Materials\deep_learning\ASRC\dataset\lm/test"
dest_path = "D:\Materials\deep_learning\ASRC\dataset/am\新建文件夹"
trn_files = [os.path.join(data_path, file) for file in os.listdir(data_path) if
             file.endswith('.trn') or file.endswith('.txt')]
pinyin_list = []
charac_list = []
for file in trn_files:
    file_name = os.path.basename(file)[:-4]
    dest_path_specific = os.path.join(dest_path, file_name)
    if not os.path.exists(dest_path_specific):
        os.makedirs(dest_path_specific)
    with open(file, 'r', encoding='utf8') as f:
        data = f.readlines()
    for line in tqdm(data):
        wav_file, pinyin, characs = line.split('\t')
        trn_file_name = os.path.basename(wav_file)
        with open(os.path.join(dest_path_specific, trn_file_name + ".trn"), 'wt') as f:
            f.write(characs)
            f.write(pinyin)
