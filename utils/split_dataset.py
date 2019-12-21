import os
import numpy as np
import shutil
# import re
data_path = "/home/jl223573/Documents/materials/machine-learning/ASRE/dataset/data_thchs30/train"
dst_path = "/home/jl223573/Documents/materials/machine-learning/ASRE/dataset/data_thchs30/val"
trn_train_dict = {}  # 保存训练集里边的transcript
trn_val_dict = {}  # 保存验证集里边的transcript
# trn_files = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.trn')]


def get_trn(trn_path):
    """
    get the transcription of the wav files
    :param trn_path: A string. The path of the transcription file
    :return: A tuple, the first element is the pinyin transcription, the second is the characters trn
    """
    trn_charac_single = []
    with open(trn_path, encoding='utf-8') as trn_obj:
        trn_text = trn_obj.read()
        trn_lines = trn_text.split('\n')  # 文本分割
        trn_pinyin = trn_lines[1].split()
    trn_obj.close()
    return trn_pinyin




files= os.listdir(data_path)  # 得到文件夹下的所有文件名称
for file in files: # 遍历文件夹
     if not os.path.isdir(file): # 判断是否是文件夹，不是文件夹才打开
         if file.endswith('.trn'):
             file_name = os.path.join(data_path, file)
             pinyin_seq = ''.join(get_trn(file_name))

             if pinyin_seq in trn_val_dict:
                 shutil.move(file_name, dst_path)
                 wav_file_name = file_name[:-4]
                 shutil.move(wav_file_name, dst_path)
                 print("Move ", file_name, "to val")
             elif pinyin_seq in trn_train_dict:
                 pass
             else:
                 random_number = np.random.uniform()
                 if random_number > 0.2:
                     trn_train_dict[pinyin_seq] = 1
                 else:
                     trn_val_dict[pinyin_seq] = 1
                     shutil.move(file_name, dst_path)
                     wav_file_name = file_name[:-4]
                     shutil.move(wav_file_name, dst_path)
                     print("Move ", file_name, "to val")

