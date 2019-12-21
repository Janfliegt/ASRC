import os
import numpy as np
import shutil
# import re
data_path = "/home/jl223573/Documents/materials/machine-learning/ASRE/dataset/data_thchs30/train"
dst_path = "/home/jl223573/Documents/materials/machine-learning/ASRE/dataset/data_thchs30/val"

files= os.listdir(data_path)   # 得到文件夹下的所有文件名称
for file in files:  # 遍历文件夹
     if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
         if file.endswith('.trn'):
             file_name = os.path.join(data_path, file)
             random_number = np.random.uniform()
             if random_number > 0.9:
                 shutil.move(file_name, dst_path)
                 print("Move ", file_name, "to val")
                 wav_file_name = file_name[:-4]
                 shutil.move(wav_file_name, dst_path)
                 print("Move ", wav_file_name, "to val")

