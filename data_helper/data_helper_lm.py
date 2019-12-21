from tqdm import tqdm
import numpy as np
import os
from random import shuffle
# 语言模型部分

class DataSpeech():
    def __init__(self, data_root, mode, batch_size=1, shuffle=True):
        """
        初始化参数
        :param path: the directory that stores the data
        :param mode: A string, types of the mode, 'train', 'dev' or 'test'
        """
        self.data_root = data_root  # 数据存放位置根目录
        self.mode = mode       # 数据类型，分为三种：训练集(train)、验证集(dev)、测试集(test)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.pinyin_dict, self.charac_dict, self.pinyin_decode_dict, self.charac_decode_dict = self.get_pinyin_characs_list(data_root)  # 全部汉语拼音和汉字列表
        self.num_pinyin = len(self.pinyin_dict)
        self.num_vocab = len(self.charac_dict)
        self.pinyin_list, self.charac_list = self.get_trn_list()
        self.num_batch = len(self.pinyin_list) // self.batch_size

    def get_trn_list(self):
        """
        get the pinyin list and the corresponding chinese character list
        :return: two lists, a pinyin list and the chinese character list
        """
        data_path = os.path.join(self.data_root, self.mode)
        trn_files = [os.path.join(data_path, file) for file in os.listdir(data_path) if
                     file.endswith('.trn') or file.endswith('.txt')]
        pinyin_list = []
        charac_list = []
        for file in trn_files:
            with open(file, 'r', encoding='utf8') as f:
                data = f.readlines()
            for line in tqdm(data):
                _, pny, han = line.split('\t')
                pinyin_list.append(pny.split(' '))
                charac_list.append(han.strip('\n'))
        return pinyin_list, charac_list


    def get_pinyin_characs_list(self, dict_list_path):
        """
        加载拼音符号列表，用于标记符号, 返回一个列表list类型变量
        :param dict_list_path: the path of the dict list with pinyin and corresponding chinese characters
        :return: A list that contains the pinyin list
        """
        dict_list_path = os.path.join(dict_list_path, 'dict.txt')
        with open(dict_list_path, 'r', encoding='UTF-8') as txt_obj:  # 打开文件并读入
            txt_text = txt_obj.read()
            txt_lines = txt_text.split('\n')  # 文本分割
            pinyin_list = ['<PAD>']  # 初始化符号列表
            charac_list = ['<PAD>']

            for i in txt_lines:
                if (i != ''):
                    txt_line = i.split()
                    pinyin_list.append(txt_line[0])
                    charac_list.extend(txt_line[1])
            pinyin_dict = {}
            charac_dict = {}
            pinyin_decode_dict = {}
            charac_decode_dict = {}
            i = 0
            for index, pinyin in enumerate(pinyin_list):
                if not pinyin_dict.get(pinyin):
                    pinyin_dict[pinyin] = i
                    pinyin_decode_dict[i] = pinyin
                    i += 1
            i = 0
            for index, charac in enumerate(charac_list):
                if not charac_dict.get(charac):
                    charac_dict[charac] = i
                    charac_decode_dict[i] = charac
                    i += 1
        return pinyin_dict, charac_dict, pinyin_decode_dict, charac_decode_dict

    def pinyin2id(self, pinyin):
        """
        convert the pinyin to id
        :param pinyin: A string of pinyin
        :return: An int, the id of the pinyins
        """
        # pinyin_id = []
        # for i in pinyin:
        #     id = self.pinyin_dict[i]
        #     pinyin_id.append(id)
        pinyin_id = [self.pinyin_dict[i] if i in self.pinyin_dict else 0 for i in pinyin]
        return pinyin_id

    def charac2id(self, charac):
        """
        convert a chinese character to a id
        :param charac:
        :return: An int, the id of the chinese character
        """

        return [self.charac_dict[i] if i in self.charac_dict else 0 for i in charac]

    def label_padding(self, max_label_length, label):
        """
        pad the label so that the labels in a batch have the same length
        :param max_label_length:
        :param label: A ndarray, acoustic or character label
        :return: A ndarray, padded acoustic or character label
        """
        if len(label) < max_label_length:
            padding_length = max_label_length - len(label)
            padded_label = np.pad(label, (0, padding_length), 'constant', constant_values=0)
            return padded_label
        return np.array(label)


    def get_lm_batch(self):
        for i in range(self.num_batch):
            begin = i * self.batch_size
            end = begin + self.batch_size
            batch_pinyin_list = self.pinyin_list[begin:end]
            batch_charac_list = self.charac_list[begin:end]
            # convert the pinyin and character to id
            batch_pinyin_id = [self.pinyin2id(pinyin) for pinyin in batch_pinyin_list]
            batch_charac_id = [self.charac2id(charac) for charac in batch_charac_list]

            pinyin_length = [len(pinyin_id) for pinyin_id in batch_pinyin_id]  # label acoustic and label language have the same length
            pinyin_length = np.array(pinyin_length)
            # pad the label
            max_pinyin_len = max(pinyin_length)
            input_pinyin = [self.label_padding(max_pinyin_len, pinyin_id) for pinyin_id in batch_pinyin_id]

            label_language = [self.label_padding(max_pinyin_len, charac_id) for charac_id in batch_charac_id]

            input_pinyin = np.array(input_pinyin)
            label_language = np.array(label_language)

            lm_batch = {'input_pinyin': input_pinyin,
                        'label_language': label_language
                        }
            yield lm_batch


if __name__ == '__main__':
    data_obj = DataSpeech('D:\Materials\deep_learning\ASRC\dataset_lm','train', batch_size=3)




