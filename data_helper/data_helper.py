from python_speech_features import mfcc, fbank, logfbank
import scipy.io.wavfile as wav
from scipy.fftpack import fft
import numpy as np
import os
from random import shuffle
import re
import librosa

# 声学部分
# 提取音频文件提取mfcc特征

def compute_audio_feat(audio, num_cep, winlen=0.025, winstep=0.01, audio_feat="spectrogram"):
    """
    compute the mfcc feature of a wav file
    :param file: A string. The path of a wav file
    :return: An array. The mfcc of the wav file
    """
    if audio_feat == "logfbank":
        audio_feat = logfbank(audio, samplerate=16000, winlen=winlen, winstep=winstep)

    elif audio_feat == "mfcc":
        audio_feat = mfcc(audio, samplerate=16000, numcep=num_cep, winlen=winlen, winstep=winstep)

    elif audio_feat == "fbank":
        # audio_feat, _ = fbank(audio, samplerate=16000, winlen=winlen, winstep=winstep)
        audio_feat = compute_fbank(audio, fs=16000)

    elif audio_feat == "spectrogram":
        tmp = librosa.stft(y=audio, n_fft=399, hop_length=160)
        audio_feat = (librosa.amplitude_to_db(np.abs(tmp), ref=np.max)).T
    pad_fbank = np.zeros((audio_feat.shape[0] // 8 * 8 + 8, audio_feat.shape[1]))
    pad_fbank[:audio_feat.shape[0], :] = audio_feat

    return pad_fbank

def compute_fbank(wavsignal, fs):
    x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))  # 汉明窗
    # fs, wavsignal = wav.read(file)
    # wav波形 加时间窗以及时移10ms
    time_window = 25  # 单位ms
    wav_arr = np.array(wavsignal)
    range0_end = int(len(wavsignal) / fs * 1000 - time_window) // 10 + 1 # 计算循环终止的位置，也就是最终生成的窗数
    data_input = np.zeros((range0_end, 200), dtype=np.float)  # 用于存放最终的频率特征数据
    data_line = np.zeros((1, 400), dtype=np.float)
    for i in range(0, range0_end):
        p_start = i * 160
        p_end = p_start + 400
        data_line = wav_arr[p_start:p_end]
        data_line = data_line * w  # 加窗
        data_line = np.abs(fft(data_line))
        data_input[i] = data_line[0:200]  # 设置为400除以2的值（即200）是取一半数据，因为是对称的
    data_input = np.log(data_input + 1)
    # data_input = data_input[::]
    return data_input

# 语言模型部分
class DataSpeech():
    def __init__(self, data_root, mode, specified_datasets=[], batch_size=1, shuffle=True, audio_feat="fbank", num_cep=26, winlen=0.025, winstep=0.01, nfft=512):
        """
        初始化参数
        :param path: the directory that stores the data
        :param mode: A string, types of the mode, 'train', 'dev' or 'test'
        :param specified_datasets: A list, which will contained the used datasets, if not specified.
                it means all the datsets will be used
        :param LoadToMem:
        :param MemWavCount:
        """
        self.data_root = data_root  # 数据存放位置根目录
        self.mode = mode       # 数据类型，分为三种：训练集(train)、验证集(dev)、测试集(test)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.audio_feat = audio_feat
        self.num_cep=num_cep
        self.winlen = winlen
        self.winstep = winstep


        if specified_datasets:
            self.datasets = specified_datasets
        else:
            dirs_plus_files = os.listdir(data_root)
            self.datasets = [dataset for dataset in dirs_plus_files if os.path.isdir(os.path.join(self.data_root, dataset))]

        self.pinyin_dict, self.charac_dict, self.pinyin_decode_dict, self.charac_decode_dict = self.get_pinyin_characs_list(data_root)  # 全部汉语拼音和汉字列表
        self.num_pinyin = len(self.pinyin_dict)
        self.num_vocab = len(self.charac_dict)
        self.wav_list, self.wav_trn_dict = self.get_wav_trn_dict()
        self.num_batch = len(self.wav_list) // self.batch_size

        pass

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
            pinyin_list = []  # 初始化符号列表
            charac_list = ['<PAD>']

            for i in txt_lines:
                if (i != ''):
                    txt_line = i.split()
                    pinyin_list.append(txt_line[0])
                    charac_list.extend(txt_line[1])
            pinyin_list.append('_')
            charac_list.append('_')
            #
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
        return self.pinyin_dict[pinyin]

    def charac2id(self, charac):
        """
        convert a chinese character to a id
        :param charac:
        :return: An int, the id of the chinese character
        """
        return self.charac_dict[charac]

    def get_wav_trn_dict(self):
        """
        get the wavfile and the transcription dict that will be used
        :return: A dic, whose keys are the wavfile list and the values are the trn
        """
        wav_trn_dict = {}  # a dict with the key is the wav file and the value is its pinyin and characaters transription

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
                trn_pinyin_id = [self.pinyin2id(pinyin) for pinyin in trn_pinyin]
                trn_charac = trn_lines[0].split()
                for word in trn_charac:
                    trn_charac_single.extend(word)
                trn_charac_id = [self.charac2id(charac) for charac in trn_charac_single]
            return trn_pinyin_id, trn_charac_id

        for dataset in self.datasets:
            data_path = os.path.join(self.data_root, dataset, self.mode)
            trn_files = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.trn') or file.endswith('.txt')]
            trn_list = list(map(get_trn, trn_files))
            wav_list = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith('.wav')]

            for i in range(len(wav_list)):
                wav_trn_dict[wav_list[i]] = trn_list[i]
        return wav_list, wav_trn_dict

    def resample_wavfile(self, wavefile_path):
        """
        resample the wavefiles to ensure that they have the same samplerate
        :param wavefile_path: A string, the path of the wavefile
        :return:
        """
        fs, audio = wav.read(wavefile_path)
        audio_librosa, _ = librosa.load(wavefile_path, sr=fs)
        fs_uniform = 16000
        audio_resampled = librosa.resample(audio_librosa, fs, fs_uniform)
        return audio_resampled

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

    def get_am_batch(self):
        if self.shuffle:
            shuffle(self.wav_list)

        def audio_padding(max_feat_len, audio_feat):
            """
            pad the audio feature so that they have the same length
            :param max_length: the max length of the audio feature in a batch
            :param audio: ndarray, audio feature
            :return: padded audio feature
            """
            if len(audio_feat) < max_feat_len:
                padded_audio_feat = np.zeros([max_feat_len, audio_feat.shape[1]])
                padded_audio_feat[:audio_feat.shape[0], :] = audio_feat
                return padded_audio_feat
            else:
                return audio_feat
        def libsora_load(wavfile):
            audio, sr = librosa.load(wavfile, sr=None)
            return audio, sr

        for i in range(self.num_batch):
            begin = i * self.batch_size
            end = begin + self.batch_size
            batch_wav_list = self.wav_list[begin:end]
            # resampled_audio = list(map(self.resample_wavfile, batch_wav_list))
            # mfcc_batch = [compute_mfcc(audio, self.num_cep, self.winlen, self.winstep) for audio in resampled_audio]
            # fs_audio_sample = list(map(wav.read, batch_wav_list))
            fs_audio_librosa= list(map(libsora_load, batch_wav_list))
            # audio_sample = [i[1] for i in fs_audio_sample]
            audio_sample = [i[0] for i in fs_audio_librosa]
            audio_feat_batch = [compute_audio_feat(audio, self.num_cep, self.winlen, self.winstep, self.audio_feat) for audio in audio_sample]
            # get the input length of the encoder

            input_length = [audio_feat.shape[0] for audio_feat in audio_feat_batch]
            input_length = np.array(input_length)
            # pad the input with zeros so that they have the same length
            max_audion_length = max(input_length)
            # input_length = np.ones_like(input_length) * max_audion_length
            padded_audio_feat = [audio_padding(max_audion_length, audio_feat) for audio_feat in audio_feat_batch]
            padded_audio_feat = np.array(padded_audio_feat)

            # get the label
            label = list(map(self.wav_trn_dict.get, batch_wav_list))
            label_acoustic = [i[0] for i in label]
            # label_language = [i[1] for i in label]
            label_length = [len(label) for label in label_acoustic]
            label_length = np.array(label_length)
            # pad the label
            max_label_len = max(label_length)
            label_acoustic = [self.label_padding(max_label_len, label) for label in label_acoustic]
            label_acoustic = np.array(label_acoustic)

            input_batch = {'input': padded_audio_feat,
                           'input_length': input_length,
                           'label': label_acoustic,
                           'label_length': label_length
                           }
            yield input_batch

    def get_batch(self):
        if self.shuffle:
            shuffle(self.wav_list)

        def audio_padding(max_feat_len, audio_feat):
            """
            pad the audio feature so that they have the same length
            :param max_length: the max length of the audio feature in a batch
            :param audio: ndarray, audio feature
            :return: padded audio feature
            """
            if len(audio_feat) < max_feat_len:
                padded_audio_feat = np.zeros([max_feat_len, audio_feat.shape[1]])
                padded_audio_feat[:audio_feat.shape[0], :] = audio_feat
                return padded_audio_feat
            else:
                return audio_feat
        def libsora_load(wavfile):
            audio, sr = librosa.load(wavfile, sr=None)
            return audio, sr

        for i in range(self.num_batch):
            begin = i * self.batch_size
            end = begin + self.batch_size
            batch_wav_list = self.wav_list[begin:end]
            # resampled_audio = list(map(self.resample_wavfile, batch_wav_list))
            # mfcc_batch = [compute_mfcc(audio, self.num_cep, self.winlen, self.winstep) for audio in resampled_audio]
            # fs_audio_sample = list(map(wav.read, batch_wav_list))
            fs_audio_librosa= list(map(libsora_load, batch_wav_list))
            # audio_sample = [i[1] for i in fs_audio_sample]
            audio_sample = [i[0] for i in fs_audio_librosa]
            audio_feat_batch = [compute_audio_feat(audio, self.num_cep, self.winlen, self.winstep, self.audio_feat) for audio in audio_sample]
            # get the input length of the encoder

            input_length = [audio_feat.shape[0] for audio_feat in audio_feat_batch]
            input_length = np.array(input_length)
            # pad the input with zeros so that they have the same length
            max_audion_length = max(input_length)
            # input_length = np.ones_like(input_length) * max_audion_length
            padded_audio_feat = [audio_padding(max_audion_length, audio_feat) for audio_feat in audio_feat_batch]
            padded_audio_feat = np.array(padded_audio_feat)

            # get the label
            label = list(map(self.wav_trn_dict.get, batch_wav_list))

            label_language = [i[1] for i in label]
            label_length = [len(label) for label in label_language]
            label_length = np.array(label_length)
            # pad the label
            max_label_len = max(label_length)
            label_language = [self.label_padding(max_label_len, label) for label in label_language]
            label_language = np.array(label_language)

            input_batch = {'input': padded_audio_feat,
                           'input_length': input_length,
                           'label': label_language,
                           'label_length': label_length
                           }
            yield input_batch

if __name__ == '__main__':
    data_obj = DataSpeech('D:\Materials\deep_learning\ASRE\dataset','train', batch_size=3)
    am_batch = data_obj.get_am_batch()
    mfcc_feat, label_am, label_lm = next(am_batch)

    print("sd")


