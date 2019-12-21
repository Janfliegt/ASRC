# data helper
data_root = "D:\Materials\deep_learning\ASRC\dataset_am"
mode = 'val'           # "train", "dev", "test"
model = "cnn_ctc"  # "transformer_encoder" ,"cnn_ctc"
opt_name = 'adam'  # the name of used optimizer
learning_rate = None  # if None, the default learning rate willl be used
ckpt_path = 'D:\Materials\deep_learning\ASRC/logs_am/'
specified_datasets = []
batch_size = 1
shuffle = True
epoch = 100
num_cep = 26
winlen = 0.025
winstep = 0.01
audio_feat = "fbank"
# if audio_feat == "logfbank" or audio_feat == "fbank":
#     num_cep = 26
attention_size = 10
dropout_rate = 0.3
# acoustic model parameter
enc_units = 4
atten_units = 3
dec_units = 10
embedding_dim = 128

# lm parameter
num_heads = 8
num_blocks = 3
d_model = 1024  # dimention of the embedding layer, also of queries and keys
d_ff = 256   # the dimention of the FF laye
