# data helper
data_root = "D:\Materials\deep_learning\ASRC\dataset_lm"
ckpt_path = "D:\Materials\deep_learning\ASRC/logs_lm"
mode = "train"           # "train", "val", "test"

batch_size = 50
shuffle = True
epoch = 2

# lm parameter
num_heads = 8
num_blocks = 3
d_model = 512  # dimention of the embedding layer, also of queries and keys
d_ff = 256   # the dimention of the FF layer
dropout_rate = 0

## ckpt_dir = ""
# ckpt_prefix = os.path.join(ckpt_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
# if (epoch + 1) % 2 == 0:
#   checkpoint.save(file_prefix=ckpt_prefix)

# checkpoint = tf.train.Checkpoint(model=model, optimizer=opt)
# latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)


