import sys
import os
import tensorflow as tf
from os.path import abspath, join, dirname
import importlib
from models.am import cnn_ctc, transformer_am, DNN, cnn_ctc_test
from data_helper import data_helper_am
from others.WER import wer


sys.path.insert(0, join(abspath(dirname(__file__)), 'config'))
gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = importlib.import_module((sys.argv[1]))

def get_data(input_batch, model_name):
    input_tensor = tf.convert_to_tensor(input_batch['input'], dtype=tf.float32)
    label_tensor = tf.convert_to_tensor(input_batch['label'], dtype=tf.int32)
    input_length_decode = tf.constant(input_batch['input_length'], dtype=tf.int32)
    if model_name == "cnn_ctc":
        input_length_decode = input_length_decode // 8
    input_length = tf.expand_dims(input_length_decode, -1)
    label_length = tf.expand_dims(tf.constant(input_batch['label_length'], dtype=tf.int32), -1)
    return input_tensor, label_tensor, input_length, label_length, input_length_decode
# @tf.function
def train_step():
    """
    :param input_tensor: padded_mfcc_feat
    :param label_tensor: label_acoustic
    :param input_length: input_length
    :param label_length: label_length
    :param enc_hidden: A Tensor, the hidden state of the encoder
    :return: the batch loss
    """
    def train(input_tensor, label_tensor, input_length, label_length):
        with tf.GradientTape() as tape:
            logits = model(input_tensor)
            # input_length = tf.ones([data_obj.batch_size, 1], dtype=tf.int32) * logits.shape[1]
            ctc_loss = tf.keras.backend.ctc_batch_cost(y_true=label_tensor,
                                                       y_pred=logits,
                                                       input_length=input_length,
                                                       label_length=label_length)
            loss = tf.reduce_mean(ctc_loss)

        variables = model.trainable_variables
        gradients = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(gradients, variables))
        train_loss(ctc_loss)
        return logits

    train_loss = tf.keras.metrics.Mean("loss")
    for epoch in range(config.epoch):
        step = 1
        dataset = data_obj.get_am_batch()
        for input_batch in dataset:
            input_tensor, label_tensor, input_length, label_length, input_length_decode = get_data(input_batch, config.model)

            logits = train(input_tensor, label_tensor, input_length, label_length)

            print("epoch", epoch, "step:", step, "batch_loss:", train_loss.result().numpy())
            sys.stdout.flush()
            step += 1
            train_loss.reset_states()
        wer_mean, wer_mean_actual = wer(logits, input_length_decode, label_tensor)
        print("epoch", epoch, "WER: ", wer_mean, "WER_actual: ", wer_mean_actual)
        if (epoch + 1) % config.save_every == 0:
            checkpoint.save(os.path.join(
                            config.ckpt_path, config.model,
                            "epoch_{}.ckpt".format(epoch)))

def val_step():
    mean_loss = tf.keras.metrics.Mean("loss")
    def val(input_tensor, label_tensor, label_length):
        """
        :param input_tensor: padded_mfcc_feat
        :param label_tensor: label_acoustic
        :param label_length: label_length
        """
        logits= model(input_tensor)
        input_length = tf.ones([data_obj.batch_size, 1], dtype=tf.int32) * logits.shape[1]
        loss = tf.reduce_sum(tf.keras.backend.ctc_batch_cost(y_true=label_tensor,
                                                             y_pred=logits,
                                                             input_length=input_length,
                                                             label_length=label_length))

        mean_loss(loss)
    step = 1
    dataset = data_obj.get_am_batch()
    for input_batch in dataset:
        input_tensor, label_tensor, input_length, label_length, input_length_decode = get_data(input_batch)
        val(input_tensor, label_tensor, label_length)

        print("step:", step, "batch_loss:", mean_loss.result().numpy())
        sys.stdout.flush()
        step += 1
        mean_loss.reset_states()

def create_model(model_name):
    if model_name == 'cnn_ctc':
        model = cnn_ctc.CNN_CTC(data_obj.num_pinyin, config.dropout_rate)

    elif model_name == 'transformer_encoder':
        model = transformer_am.Transformer_encoder(num_heads=config.num_heads,
                                                   num_blocks=config.num_blocks,
                                                   d_model=config.d_model,
                                                   d_ff=config.d_ff,
                                                   pinyin_size=data_obj.num_pinyin,
                                                   dropout_rate=config.dropout_rate)
    elif model_name == 'DNN':
        model = DNN.DNN_model(data_obj.num_pinyin)
    elif model_name == 'cnn_ctc_test':
        model = cnn_ctc_test.CNN_CTC(data_obj.num_pinyin, config.feat_dim, config.attention_size)
    return model

def create_opt(opt_name, learning_rate=None):
    if opt_name == "adam":
        if learning_rate is None:
            optimizer = tf.keras.optimizers.Adam()
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate)
    return optimizer

if __name__ == '__main__':
    data_obj = data_helper_am.DataSpeech(data_root=config.data_root,
                                         mode=config.mode,
                                         specified_datasets=config.specified_datasets,
                                         batch_size=config.batch_size,
                                         audio_feat=config.audio_feat,
                                         shuffle=config.shuffle,
                                         winfunc=config.winfunc)
    model = create_model(config.model)
    optimizer = create_opt(config.opt_name, config.learning_rate)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

    latest_checkpoint = tf.train.latest_checkpoint(os.path.join(config.ckpt_path, config.model))
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint)

    if config.mode == 'val' or 'test':
        train_step()
    elif config.mode == 'val' or 'test':
        val_step()

