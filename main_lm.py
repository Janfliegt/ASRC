import sys
from model_language import transformer_lm
import os
import tensorflow as tf
from absl import logging
from data_helper import data_helper_lm
from os.path import abspath, join, dirname
sys.path.insert(0, join(abspath(dirname(__file__)), 'config'))
import importlib
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = importlib.import_module((sys.argv[1]))
# main(sys.argv)
data_obj = data_helper_lm.DataSpeech(data_root=config.data_root,
                                  mode=config.mode,
                                  batch_size=config.batch_size,
                                  shuffle=config.shuffle
                                  )
data_obj_val = data_helper_lm.DataSpeech(data_root=config.data_root,
                                  mode='val',
                                  batch_size=500,
                                  shuffle=config.shuffle
                                  )

def train_step():
    """
    train the model
    :param input_pinyin: A tensor with the shape [batch_size, pinyin_seq_len]
    :param label_language: A tensor with the shape [batch_size, charac_seq_len]
    :return: two tensors, the mena loss in a batch and the accuracy in a batch
    """
    transformer_encoder = transformer_lm.Transformer_encoder(num_heads=config.num_heads,
                                                              num_blocks=config.num_blocks,
                                                              d_model=config.d_model,
                                                              d_ff=config.d_ff,
                                                              pinyin_size=data_obj.num_pinyin,
                                                              vocab_size=data_obj.num_vocab,
                                                              dropout_rate=config.dropout_rate)
    opt = tf.keras.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(model=transformer_encoder, optimizer=opt)
    train_loss = tf.keras.metrics.Mean("loss")
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy("train acc")
    def train():
        with tf.GradientTape() as tape:
            # forward
            logits = transformer_encoder(input_pinyin, training=True)
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true=label_language,
                                                                                  y_pred=logits,
                                                                                  from_logits=True))
        # 计算梯度并更新参数
        variables = transformer_encoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        opt.apply_gradients(zip(gradients, variables))
        train_loss(loss)
        train_acc(label_language, logits)


    def eval():

        # forward
        logits = transformer_encoder.predict(input_pinyin)# label_language_onehot = tf.one_hot(label_language, data_obj.num_vocab)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_true=label_language,
                                                                              y_pred=logits,
                                                                              from_logits=True))
        # 计算loss 和准确率
        train_loss(loss)
        train_acc(label_language, logits)

    for epoch in range(config.epoch):
        current_step = 1
        for lm_batch in data_obj.get_lm_batch():
            input_pinyin = tf.convert_to_tensor(lm_batch['input_pinyin'], dtype=tf.int32)
            label_language = tf.convert_to_tensor(lm_batch['label_language'], dtype=tf.int32)
            train()
            print("epoch:{}, step:{}, batch_loss:{:.3f}, acc:{:.3f}".format(epoch,
                                                                            current_step,
                                                                            train_loss.result().numpy(),
                                                                            train_acc.result().numpy()))
            train_loss.reset_states()
            train_acc.reset_states()
            sys.stdout.flush()
            current_step += 1
        checkpoint_name = checkpoint.save(
            os.path.join(
                config.ckpt_path,
                "ctl_step_{}.ckpt".format(current_step)))
        logging.info("Saved checkpoint to %s", checkpoint_name)

    current_step = 1
    for lm_batch in data_obj_val.get_lm_batch():
        input_pinyin = tf.convert_to_tensor(lm_batch['input_pinyin'], dtype=tf.int32)
        label_language = tf.convert_to_tensor(lm_batch['label_language'], dtype=tf.int32)
        eval()
        print("epoch:{}, step:{}, batch_loss:{:.3f}, acc:{:.3f}".format(epoch,
                                                                      current_step,
                                                                      train_loss.result().numpy(),
                                                                      train_acc.result().numpy()))
        sys.stdout.flush()
        current_step += 1

def evaluate_step():
    """
    evaluate the trained model
    :param input_pinyin: A tensor with the shape [batch_size, pinyin_seq_len]
    :param label_language: A tensor with the shape [batch_size, charac_seq_len]
    :return: two tensors, the mena loss in a batch and the accuracy in a batch
    # """
    transformer_encoder = transformer_TF2.Transformer_encoder(num_heads=config.num_heads,
                                                              num_blocks=config.num_blocks,
                                                              d_model=config.d_model,
                                                              d_ff=config.d_ff,
                                                              pinyin_size=data_obj.num_pinyin,
                                                              vocab_size=data_obj.num_vocab,
                                                              dropout_rate=config.dropout_rate)
    opt = tf.keras.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(model=transformer_encoder, optimizer=opt)
    latest_checkpoint = tf.train.latest_checkpoint(config.ckpt_path)

    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint)

    def eval():
        with tf.GradientTape() as tape:
            # forward
            logits = transformer_encoder(input_pinyin)
            # label_language_onehot = tf.one_hot(label_language, data_obj.num_vocab)
            ce = tf.nn.softmax_cross_entropy_with_logits(labels=label_language_onehot, logits=logits)
            loss = tf.reduce_sum(ce)

        # 计算loss 和准确率
        batch_loss = (loss / int(input_pinyin.shape[0]))
        preds = tf.dtypes.cast(tf.argmax(logits, axis=-1), tf.int32)
        istarget = tf.dtypes.cast(tf.not_equal(label_language, 0), tf.float32)
        acc = tf.reduce_sum(tf.dtypes.cast(tf.equal(preds, label_language), tf.float32) * istarget) / (
            tf.reduce_sum(istarget))
        return batch_loss, acc

    current_step = 1
    for lm_batch in data_obj.get_lm_batch():
        input_pinyin = tf.convert_to_tensor(lm_batch['input_pinyin'], dtype=tf.int32)
        label_language = tf.convert_to_tensor(lm_batch['label_language'], dtype=tf.int32)
        label_language_onehot = tf.one_hot(label_language, data_obj.num_vocab)
        batch_loss, acc = eval()
        print("step:", current_step, "valadate batch_loss:", batch_loss.numpy(), "acc:", acc.numpy())
        current_step += 1

if __name__ == '__main__':
    if data_obj.mode == "train":
        train_step()
    else:
        evaluate_step()
