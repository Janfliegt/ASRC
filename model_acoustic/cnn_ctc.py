import tensorflow as tf
import numpy as np

# =============================搭建模型====================================
class CNN_CTC(tf.keras.Model):
    """docstring for Amodel."""
    def __init__(self, pinyin_size, dropout_rate=0.2):
        super().__init__()
        self.pinyin_size = pinyin_size
        self.block_1 = cnn_cell(32)
        self.block_2 = cnn_cell(64)
        self.block_3 = cnn_cell(128)
        self.block_4 = cnn_cell(128, pool=False)
        self.block_5 = cnn_cell(128, pool=False)

        self.reshspe = tf.keras.layers.Reshape((-1, 3200))
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dense_output1 = tf.keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal")
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dense_output2 = tf.keras.layers.Dense(pinyin_size, activation="softmax")
    #
    def call(self, input):
        # batch_size, num_frames = input.shape[0], input.shape[1]
        input = tf.expand_dims(input, -1)
        x = self.block_1(input)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.reshspe(x)
        x = self.dropout1(x)
        x = self.dense_output1(x)
        x = self.dropout2(x)
        x = self.dense_output2(x)
        # final_output = tf.reshape(x, [x.shape[1], x.shape[0], -1])
        return x

class cnn_cell(tf.keras.Model):
  def __init__(self, num_outputs, pool=True):
    super(cnn_cell, self).__init__()
    self.num_outputs = num_outputs
    self.conv_layer1 = tf.keras.layers.Conv2D(num_outputs,
                                             kernel_size=(3, 3),
                                             activation="relu",
                                             padding="same",
                                             kernel_initializer="he_normal")
    self.batch_norm1 = tf.keras.layers.BatchNormalization(axis=-1)
    self.conv_layer2 = tf.keras.layers.Conv2D(num_outputs,
                                             kernel_size=(3, 3),
                                             activation="relu",
                                             padding="same",
                                             kernel_initializer="he_normal")
    self.batch_norm2 = tf.keras.layers.BatchNormalization(axis=-1)
    self.pool = pool
    if pool:
        self.pool = tf.keras.layers.MaxPool2D()

  def call(self, input):
    x = self.conv_layer1(input)
    x = self.batch_norm1(x)
    x = self.conv_layer2(x)
    output = self.batch_norm2(x)
    if self.pool:
        output = self.pool(output)
    return output
