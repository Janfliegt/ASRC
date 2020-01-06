import tensorflow as tf

# =============================搭建模型====================================
class GRU_CTC(tf.keras.Model):
    """docstring for Amodel."""
    def __init__(self, pinyin_size, dropout_rate=0.2):
        super().__init__()
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dense_output1 = tf.keras.layers.Dense(512, activation="relu", use_bias=True, kernel_initializer="he_normal")
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)
        self.dense_output2 = tf.keras.layers.Dense(512, activation="relu", use_bias=True, kernel_initializer="he_normal")
        self.block_1 = bi_gru(512)
        self.block_2 = bi_gru(512)
        self.block_3 = bi_gru(512)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)
        self.dense_output3 = tf.keras.layers.Dense(512, activation="relu", use_bias=True, kernel_initializer="he_normal")
        self.dropout4 = tf.keras.layers.Dropout(dropout_rate)
        self.dense_output4 = tf.keras.layers.Dense(pinyin_size, activation="softmax", use_bias=True, kernel_initializer="he_normal")
    #
    def call(self, input):
        # batch_size, num_frames = input.shape[0], input.shape[1]
        x = self.dropout1(input)
        x = self.dense_output1(x)
        x = self.dropout2(x)
        x = self.dense_output2(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.dropout3(x)
        x = self.dense_output3(x)
        x = self.dropout4(x)
        x = self.dense_output4(x)
        # final_output = tf.reshape(x, [x.shape[1], x.shape[0], -1])
        return x

class bi_gru(tf.keras.layers.Layer):
  def __init__(self, units, dropout_rate=0.2):
    super(bi_gru, self).__init__()
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.gru_layer_forward = tf.keras.layers.GRU(units, return_sequences=True, kernel_initializer='he_normal')
    self.gru_layer_backward = tf.keras.layers.GRU(units, return_sequences=True, go_backwards=True, kernel_initializer='he_normal')
    self.added = tf.keras.layers.Add()

  def call(self, input):
    x = self.dropout(input)
    f = self.gru_layer_forward(x)
    b = self.gru_layer_backward(x)
    output = self.added([f, b])
    return output
