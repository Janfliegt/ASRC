import tensorflow as tf

class DNN_model(tf.keras.Model):

    def __init__(self, pinyin_size):
        super().__init__()
        self.FF_layer1 = tf.keras.layers.Dense(pinyin_size / 2,  activation=tf.nn.relu)
        self.FF_layer_out = tf.keras.layers.Dense(pinyin_size,  activation=tf.nn.relu)

    def __call__(self, input):
        output = self.FF_layer1(input)
        output = self.FF_layer_out(output)
        return output

