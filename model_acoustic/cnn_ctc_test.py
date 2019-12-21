import tensorflow as tf
from tensorflow.keras import layers

class CNN_CTC(tf.keras.Model):
    def __init__(self, pinyin_size, feat_dim, attention_size):
        super(CNN_CTC, self).__init__()
        self.pinyin_size = pinyin_size
        self.height_filter = attention_size
        self.width_filter = feat_dim
        # self.conv_layer1 = layers.Conv2D(filters=128, kernel_size=(attention_size, num_feat), padding='same', activation='relu')
        self.conv_layer2 = layers.Conv2D(filters=pinyin_size, kernel_size=(self.height_filter, self.width_filter),strides=(1, 1), padding='valid', activation='relu')

    def call(self, input):
        # x = self.conv_layer1(input)
        batch_size, num_frames = input.shape[0], input.shape[1]
        output = self.conv_layer2(input)
        output = tf.squeeze(output)
        num_frames_after_conv = output.shape[1]
        # padded = tf.zeros([num_frames - num_frames_after_conv,batch_size, self.pinyin_size])
        # logits = tf.reshape(output, [output.shape[1], output.shape[0], -1])
        # logits = tf.concat([logits, padded], 0)
        return output

