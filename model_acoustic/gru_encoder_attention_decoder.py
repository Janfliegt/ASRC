import tensorflow as tf

class Encoder(tf.keras.Model):
    def __init__(self, enc_units, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units
        # self.embedding = tf.keras.layers.Embedding(1425, 256)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def __call__(self, enc_input, hidden_state):
        # x = self.embedding(x)
        enc_output, hidden_state = self.gru(enc_input, initial_state=hidden_state)
        return enc_output, hidden_state

    def initialize_hidden_state(self):
        initial_hidden = tf.zeros([self.batch_size, self.enc_units])
        return initial_hidden

# Attention
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def __call__(self, query, values):
        #
        hidden_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh
                       (self.W1(values) + self.W2(hidden_with_time_axis))
                       )
        attention_weights = tf.nn.softmax(score, axis=1)

        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# Decoder
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super().__init__()
        # self.batch_size = batch_size
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def __call__(self, input, hidden, enc_output):

        context_vector, _ = self.attention(hidden, enc_output)
        input = self.embedding(input)
        context_vector_expand = tf.expand_dims(context_vector, 1)

        input_concat = tf.concat([context_vector_expand, input], axis=-1)

        # passing the concatenated vector to the GRU
        dec_output, hidden_state = self.gru(input_concat)

        # output shape == (batch_size * 1, hidden_size)
        output_gru_reshpe = tf.reshape(dec_output, (-1, dec_output.shape[2]))

        output_fc = self.fc(output_gru_reshpe)

        return output_fc, hidden_state

def loss_function(targets, logits):
    """
    calculate the ctc loss
    :param targets:
    :param logits:
    :return: the ctc loss
    """
    return tf.nn.ctc_loss(targets, logits)



