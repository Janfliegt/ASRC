import tensorflow as tf
import numpy as np

def get_pinyin_embeddings(pinyin_size, embedding_dim, zero_pad=True):
    '''Constructs token embedding matrix.
    Note that the column of index 0's are set to zeros.
    vocab_size: scalar. V.
    num_units: embedding dimensionalty. E.
    zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
    To apply query/key masks easily, zero pad is turned on.

    Returns
    weight variable: (V, E)
    '''

    embeddings = tf.Variable(initial_value=tf.random.uniform(shape=(pinyin_size, embedding_dim)),
                             dtype=tf.float32)
    if zero_pad:
        embeddings = tf.concat((tf.zeros(shape=[1, embedding_dim]),
                                embeddings[1:, :]), 0)
    return embeddings


def positional_encoding(inputs, maxlen):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.

    returns
    3d tensor that has the same shape as inputs.
    '''

    E = inputs.shape.as_list()[-1]  # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
    # position indices
    position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)

    # First part of the PE function: sin and cos argument
    position_enc = np.array([
        [pos / np.power(10000, (i - i % 2)/E) for i in range(E)]
        for pos in range(maxlen)])

    # Second part, apply the cosine to even columns and sin to odds.
    position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
    position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
    position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

    outputs = tf.nn.embedding_lookup(position_enc, position_ind)

    return tf.dtypes.cast(outputs, tf.float32)

class multihead_attention(tf.keras.Model):
    def __init__(self, num_heads, d_ff, d_model, dropout_rate):
        super().__init__()
        self.num_heads = num_heads
        self.Q_Dense = tf.keras.layers.Dense(d_model)
        self.K_Dense = tf.keras.layers.Dense(d_model)
        self.V_Dense = tf.keras.layers.Dense(d_model)
        self.dropout_layer = tf.keras.layers.Dropout(rate=dropout_rate)
        self.ln = tf.keras.layers.LayerNormalization()

    def call(self, input):
        Q = self.Q_Dense(input)
        K = self.K_Dense(input)
        V = self.V_Dense(input)

        # Split and concat - multihead attention
        Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, self.num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, self.num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        # Attention
        outputs = self.scaled_dot_product_attention(Q_, K_, V_, self.dropout_layer)
        # Restore shape
        outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0), axis=2)  # (N, T_q, d_model)
        # Residual connection
        outputs += input

        # Normalize
        outputs = self.ln(outputs)
        return outputs

    def scaled_dot_product_attention(self, Q, K, V, dropout_layer):
        d_k = Q.shape.as_list()[-1]  # d_k is  d_model

        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

        # scale
        outputs /= d_k ** 0.5

        # softmax
        outputs = tf.nn.softmax(outputs)

        outputs = dropout_layer(outputs)
        # weighted sum (context vectors)
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

        return outputs

class feedforward(tf.keras.Model):
    def __init__(self, d_ff, d_model):
        super().__init__()
        self.dense_layer_ff1 = tf.keras.layers.Dense(d_ff, activation=tf.nn.relu)
        self.dense_layer_ff2 = tf.keras.layers.Dense(d_model)
        self.ln = tf.keras.layers.LayerNormalization()
    def call(self, input):
        x = self.dense_layer_ff1(input)
        output = self.dense_layer_ff2(x)
        # Residual connection
        output += input
        # Normalize
        output = self.ln(output)
        return output


###########
class Transformer_encoder(tf.keras.Model):
    def __init__(self, num_heads, num_blocks, d_model, d_ff, pinyin_size, vocab_size, dropout_rate):
        super().__init__()
        self.pinyin_size = pinyin_size
        self.num_blocks = num_blocks
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.embeddings = get_pinyin_embeddings(self.pinyin_size, self.d_model, zero_pad=True)
        self.dropout_layer_in = tf.keras.layers.Dropout(rate=dropout_rate)
        self.attention_with_ff = []
        for _ in range(num_blocks):
            self_attention_layer = multihead_attention(num_heads=num_heads,
                                              d_ff=d_ff,
                                              d_model=d_model,
                                              dropout_rate=dropout_rate)
            feedforward_network = feedforward(d_ff=d_ff, d_model=d_model)
            self.attention_with_ff.append(
                [self_attention_layer,
                 feedforward_network
                ])
        self.Dense_layer_out = tf.keras.layers.Dense(vocab_size)

    def call(self, pinyin_tensor):
        """
        :param xs:
        :param training:
        :return:
        """
        batch_size, seq_len = pinyin_tensor.shape[0], pinyin_tensor.shape[1]
        # embedding
        enc = tf.nn.embedding_lookup(self.embeddings, pinyin_tensor)  # (N, T1, d_model)
        enc *= self.d_model**0.5  # scale

        enc += positional_encoding(enc, seq_len)
        enc = self.dropout_layer_in(enc)
        for layer in self.attention_with_ff:
            self_attention_layer = layer[0]
            feedforward_network = layer[1]
            enc = self_attention_layer(enc)
            # enc = feedforward_network(enc)

        # Final linear projection
        logits = self.Dense_layer_out(enc)

        return logits