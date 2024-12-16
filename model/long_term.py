import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense
from tensorflow.keras.models import Model

class LongRec(Model):
    def __init__(self, num_items, embedding_dim, num_classes, gru_units):
        super(LongRec, self).__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.gru_units = gru_units

        # Embedding and dense layers
        self.embedding_item = Embedding(input_dim=num_items, output_dim=embedding_dim)
        self.embedding_time = Embedding(input_dim=num_items, output_dim=embedding_dim)
        self.embedding_freq = Embedding(input_dim=num_items, output_dim=embedding_dim)

        # self.embedding_time = Dense(embedding_dim)
        # self.embedding_freq = Dense(embedding_dim)

        # GRU weights
        self.W_xr = self.add_weight(shape=(embedding_dim, gru_units), initializer="glorot_uniform", name="W_xr")
        self.W_hr = self.add_weight(shape=(gru_units, gru_units), initializer="glorot_uniform", name="W_hr")
        self.W_tr = self.add_weight(shape=(embedding_dim, gru_units), initializer="glorot_uniform", name="W_tr")
        self.b_r = self.add_weight(shape=(gru_units,), initializer="zeros", name="b_r")

        self.W_xz = self.add_weight(shape=(embedding_dim, gru_units), initializer="glorot_uniform", name="W_xz")
        self.W_hz = self.add_weight(shape=(gru_units, gru_units), initializer="glorot_uniform", name="W_hz")
        self.W_tz = self.add_weight(shape=(embedding_dim, gru_units), initializer="glorot_uniform", name="W_tz")
        self.W_fz = self.add_weight(shape=(embedding_dim, gru_units), initializer="glorot_uniform", name="W_fz")
        self.b_z = self.add_weight(shape=(gru_units,), initializer="zeros", name="b_z")

        self.W_xh = self.add_weight(shape=(embedding_dim, gru_units), initializer="glorot_uniform", name="W_xh")
        self.W_hh = self.add_weight(shape=(gru_units, gru_units), initializer="glorot_uniform", name="W_hh")
        self.W_th = self.add_weight(shape=(embedding_dim, gru_units), initializer="glorot_uniform", name="W_th")
        self.W_fh = self.add_weight(shape=(embedding_dim, gru_units), initializer="glorot_uniform", name="W_fh")
        self.b_h = self.add_weight(shape=(gru_units,), initializer="zeros", name="b_h")

        self.W_a = self.add_weight(shape=(gru_units * 2, 1), initializer="glorot_uniform", name="W_a")

        # Output layer
        self.output_layer = Dense(num_classes, activation="softmax")

    def call(self, inputs):
        item, time, freq = inputs

        # Embedding layers
        item_emb = self.embedding_item(item)
        time_emb = self.embedding_time(time)
        freq_emb = self.embedding_freq(freq)

        # GRU computation
        batch_size = tf.shape(item_emb)[0]
        seq_len = tf.shape(item_emb)[1]
        h_prev = tf.zeros((batch_size, self.gru_units))

        # TensorArray save results
        outputs = tf.TensorArray(dtype=tf.float32, size=seq_len)

        for t in tf.range(seq_len):
            item_seq = item_emb[:, t, :]
            time_seq = time_emb[:, t, :]
            freq_seq = freq_emb[:, t, :]

            # Reset gate
            reset_gate = tf.nn.sigmoid(
                tf.matmul(item_seq, self.W_xr) +
                tf.matmul(h_prev, self.W_hr) +
                tf.matmul(time_seq, self.W_tr) +
                self.b_r
            )

            # Update gate
            update_gate = tf.nn.sigmoid(
                tf.matmul(item_seq, self.W_xz) +
                tf.matmul(h_prev, self.W_hz) +
                tf.matmul(time_seq, self.W_tz) +
                tf.matmul(freq_seq, self.W_fz) +
                self.b_z
            )

            # Candidate hidden state
            h_candidate = tf.nn.tanh(
                tf.matmul(item_seq, self.W_xh) +
                tf.matmul(reset_gate * h_prev, self.W_hh) +
                tf.matmul(time_seq, self.W_th) +
                tf.matmul(freq_seq, self.W_fh) +
                self.b_h
            )

            # Attention mechanism
            concat_hx = tf.concat([h_prev, item_seq], axis=-1)
            alpha_m = tf.nn.softmax(tf.matmul(concat_hx, self.W_a), axis=1)
            h_att = alpha_m * h_prev

            # Final hidden state
            h_prev = update_gate * h_att + (1 - update_gate) * h_candidate

            # save result in each time step
            outputs = outputs.write(t, h_prev)

        # last time step output
        final_output = outputs.stack()[-1]
        outputs = self.output_layer(final_output)  # (batch_size, num_classes)

        return outputs

