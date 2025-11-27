import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense
from tensorflow.keras.models import Model

class LongRec(Model):
    def __init__(self, num_items, embedding_dim, num_classes, gru_units):
        super(LongRec, self).__init__()
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.gru_units = gru_units

        # Embedding layers
        self.embedding_item = Embedding(input_dim=num_items, output_dim=embedding_dim)
        self.embedding_time = Embedding(input_dim=num_items, output_dim=embedding_dim)
        self.embedding_freq = Embedding(input_dim=num_items, output_dim=embedding_dim)

        # GRU weights
        self.W_xr = self.add_weight(shape=(embedding_dim, gru_units),
                                    initializer="glorot_uniform", name="W_xr")
        self.W_hr = self.add_weight(shape=(gru_units, gru_units),
                                    initializer="glorot_uniform", name="W_hr")
        self.b_r = self.add_weight(shape=(gru_units,),
                                   initializer="zeros", name="b_r")

        self.W_xz = self.add_weight(shape=(embedding_dim, gru_units),
                                    initializer="glorot_uniform", name="W_xz")
        self.W_hz = self.add_weight(shape=(gru_units, gru_units),
                                    initializer="glorot_uniform", name="W_hz")
        self.b_z = self.add_weight(shape=(gru_units,),
                                   initializer="zeros", name="b_z")

        self.W_xh = self.add_weight(shape=(embedding_dim, gru_units),
                                    initializer="glorot_uniform", name="W_xh")
        self.W_hh = self.add_weight(shape=(gru_units, gru_units),
                                    initializer="glorot_uniform", name="W_hh")
        self.b_h = self.add_weight(shape=(gru_units,),
                                   initializer="zeros", name="b_h")

        # Time gate and frequency gate 
        self.W_xtg = self.add_weight(shape=(embedding_dim, gru_units),
                                     initializer="glorot_uniform", name="W_xtg")
        self.W_tg = self.add_weight(shape=(embedding_dim, gru_units),
                                    initializer="glorot_uniform", name="W_tg")
        self.b_tg = self.add_weight(shape=(gru_units,),
                                    initializer="zeros", name="b_tg")

        self.W_xfg = self.add_weight(shape=(embedding_dim, gru_units),
                                     initializer="glorot_uniform", name="W_xfg")
        self.W_fg = self.add_weight(shape=(embedding_dim, gru_units),
                                    initializer="glorot_uniform", name="W_fg")
        self.b_fg = self.add_weight(shape=(gru_units,),
                                    initializer="zeros", name="b_fg")

        # NEW: time+freq -> adaptive decay rate δ_t
        self.W_delta = self.add_weight(shape=(gru_units * 2, gru_units),
                                       initializer="glorot_uniform", name="W_delta")
        self.b_delta = self.add_weight(shape=(gru_units,),
                                       initializer="zeros", name="b_delta")

        # NEW: frequency-direction branch & direction gate
        self.W_f_dir = self.add_weight(shape=(gru_units, gru_units),
                                       initializer="glorot_uniform", name="W_f_dir")
        self.b_f_dir = self.add_weight(shape=(gru_units,),
                                       initializer="zeros", name="b_f_dir")

        self.W_psi = self.add_weight(shape=(gru_units, gru_units),
                                     initializer="glorot_uniform", name="W_psi")
        self.b_psi = self.add_weight(shape=(gru_units,),
                                     initializer="zeros", name="b_psi")

        # Attention weight 
        self.W_a = self.add_weight(shape=(gru_units * 2, 1),
                                   initializer="glorot_uniform", name="W_a")

        # Output layer 
        self.output_layer = Dense(num_classes, activation="softmax")

    def get_representation(self, inputs, training=None):
        item, time, freq = inputs

        # Embedding layers
        item_emb = self.embedding_item(item)  # [B, L, D]
        time_emb = self.embedding_time(time)  # [B, L, D]
        freq_emb = self.embedding_freq(freq)  # [B, L, D]

        batch_size = tf.shape(item_emb)[0]
        seq_len = tf.shape(item_emb)[1]
        h_prev = tf.zeros((batch_size, self.gru_units))

        for t in tf.range(seq_len):
            item_seq = item_emb[:, t, :]   # [B, D]
            time_seq = time_emb[:, t, :]   # [B, D]
            freq_seq = freq_emb[:, t, :]   # [B, D]

            # Time and frequency gates
            time_gate = tf.nn.sigmoid(
                tf.matmul(item_seq, self.W_xtg) +
                tf.matmul(time_seq, self.W_tg) +
                self.b_tg
            )  # [B, H]

            freq_gate = tf.nn.sigmoid(
                tf.matmul(item_seq, self.W_xfg) +
                tf.matmul(freq_seq, self.W_fg) +
                self.b_fg
            )  # [B, H]

            # adaptive decay rate
            decay_in = tf.concat([time_gate, freq_gate], axis=-1)  # [B, 2H]
            delta_t = tf.nn.softplus(tf.matmul(decay_in, self.W_delta) + self.b_delta)  # [B, H]
            h_prev_decay = tf.exp(-delta_t) * h_prev  # [B, H]

            # Reset gate、
            reset_gate = tf.nn.sigmoid(
                tf.matmul(item_seq, self.W_xr) +
                tf.matmul(h_prev_decay, self.W_hr) +
                self.b_r
            )

            # Update gate
            update_gate = tf.nn.sigmoid(
                tf.matmul(item_seq, self.W_xz) +
                tf.matmul(h_prev_decay, self.W_hz) +
                self.b_z
            )

            # Base candidate
            h_base = tf.nn.tanh(
                tf.matmul(item_seq, self.W_xh) +
                tf.matmul(reset_gate * h_prev_decay, self.W_hh) +
                self.b_h
            )  # [B, H]

            # Frequency-direction candidate
            h_freq = tf.nn.tanh(
                h_base +
                tf.matmul(freq_gate, self.W_f_dir) +
                self.b_f_dir
            )  # [B, H]

            # Direction gate ψ_t
            psi = tf.nn.sigmoid(
                tf.matmul(freq_gate, self.W_psi) + self.b_psi
            )  # [B, H]

            h_candidate = (1.0 - psi) * h_base + psi * h_freq  # [B, H]

            concat_hx = tf.concat([h_prev_decay, item_seq], axis=-1)  # [B, H+D]
            alpha_m = tf.nn.softmax(tf.matmul(concat_hx, self.W_a), axis=1)  # [B, 1]
            h_att = alpha_m * h_prev_decay  # [B, H]

            # Final hidden state update
            h_prev = (1.0 - update_gate) * h_att + update_gate * h_candidate  # [B, H]

        e_long = h_prev  # [B, H]
        return e_long

    def call(self, inputs, training=None, return_representation: bool = False):
        e_long = self.get_representation(inputs, training=training)

        if return_representation:
            return e_long

        logits = self.output_layer(e_long)  # [B, num_classes]
        return logits
