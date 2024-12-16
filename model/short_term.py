import tensorflow as tf
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, LayerNormalization

class ShortRec(tf.keras.Model):
    def __init__(self, num_items, embed_dim, num_classes):
        super(ShortRec, self).__init__()

        self.item_embedding = Embedding(input_dim=num_items, output_dim=embed_dim, mask_zero=True,
                                        embeddings_regularizer=tf.keras.regularizers.l2(0.01))
        self.time_gru = GRU(embed_dim, return_sequences=True, name="TimeGRU")
        self.freq_embedding = Embedding(input_dim=num_items, output_dim=embed_dim, mask_zero=False)

        self.WQ = Dense(embed_dim, name="QueryWeight")
        self.WK = Dense(embed_dim, name="KeyWeight")
        self.WV = Dense(embed_dim, name="ValueWeight")

        self.ffn = Dense(embed_dim, activation="relu")
        self.dropout = Dropout(0.5)
        self.layer_norm = LayerNormalization(epsilon=1e-6)

        self.output_layer = Dense(num_classes, activation='softmax', name="OutputLayer")

    def call(self, inputs):
        item_seq, time_seq, freq_seq = inputs

        item_emb = self.item_embedding(item_seq)
        time_emb = self.time_gru(tf.expand_dims(time_seq, axis=-1))
        freq_emb = self.freq_embedding(freq_seq)


        Q = self.WQ(item_emb)
        K = self.WK(item_emb) + time_emb + freq_emb
        V = self.WV(item_emb) + time_emb + freq_emb

        attention_logits = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(Q.shape[-1], tf.float32))
        attention_weights = tf.nn.softmax(attention_logits, axis=-1)
        attention_output = tf.matmul(attention_weights, V)

        ffn_output = self.ffn(attention_output)
        ffn_output = self.dropout(ffn_output)
        output = self.layer_norm(attention_output + ffn_output)

        # global avg pooling
        pooled_output = tf.reduce_mean(output, axis=1)  # (batch_size, embed_dim)
        output = self.output_layer(pooled_output)  # (batch_size, num_classes)

        # pooled_output = tf.reduce_max(output, axis=1)  # (batch_size, embed_dim)
        # output = self.output_layer(pooled_output)  # (batch_size, num_classes)

        # output = self.output_layer(output[:, -1, :])  # 只取最后一个时间步的输出


        return output