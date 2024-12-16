import tensorflow as tf
from keras.layers import LayerNormalization
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
from data.data_preprocess import DataPreprocessor
from util import *


class SASRec(tf.keras.Model):
    def __init__(self, max_seq_len, num_items, embed_dim, num_classes):
        super(SASRec, self).__init__()

        self.item_embedding = Embedding(input_dim=num_items, output_dim=embed_dim)
        self.positional_encoding = Embedding(input_dim=max_seq_len, output_dim=embed_dim)

        self.WQ = Dense(embed_dim, name="QueryWeight")
        self.WK = Dense(embed_dim, name="KeyWeight")
        self.WV = Dense(embed_dim, name="ValueWeight")

        self.ffn = Dense(embed_dim, activation="relu")
        self.dropout = Dropout(0.5)
        self.layer_norm = LayerNormalization(epsilon=1e-6)

        self.output_layer = Dense(num_classes, activation='softmax', name="OutputLayer")

    def call(self, inputs):
        item_seq = inputs

        item_emb = self.item_embedding(item_seq)
        positions = tf.range(start=0, limit=max_seq_len, delta=1)
        position_embeddings = self.positional_encoding(positions)

        Q = self.WQ(item_emb)
        K = self.WK(item_emb) + position_embeddings
        V = self.WV(item_emb) + position_embeddings

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



# 数据预处理
# loaded_data_path = '../dataset/Cell_Phones_and_Accessories.csv'
# loaded_data_path = '../dataset/Movies_and_TV.csv'
loaded_data_path = '../dataset/Clothing_Shoes_and_Jewelry.csv'
preprocessor = DataPreprocessor(file_path=loaded_data_path)
preprocessed_data = preprocessor.save_preprocessed_data()
train_short, train_long, test_short, test_long = preprocessor.split_train_test()

# 设置参数
max_seq_len = 8  # 设置序列最大长度
num_classes = 100
top_k = 10
learning_rate = 0.000001

# 获取训练和测试数据
train_item_seqs, train_time_seqs, train_freq_seqs, train_target_seqs = preprocessor.data_for_model(train_short, 10)
test_item_seqs, test_time_seqs, test_freq_seqs, test_target_seqs = preprocessor.data_for_model(test_short, 10)

# 将目标标签转换为独热编码格式
train_target_seqs = tf.keras.utils.to_categorical(train_target_seqs, num_classes=num_classes)
test_target_seqs = tf.keras.utils.to_categorical(test_target_seqs, num_classes=num_classes)




# 创建 SASRec 模型
model = SASRec(max_seq_len=max_seq_len, embed_dim=64, num_classes=num_classes, num_items=10000)

# 编译模型
model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(
    train_item_seqs,
    train_target_seqs,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# 评估模型
y_pred = model.predict(test_item_seqs)

# 计算评估指标
precision_10 = precision_at_k(test_target_seqs, y_pred, k=top_k)
recall_10 = recall_at_k(test_target_seqs, y_pred, k=top_k)
map_10 = map_at_k(test_target_seqs, y_pred, k=top_k)
ndcg_10 = ndcg_at_k(test_target_seqs, y_pred, k=top_k)

print(f"Precision@10: {precision_10}, Recall@10: {recall_10}, NDCG@10: {ndcg_10}, MAP@10: {map_10}")
