import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout, LayerNormalization
from tensorflow.keras.optimizers import Adam
from data.data_preprocess import DataPreprocessor
from util import *


# GRU4Rec 模型
class GRU4Rec(tf.keras.Model):
    def __init__(self, max_seq_len, num_items, embed_dim, num_classes):
        super(GRU4Rec, self).__init__()

        # 嵌入层：物品嵌入和位置嵌入
        self.item_embedding = Embedding(input_dim=num_items, output_dim=embed_dim, mask_zero=True)
        self.gru = GRU(embed_dim, return_sequences=True, return_state=True, name="GRU")

        # 全连接层
        self.dropout = Dropout(0.5)
        self.layer_norm = LayerNormalization(epsilon=1e-6)
        self.output_layer = Dense(num_classes, activation='softmax', name="OutputLayer")

    def call(self, inputs):
        # 输入序列
        item_seq = inputs

        # 嵌入物品序列
        item_emb = self.item_embedding(item_seq)  # (batch_size, seq_len, embed_dim)

        # 经过 GRU 模块
        gru_output, last_hidden_state = self.gru(item_emb)  # (batch_size, seq_len, embed_dim), (batch_size, embed_dim)

        # Dropout 和归一化
        gru_output = self.dropout(gru_output)
        gru_output = self.layer_norm(gru_output)

        # 只取最后时间步的隐藏状态作为最终输出
        output = self.output_layer(last_hidden_state)  # (batch_size, num_classes)

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
learning_rate = 0.00005

# 获取训练和测试数据
train_item_seqs, train_time_seqs, train_freq_seqs, train_target_seqs = preprocessor.data_for_model(train_short, 10)
test_item_seqs, test_time_seqs, test_freq_seqs, test_target_seqs = preprocessor.data_for_model(test_short, 10)

# 将目标标签转换为独热编码格式
train_target_seqs = tf.keras.utils.to_categorical(train_target_seqs, num_classes=num_classes)
test_target_seqs = tf.keras.utils.to_categorical(test_target_seqs, num_classes=num_classes)

# 创建 GRU4Rec 模型
model = GRU4Rec(max_seq_len=max_seq_len, embed_dim=64, num_classes=num_classes, num_items=10000)

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
