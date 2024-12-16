import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from data.data_preprocess import DataPreprocessor
from util import *


class MARankModel(tf.keras.Model):
    def __init__(self, max_seq_len, num_classes, embedding_dim=32, num_heads=4, num_layers=2):
        super(MARankModel, self).__init__()

        self.embedding_layer = layers.Embedding(input_dim=10000, output_dim=embedding_dim)
        self.attention_layers = [layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim) for _ in
                                 range(num_layers)]
        self.global_attention_layer = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)

        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        item_input = inputs  # 输入只有 item 序列

        # 嵌入层
        x = self.embedding_layer(item_input)

        # Multi-order attention 层
        for att_layer in self.attention_layers:
            x = att_layer(x, x)

        # 全局注意力
        x = self.global_attention_layer(x, x)

        # 分类层
        x = self.flatten(x)
        x = self.dense1(x)
        output = self.dense2(x)

        return output


def build_marank_model(max_seq_len, num_classes):
    model = MARankModel(max_seq_len, num_classes)
    model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model



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
learning_rate = 0.1

# 获取训练和测试数据
train_item_seqs, train_time_seqs, train_freq_seqs, train_target_seqs = preprocessor.data_for_model(train_short, 10)
test_item_seqs, test_time_seqs, test_freq_seqs, test_target_seqs = preprocessor.data_for_model(train_short, 10)

train_target_seqs = tf.keras.utils.to_categorical(train_target_seqs, num_classes=num_classes)
test_target_seqs = tf.keras.utils.to_categorical(test_target_seqs, num_classes=num_classes)

# 构建并编译模型
model = build_marank_model(max_seq_len=48, num_classes=100)

# 训练 MARank 模型
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
