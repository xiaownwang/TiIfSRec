import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from data.data_preprocess import DataPreprocessor
from util import *


class TiSASRecModel(tf.keras.Model):
    def __init__(self, max_seq_len, num_classes, embedding_dim=64, num_heads=4, num_layers=2, time_emb_dim=64):
        super(TiSASRecModel, self).__init__()

        # Item embedding
        self.embedding_layer = layers.Embedding(input_dim=10000, output_dim=embedding_dim)

        # Time interval embedding
        self.time_embedding_layer = layers.Embedding(input_dim=100, output_dim=time_emb_dim)

        # Multi-head self-attention layers
        self.attention_layers = [layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim) for _ in
                                 range(num_layers)]

        # Global attention layer
        self.global_attention_layer = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embedding_dim)

        # Feed-forward layers
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        item_input, time_interval_input = inputs

        # Embedding lookup
        item_emb = self.embedding_layer(item_input)
        time_emb = self.time_embedding_layer(time_interval_input)

        # Concatenate item embeddings with time interval embeddings
        x = item_emb + time_emb  # element-wise addition

        # Multi-order attention layers
        for att_layer in self.attention_layers:
            x = att_layer(query=x, value=x, key=x)  # 批量处理，不需要显式迭代

        # Global attention layer
        x = self.global_attention_layer(query=x, value=x, key=x)

        # Output layers
        x = self.flatten(x)
        x = self.dense1(x)
        output = self.dense2(x)

        return output



def build_tisasrec_model(max_seq_len, num_classes):
    model = TiSASRecModel(max_seq_len, num_classes)
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
learning_rate = 0.000005

# 获取训练和测试数据
train_item_seqs, train_time_seqs, train_freq_seqs, train_target_seqs = preprocessor.data_for_model(train_short, 10)
test_item_seqs, test_time_seqs, test_freq_seqs, test_target_seqs = preprocessor.data_for_model(test_short, 10)

train_target_seqs = tf.keras.utils.to_categorical(train_target_seqs, num_classes=num_classes)
test_target_seqs = tf.keras.utils.to_categorical(test_target_seqs, num_classes=num_classes)



# 构建并编译模型
model = build_tisasrec_model(max_seq_len=48, num_classes=100)

# 训练 TiSASRec 模型
history = model.fit(
    [train_item_seqs, train_time_seqs],  # 输入是 item 序列和时间间隔序列
    train_target_seqs,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)

# 评估模型
y_pred = model.predict([test_item_seqs, test_time_seqs])

# 计算评估指标
precision_10 = precision_at_k(test_target_seqs, y_pred, k=top_k)
recall_10 = recall_at_k(test_target_seqs, y_pred, k=top_k)
map_10 = map_at_k(test_target_seqs, y_pred, k=top_k)
ndcg_10 = ndcg_at_k(test_target_seqs, y_pred, k=top_k)

print(f"Precision@10: {precision_10}, Recall@10: {recall_10}, NDCG@10: {ndcg_10}, MAP@10: {map_10}")

