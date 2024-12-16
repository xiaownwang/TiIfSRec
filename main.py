import numpy as np
import config
import tensorflow as tf
from util import *
from data.data_load import DataLoader
from data.data_preprocess import DataPreprocessor
from model.short_term import *
from model.long_term import *
from tensorflow.keras.optimizers import Adam


def TiIfSRec():
    # load parameter from config
    folder = config.folder
    file_name = config.file_name
    max_users = config.max_users
    min_rows_per_user = config.min_rows_per_user
    max_rows = config.max_rows
    sl_split = config.sl_split
    short_max_len = config.short_max_len
    long_max_len = min_rows_per_user
    num_classes = config.num_classes
    embed_dim = config.embed_dim
    gru_units = embed_dim
    batch_size = config.batch_size
    epochs = config.epochs
    top_k = config.top_k
    learning_rate = config.learning_rate

    # loader = DataLoader(folder, file_name, max_users, min_rows_per_user)
    # loaded_data_path = loader.save_loaded_data()
    # loaded_data_path = './dataset/Cell_Phones_and_Accessories.csv'
    # loaded_data_path = './dataset/Movies_and_TV.csv'
    loaded_data_path = './dataset/Clothing_Shoes_and_Jewelry.csv'
    preprocessor = DataPreprocessor(loaded_data_path, max_rows, sl_split, num_classes)
    preprocessed_data = preprocessor.save_preprocessed_data()
    train_short, train_long, test_short, test_long = preprocessor.split_train_test()

    # Short
    X_train_item_S, X_train_time_S, X_train_freq_S, y_train_S = preprocessor.data_for_model(train_short, short_max_len)
    X_test_item_S, X_test_time_S, X_test_freq_S, y_test_S = preprocessor.data_for_model(test_short, short_max_len)

    X_train_S = [X_train_item_S, X_train_time_S, X_train_freq_S]
    X_test_S = [X_test_item_S, X_test_time_S, X_test_freq_S]

    y_train_S = tf.keras.utils.to_categorical(y_train_S, num_classes=num_classes)
    y_test_S = tf.keras.utils.to_categorical(y_test_S, num_classes=num_classes)

    num_items_S = X_train_item_S.shape[0] * X_train_item_S.shape[1]

    # Long
    X_train_item_L, X_train_time_L, X_train_freq_L, y_train_L = preprocessor.data_for_model(train_long, long_max_len)
    X_test_item_L, X_test_time_L, X_test_freq_L, y_test_L = preprocessor.data_for_model(test_long, long_max_len)

    X_train_L = [X_train_item_L, X_train_time_L, X_train_freq_L]
    X_test_L = [X_test_item_L, X_test_time_L, X_test_freq_L]
    y_train_L = tf.keras.utils.to_categorical(y_train_L, num_classes=num_classes)
    y_test_L = tf.keras.utils.to_categorical(y_test_L, num_classes=num_classes)

    num_items_L = X_train_item_L.shape[0] * X_train_item_L.shape[1]

    model = ShortRec(num_items_S, embed_dim, num_classes)
    model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_S, y_train_S, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    # Evaluation
    y_pred = model.predict(X_test_S)
    precision_10 = precision_at_k(y_test_S, y_pred, k=top_k)
    recall_10 = recall_at_k(y_test_S, y_pred, k=top_k)
    map_10 = map_at_k(y_test_S, y_pred, k=top_k)
    ndcg_10 = ndcg_at_k(y_test_S, y_pred, k=top_k)
    print(f"Precision@10: {precision_10}, Recall@10: {recall_10}, NDCG@10: {ndcg_10}, MAP@10: {map_10}")


    model = LongRec(num_items_L, embed_dim, num_classes, gru_units)
    model.compile(optimizer=Adam(learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(X_train_L, y_train_L, batch_size, epochs)
    # Evaluation
    y_pred = model.predict(X_test_L)
    precision_10 = precision_at_k(y_test_L, y_pred, k=top_k)
    recall_10 = recall_at_k(y_test_L, y_pred, k=top_k)
    map_10 = map_at_k(y_test_L, y_pred, k=top_k)
    ndcg_10 = ndcg_at_k(y_test_L, y_pred, k=top_k)
    print(f"Precision@10: {precision_10}, Recall@10: {recall_10}, NDCG@10: {ndcg_10}, MAP@10: {map_10}")

    short_model = ShortRec(num_items_S, embed_dim, num_classes)
    long_model = LongRec(num_items_L, embed_dim, num_classes, gru_units)

    fusion_model = GateFusion(short_model, long_model, embed_dim, num_classes)
    fusion_model.compile(optimizer=Adam(learning_rate), loss="categorical_crossentropy", metrics=["accuracy"])

    fusion_model.fit([X_train_S, X_train_L], y_train_S, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    # Evaluation
    y_pred = fusion_model.predict([X_test_S, X_test_L])
    precision_10 = precision_at_k(y_test_L, y_pred, k=top_k)
    recall_10 = recall_at_k(y_test_L, y_pred, k=top_k)
    map_10 = map_at_k(y_test_L, y_pred, k=top_k)
    ndcg_10 = ndcg_at_k(y_test_L, y_pred, k=top_k)
    print(f"Precision@10: {precision_10}, Recall@10: {recall_10}, NDCG@10: {ndcg_10}, MAP@10: {map_10}")

if __name__ == "__main__":
    TiIfSRec()