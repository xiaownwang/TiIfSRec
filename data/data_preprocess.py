import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import tensorflow as tf


class DataPreprocessor:
    def __init__(self, file_path, max_rows=None, sl_split=0.2, num_classes=50):
        self.file_path = file_path
        self.max_rows = max_rows
        self.data = None
        self.preprocessed_data = None
        self.sl_split = sl_split
        self.num_classes = num_classes

    def load_data(self):
        if self.file_path.endswith(".csv"):
            self.data = pd.read_csv(self.file_path)
        elif self.file_path.endswith(".jsonl"):
            self.data = pd.read_json(self.file_path, lines=True)

        if self.max_rows:
            self.data = self.data.head(self.max_rows)

        print(f"Loaded {len(self.data)} rows from {self.file_path}")

    def generate_product_ids(self):
        # Convert titles to TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english')
        title_vectors = vectorizer.fit_transform(self.data['title'])

        # KMeans cluster
        kmeans = KMeans(n_clusters=self.num_classes, random_state=42)
        self.data['product_id'] = kmeans.fit_predict(title_vectors)

    def preprocess_data(self):
        self.data = self.data.dropna(subset=['parent_asin', 'user_id', 'timestamp', 'title'])
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'], errors='coerce')
        self.data = self.data.dropna(subset=['timestamp'])
        self.data = self.data.sort_values(by=['user_id', 'timestamp'])
        self.generate_product_ids()

        # timestamp (seconds)
        # self.data['time_seq'] = self.data.groupby('user_id')['timestamp'].diff().dt.total_seconds().fillna(0)
        self.data['time_seq'] = pd.to_datetime(self.data['timestamp']).astype(int) / 10 ** 9
        self.data['time_seq'] = self.data.groupby('user_id')['time_seq'].diff().fillna(0).astype(float)
        mean_time_seq = self.data['time_seq'][self.data['time_seq'] > 0].mean()
        self.data['time_seq'] = self.data['time_seq'].apply(lambda x: mean_time_seq if x == 0 else x)

        # user history sequence
        self.data['product_sequence'] = self.data.groupby('user_id')['parent_asin'].transform(lambda x: x.tolist())

        # itme frequency
        item_freq = self.data['product_id'].value_counts().to_dict()
        item_freq_inv = {key: 1 / value for key, value in item_freq.items()}
        self.data['item_freq'] = self.data['product_id'].map(item_freq_inv)

        # position
        self.data['position'] = self.data.groupby('user_id').cumcount()

        # scaler
        scaler = MinMaxScaler()
        self.data['time_seq'] = scaler.fit_transform(self.data[['time_seq']])
        self.data['item_freq'] = scaler.fit_transform(self.data[['item_freq']])

        self.preprocessed_data = self.data[
            ['user_id', 'product_sequence', 'product_id', 'time_seq', 'item_freq', 'position']].drop_duplicates()

    def save_preprocessed_data(self):
        self.load_data()
        self.preprocess_data()

        output_file = os.path.splitext(self.file_path)[0] + "_Preprocessed.csv"
        self.preprocessed_data.to_csv(output_file, index=False)
        print(f"Preprocessed data saved to {output_file}")

        return self.preprocessed_data


    def data_for_model(self, data, max_seq_len):
        user_data = data.groupby('user_id')
        item_seqs, time_seqs, freq_seqs, target_seqs = [], [], [], []
        max_seq_index = max_seq_len - 1

        for user_id, group in user_data:
            item_seq = group['product_id'].values
            time_seq = group['time_seq'].values
            freq_seq = group['item_freq'].values

            padded_item_seq = [0] * (max_seq_len - len(item_seq)) + list(item_seq)
            padded_time_seq = [0] * (max_seq_len - len(time_seq)) + list(time_seq)
            padded_freq_seq = [0] * (max_seq_len - len(freq_seq)) + list(freq_seq)

            item_seqs.append(padded_item_seq[:max_seq_index - 1])
            time_seqs.append(padded_time_seq[:max_seq_index - 1])
            freq_seqs.append(padded_freq_seq[:max_seq_index - 1])
            target_seqs.append(padded_item_seq[max_seq_index])

        return (
            tf.constant(item_seqs, dtype=tf.int32),
            tf.constant(time_seqs, dtype=tf.float32),
            tf.constant(freq_seqs, dtype=tf.float32),
            tf.constant(target_seqs, dtype=tf.int32)
        )

