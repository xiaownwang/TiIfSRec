import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import Model


class GateFusion(tf.keras.Model):
    def __init__(self, short_model, long_model, embed_dim, num_classes):
        super(GateFusion, self).__init__()
        self.short_model = short_model
        self.long_model = long_model

        # Gate fusion layer
        self.W_sg = Dense(num_classes)  # short-term weight
        self.W_lg = Dense(num_classes)  # long-term weight
        self.b_g = Dense(num_classes, activation='sigmoid')  # bias

        self.output_layer = Dense(num_classes, activation='softmax')  # output layer

    def call(self, inputs):
        inputs_short, inputs_long = inputs

        e_short = self.short_model(inputs_short)
        e_long = self.long_model(inputs_long)

        g = tf.nn.tanh(self.W_sg(e_short) + self.W_lg(e_long) + self.b_g(e_long))
        e_fusion = g * e_short + (1 - g) * e_long

        outputs = self.output_layer(e_fusion)

        return outputs




# Function to calculate Precision@k
def precision_at_k(y_true, y_pred, k=10):
    # y_true is a binary vector with ground truth (relevant items)
    # y_pred is a predicted score for each item (after model prediction)
    top_k_preds = np.argsort(y_pred, axis=1)[:, -k:]  # Get top k predictions for each user
    relevant = np.take_along_axis(y_true, top_k_preds, axis=1)
    precision = np.mean(np.sum(relevant, axis=1) / k)  # Precision@k for each user
    return precision

# Function to calculate Recall@k
def recall_at_k(y_true, y_pred, k=10):
    top_k_preds = np.argsort(y_pred, axis=1)[:, -k:]
    relevant = np.take_along_axis(y_true, top_k_preds, axis=1)
    recall = np.mean(np.sum(relevant, axis=1) / np.sum(y_true, axis=1))  # Recall@k
    return recall

# Function to calculate MAP@k
def map_at_k(y_true, y_pred, k=10):
    map_scores = []
    for i in range(y_true.shape[0]):
        avg_precision = 0
        top_k_preds = np.argsort(y_pred[i, :])[-k:]  # Get top k predictions for user i
        relevant_items = y_true[i, top_k_preds]

        if np.sum(relevant_items) == 0:  # If there are no relevant items, skip this user
            continue

        for j, rel in enumerate(relevant_items):
            if rel == 1:
                avg_precision += np.sum(relevant_items[:j + 1]) / (j + 1)
        map_scores.append(avg_precision / np.sum(relevant_items))

    if len(map_scores) == 0:  # Handle case where no valid MAP scores were calculated
        return 0.0
    return np.mean(map_scores)

# Function to calculate NDCG@k
def ndcg_at_k(y_true, y_pred, k=10):
    ndcg_scores = []
    for i in range(y_true.shape[0]):
        top_k_preds = np.argsort(y_pred[i, :])[-k:]  # Get top k predictions for user i
        relevant_items = y_true[i, top_k_preds]

        if np.sum(relevant_items) == 0:  # If there are no relevant items, skip this user
            continue

        dcg = np.sum(relevant_items / np.log2(np.arange(2, len(relevant_items) + 2)))
        ideal_dcg = np.sum(np.sort(relevant_items)[::-1] / np.log2(np.arange(2, len(relevant_items) + 2)))

        if ideal_dcg == 0:  # Avoid division by zero in case of no relevant items
            continue

        ndcg_scores.append(dcg / ideal_dcg)

    if len(ndcg_scores) == 0:  # Handle case where no valid NDCG scores were calculated
        return 0.0
    return np.mean(ndcg_scores)


