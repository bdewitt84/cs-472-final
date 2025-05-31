#./src/eval_k.py
"""
Evaluates accuracy of model at various values of k
"""


# Imports
import pandas as pd

from preprocessor.data_preprocessor import DataPreprocessor
from src.knn import KNearestNeighbors
from src.utils import always_one, euclidian
from cross_val import llocv


# Metadata
__author__ = "Brett DeWitt"
__date__ = "2025/5/27"


if __name__ == "__main__":

    # Read in training data
    train_data = pd.read_csv(r'data/train.csv')
    train_df = pd.DataFrame(train_data)
    train_dp = DataPreprocessor(train_df)

    # Get training features and labels
    train_features = train_dp.get_features('Survived')
    train_labels = train_dp.get_labels('Survived')

    # Read in test data
    val_data = pd.read_csv('data/dev.csv')
    val_df = pd.DataFrame(val_data)
    val_dp = DataPreprocessor(val_df)

    # Get test features and lables
    val_features = val_dp.get_features('Survived')
    val_labels = val_dp.get_labels('Survived')

    # Configure and instantiate classifier
    params = {
        "weights": always_one,
        "metric": euclidian,
    }

    # Get odd values of K from 1 to 21
    k_vals = [ k for k in range(1, 22, 2)]

    data = "k,accuracy\n"
    for k in k_vals:
        params.update({
            "n_neighbors": k,
        })
        classifier = KNearestNeighbors(**params)
        accuracy = llocv(classifier, train_features, train_labels)
        print(k, accuracy)
        data += f"{k},{accuracy}\n"

    with open("./eval_k_out.csv", "w") as f:
        f.write(data)
