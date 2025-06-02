#./src/bfe.py
"""
Backward feature elimination over selected range of k
"""


# Imports
import pandas as pd

from preprocessor.data_preprocessor import DataPreprocessor
from src.utils import always_one, euclidian
from src.knn import KNearestNeighbors
from src.cross_val import llocv
from src.bfe import backward_feature_elimination
from pprint import pprint


# Metadata
__author__ = "Brett DeWitt"
__date__ = "2025.5.30"



if __name__ == "__main__":

    # Read in training data
    train_data = pd.read_csv(r'data/train.csv')
    train_df = pd.DataFrame(train_data)
    train_dp = DataPreprocessor(train_df)

    # Get training features and labels
    train_features = train_dp.get_features('Survived')
    train_labels = train_dp.get_labels('Survived')

    # Parameters
    MIN_K = 11
    MAX_K = 17

    for k in range(MIN_K, MAX_K+1, 2):
        params = {
            "n_neighbors": k,
            "weights": always_one,
            "metric": euclidian,
        }

        classifier = KNearestNeighbors(**params)

        drop_order = backward_feature_elimination(classifier, train_features, train_labels, llocv)
        pprint(drop_order)

        data = "feature,accuracy\n"
        for feature, accuracy in drop_order:
            data += f"{feature},{accuracy}\n"

        with open(f"./eval_bfe_xval_{k}_out.csv", "w") as f:
            f.write(data)
