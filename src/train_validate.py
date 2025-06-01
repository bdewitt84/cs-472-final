#src/train_evaluate.py
'''
Train and evaluate K Nearest Neighbor on the Titanic Dataset.
Derrived from skl/knn.py by Brett DeWitt
Author:  Makani Buckley
Date: May 24, 2025
'''
import pandas as pd

from preprocessor.data_preprocessor import DataPreprocessor
from src.knn import KNearestNeighbors
from src.utils import always_one, inverse, euclidian, negative, hump, decay, normalized_negative, minkowski
from src.grid_search import grid_search

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

    K = [5]
    WEIGHTS = {
        "Always One": (always_one, {"Empty": {}}),
        "Inverse Distance": (inverse, {"Empty": {}}),
        "Negative Distance": (negative, {"Empty": {}}),
        "Hump": (hump, {"Empty": {}}),
        "Decay": (decay, {"Empty": {}}),
        "Normalized Negative Distance": (normalized_negative, {"Empty": {}})
    }
    DISTANCE = {
        "Minkowski": (
            minkowski,
            dict([(p, {"p": p}) for p in range(1,11)]) 
        )
    }

    (k, (w, (_, (wp, _))), (d, (_, (dp, _)))), best_accuracy = grid_search(K, WEIGHTS, DISTANCE, train_features, train_labels, val_features, val_labels) 

    print("Best Options: %d, %s, %s, %s, %s - Best Accuracy: %f" % (k, w, wp, d, dp, best_accuracy))
