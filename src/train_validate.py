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
from src.utils import always_one, euclidian

def accuracy_score(y_pred, y_true):
    if len(y_pred) != len(y_true):
        raise ValueException("The y_pred and y_true have different lengths")

    correct = [a == b for a, b in zip(y_pred, y_true)]
    return sum(correct) / len(y_pred)

if __name__ == "__main__":

    # Read in training data
    train_data = pd.read_csv(r'./train.csv')
    train_df = pd.DataFrame(train_data)
    train_dp = DataPreprocessor(train_df)

    # Get training features and labels
    train_features = train_dp.get_features('Survived')
    train_labels = train_dp.get_labels('Survived')

    # Read in test data
    val_data = pd.read_csv('./dev.csv')
    val_df = pd.DataFrame(val_data)
    val_dp = DataPreprocessor(val_df)

    # Get test features and lables
    val_features = val_dp.get_features('Survived')
    val_labels = val_dp.get_labels('Survived')

    # Configure and instantiate classifier
    params = {
        "n_neighbors": 5,
        "weights": always_one,
        "metric": euclidian,
    }
    classifier = KNeighborsClassifier(**params)

    # Train classifier on training data
    classifier.fit(train_features, train_labels)

    # Make predictions on test features
    y_pred = classifier.predict(val_features)

    # Calculate accuracy on test labels
    accuracy = accuracy_score(y_pred, val_labels)

    print("Accuracy:", accuracy)
