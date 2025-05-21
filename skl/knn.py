#./skl/knn.py

"""Sci-kit Learn KNN model used for reference and comparison in cs-472-final"""
from sklearn.metrics import accuracy_score
# Imports
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from preprocessor.data_preprocessor import DataPreprocessor


# Metadata
__author__ = "Brett DeWitt"
__date__ = "2025-5-20"


if __name__ == "__main__":

    # Read in training data
    train_data = pd.read_csv(r'./train.csv')
    train_df = pd.DataFrame(train_data)
    train_dp = DataPreprocessor(train_df)

    # Get training features and labels
    train_features = train_dp.get_features('Survived')
    train_labels = train_dp.get_labels('Survived')

    # Read in test data
    test_data = pd.read_csv('./dev.csv')
    test_df = pd.DataFrame(test_data)
    test_dp = DataPreprocessor(test_df)

    # Get test features and lables
    test_features = test_dp.get_features('Survived')
    test_labels = test_dp.get_labels('Survived')

    # Configure and instantiate classifier
    params = {
        "n_neighbors": 5,
        "weights": "uniform",
        "algorithm": "brute",
        "metric": "minkowski",
        "p": 2,
        "n_jobs": None
    }
    classifier = KNeighborsClassifier(**params)

    # Train classifier on training data
    classifier.fit(train_features, train_labels)

    # Make predictions on test features
    y_pred = classifier.predict(test_features)

    # Calculate accuracy on test labels
    accuracy = accuracy_score(y_pred, test_labels)

    print("Accuracy:", accuracy)
