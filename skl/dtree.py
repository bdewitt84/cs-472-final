#./skl/knn.py

"""Sci-kit Learn Dtree model used for reference and comparison in cs-472-final"""


# Imports
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from preprocessor.data_preprocessor import DataPreprocessor


# Metadata
__author__ = "Brett DeWitt"
__date__ = "2025-6-5"


if __name__ == "__main__":

    # Read in training data
    train_data = pd.read_csv(r'./data/train.csv')
    dev_data = pd.read_csv(r'./data/dev.csv')
    train_df = pd.DataFrame(train_data)
    dev_df = pd.DataFrame(dev_data)
    train_plus_dev_df = pd.concat([train_df, dev_df])
    train_plus_dev_dp = DataPreprocessor(train_plus_dev_df)

    # Get training features and labels
    X_train_plus_dev = train_plus_dev_dp.get_features('Survived')
    y_train_plus_dev = train_plus_dev_dp.get_labels('Survived')

    # Read in test data
    test_data = pd.read_csv('./data/test.csv')
    test_df = pd.DataFrame(test_data)
    test_dp = DataPreprocessor(test_df)

    # Get test features and labels
    test_features = test_dp.get_features('Survived')
    test_labels = test_dp.get_labels('Survived')

    # Configure and instantiate classifier
    params = {'criterion': 'entropy',
              'max_depth': None,
              'max_features': None,
              'min_samples_leaf': 4,
              'min_samples_split': 20,
              'splitter': 'best'
              }
    classifier = DecisionTreeClassifier(**params)

    # Train classifier on training data
    classifier.fit(X_train_plus_dev, y_train_plus_dev)

    # Make predictions on test features
    y_pred = classifier.predict(test_features)

    # Calculate accuracy on test labels
    accuracy = accuracy_score(test_labels, y_pred)

    print("Accuracy:", accuracy)
