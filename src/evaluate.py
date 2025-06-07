# src/evaluate.py
'''
Evaluate the best KNN and Perceptron models on the
test data.
Derived from skl/knn.py by Brett Dewitt.
Author: Makani Buckley
Date: June 4, 2025
'''

# Imports
from sklearn.linear_model import Perceptron
from src.knn import KNearestNeighbors
from src.utils import accuracy_score
import pandas as pd
from preprocessor.data_preprocessor import DataPreprocessor
import pickle

if __name__ == "__main__":

    # Read in test data
    test_data = pd.read_csv(r'data/test.csv')
    test_df = pd.DataFrame(test_data)
    test_dp_knn = DataPreprocessor(test_df)
    test_dp_knn.drop_features(["SibSp", "Fare", "Pclass_3", "Embarked_S"])

    test_dp_per = DataPreprocessor(test_df.copy())

    # Get training features and labels

    test_features_knn = test_dp_knn.get_features('Survived')
    test_labels_knn = test_dp_knn.get_labels('Survived')

    test_features_per = test_dp_per.get_features('Survived')
    test_labels_per = test_dp_per.get_labels('Survived')

    with open("models/model_bfe_trained.pkl", "rb") as f:
        classifier = pickle.load(f)
        # Make predictions on test features (knn)
        y_pred = classifier.predict(test_features_knn)

        # Calculate accuracy on test labels
        accuracy = accuracy_score(y_pred, test_labels_knn)

        print("KNN Accuracy: ", accuracy)

    with open("models/perceptron_trained.pkl", "rb") as f:
        classifier = pickle.load(f)

        # Calculate accuracy on test labels
        accuracy = classifier.score(test_features_per, test_labels_per)

        print("Perceptron Accuracy:", accuracy)
