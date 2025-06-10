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
import numpy as np
from preprocessor.data_preprocessor import DataPreprocessor
import pickle
from util.helper import safe_write

if __name__ == "__main__":

    # Read in test data
    test_data = pd.read_csv(r'data/kaggle-test-proc.csv')
    test_df = pd.DataFrame(test_data)
    test_dp_knn = DataPreprocessor(test_df)
    test_dp_knn.drop_features(["SibSp", "Fare", "Pclass_3", "Embarked_S"])

    # Features for non KNN models
    test_dp = DataPreprocessor(test_df.copy())

    # Get training features and IDs

    test_features_knn = test_dp_knn.get_features('PassengerId')
    test_ids_knn = test_dp_knn.get_labels('PassengerId')

    test_features = test_dp.get_features('PassengerId')
    test_ids = test_dp.get_labels('PassengerId')

    with open("models/model_bfe_trained.pkl", "rb") as f:
        classifier = pickle.load(f)

        # Make predictions on test features (knn)
        y_pred = classifier.predict(test_features_knn).astype(np.int64)
        
        # Store results
        result = pd.DataFrame({"PassengerId":test_ids_knn.values, "Survived":y_pred})
        print("Saving results...")
        safe_write('output/knn_predictions.csv', result.to_csv(index=False))


    with open("models/perceptron_trained.pkl", "rb") as f:
        classifier = pickle.load(f)

        # Make predictions on test features (perceptron)
        y_pred = classifier.predict(test_features)

        # Store results
        result = pd.DataFrame({"PassengerId":test_ids.values, "Survived":y_pred})
        print("Saving results...")
        safe_write('output/perceptron_predictions.csv', result.to_csv(index=False))

    with open("models/dt_trained.pkl", "rb") as f:
        classifier = pickle.load(f)

        # Make predictions on test features (perceptron)
        y_pred = classifier.predict(test_features)

        # Store results
        result = pd.DataFrame({"PassengerId":test_ids.values, "Survived":y_pred})
        print("Saving results...")
        safe_write('output/dt_predictions.csv', result.to_csv(index=False))

