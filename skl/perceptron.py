#./skl/knn.py

"""Sci-kit Learn KNN model used for reference and comparison in cs-472-final"""


# Imports
from sklearn.linear_model import Perceptron
import pandas as pd
from preprocessor.data_preprocessor import DataPreprocessor
import numpy as np


# Metadata
__author__ = "Brett DeWitt"
__date__ = "2025-5-20"


if __name__ == "__main__":

    # Read in training data
    train_data = pd.read_csv(r'data/train.csv')
    train_df = pd.DataFrame(train_data)
    train_dp = DataPreprocessor(train_df)

    # Get training features and labels
    train_features = train_dp.get_features('Survived')
    train_labels = train_dp.get_labels('Survived')

    # Read in val data
    val_data = pd.read_csv('data/dev.csv')
    val_df = pd.DataFrame(val_data)
    val_dp = DataPreprocessor(val_df)

    # Get val features and lables
    val_features = val_dp.get_features('Survived')
    val_labels = val_dp.get_labels('Survived')

    # Configure and instantiate classifier
    classifier = Perceptron()

    print("Epoch\tTrain Accuracy\tValidation Accuracy")
    for i in range(1,1001):
        # Train classifier on training data
        classifier.partial_fit(train_features, train_labels, np.unique(train_labels))

        # Calculate accuracy on val labels
        t_accuracy = classifier.score(train_features, train_labels)
        v_accuracy = classifier.score(val_features, val_labels)
        
        if i % 50 == 0:
            print("%d\t%f" % (i, t_accuracy, v_accuracy))

    
    
