#./skl/perceptron.py
'''
A scikit-learn perceptron applied to the Titanic dataset. Copied and edited
from skl/knn.py by Brett Dewitt.
Author: Makani Buckley
Date: May 4, 2025
'''

# Imports
from sklearn.linear_model import Perceptron
import pandas as pd
from preprocessor.data_preprocessor import DataPreprocessor
import numpy as np

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

    max_epochs = 1000000
    
    print("Epoch\tRegularizer\tTrain Accuracy\tValidation Accuracy")
    for op in {None, "l1", "l2"}:
        # Configure and instantiate classifier
        classifier = Perceptron(penalty=op, max_iter=max_epochs)

        for i in range(1,max_epochs+1):
            # Train classifier on training data
            classifier.partial_fit(train_features, train_labels, np.unique(train_labels))

            if i % 10000 == 0:
                # Calculate accuracy on val labels
                t_accuracy = classifier.score(train_features, train_labels)
                v_accuracy = classifier.score(val_features, val_labels)
                print("%d\t%s\t%f\t%f" % (i, op, t_accuracy, v_accuracy))

    
    
