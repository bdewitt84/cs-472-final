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
from src.cross_val import llocv

def accuracy_score(y_pred, y_true):
    if len(y_pred) != len(y_true):
        raise ValueException("The y_pred and y_true have different lengths")

    correct = [a == b for a, b in zip(y_pred, y_true)]
    return sum(correct) / len(y_pred)

def search_distance(
    options,
    train_features, train_labels,
    val_features, val_labels
):
    best_op = None
    best_acc = float('-inf')
    for name, option in options.items():
        # Configure and instantiate classifier
        params = {
            "n_neighbors": 5,
            "weights": always_one,
            "metric": minkowski,
            "metric_parameters": option
        }
        classifier = KNearestNeighbors(**params)

        # Hold Out Validationn        

        # Train classifier on training data
        classifier.fit(train_features, train_labels)
        # Make predictions on train features
        y_pred_train = classifier.predict(train_features) 
        # Make predictions on test features
        y_pred_val = classifier.predict(val_features)

        # Calculate accuracy on train and test labels
        train_accuracy = accuracy_score(y_pred_train, train_labels)
        val_accuracy = accuracy_score(y_pred_val, val_labels)
        cross_val_accuracy = llocv(classifier, train_features, train_labels)
        print("Option: %s - Accuracy (train, validate, cross validate): %f, %f, %f" % (name, train_accuracy, val_accuracy, cross_val_accuracy))
        if val_accuracy > best_acc:
            best_op = (name, option)
            best_acc = val_accuracy
        
    return best_op, best_acc
        

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

    (name, _), accuracy = search_distance(
        dict([(f"lambda{p}", {"p": p}) for p in range(1,11)]),
        train_features, train_labels,
        val_features, val_labels
    )

    print("Best Option: %s - Validate Accuracy: %f" % (name, accuracy))
