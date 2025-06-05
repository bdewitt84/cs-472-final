'''
Implements a Grid Search method for finding the optimal hyperparameters for a 
KNN model.
Author: Makani Buckley
Date: May 31, 2025
'''
from itertools import product
from src.knn import KNearestNeighbors
from src.cross_val import loocv
from src.utils import accuracy_score

'''
A grid search method for finding hyperparameters
Arguments
    - K: a list of values of n_neighbors to test
    - WEIGHTS: a dictionary of labeled weight functions to test. Weight
      functions are stored as (function, PARAMETERS) pairs where PARAMETERS is
      a labeled dictionary of parameter dictionaries to test.
    - DISTANCE: a dictionary of labeled metric functions to test. Metric
      functions are stored as (function, PARAMETERS) pairs where PARAMETERS is
      a labeled dictionary of parameter dictionaries to test.
    - train_features, train_labels: the training data to train the model with.
    - val_features, val_labels: the validation data to evaluate the model with.
'''
def grid_search (
    K, WEIGHTS, DISTANCE,
    train_features, train_labels,
    val_features, val_labels
):
    # Initialize Tracking Variables
    best_op = None
    best_acc = float("-inf")

    # Print header
    print("N_NEIGHBORS\tWEIGHTS\tDISTANCE\tWEIGHTS PARAMS\tDIST PARAMS\tTRAIN ACC\tVAL ACC\tCROSS VAL ACC")

    # Iterate through all combinations of options
    for n_neighbors, (w_name, (w_func, w_PAR)), (d_name, (d_func, d_PAR)) in product(K, WEIGHTS.items(), DISTANCE.items()):

        # Iterate through all combinations of parameters
        for (w_par_name, w_par), (d_par_name, d_par) in product(w_PAR.items(), d_PAR.items()):
            # Configure and instantiate classifier
            params = {
                "n_neighbors": n_neighbors,
                "weights": w_func,
                "metric": d_func,
                "weights_parameters": w_par,
                "metric_parameters": d_par
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
            cross_val_accuracy = loocv(classifier, train_features, train_labels)
            print("%d\t%s\t%s\t%s\t%s\t%f\t%f\t%f" % (n_neighbors, w_name, d_name, w_par_name, d_par_name, train_accuracy, val_accuracy, cross_val_accuracy))
            if cross_val_accuracy > best_acc:
                best_op = (n_neighbors, (w_name, (w_func, (w_par_name, w_par))), (d_name, (d_func, (d_par_name, d_par))))
                best_acc = cross_val_accuracy

    return best_op, best_acc
