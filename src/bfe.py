#./src/bfe.py
"""
Backward features selection
Finds the most effective features to eliminate from a set
"""


# Imports
import pandas as pd


# Metadata
__author__ = "Brett DeWitt"
__date__ = "2025.5.30"


def backward_feature_elimination(model, features:pd.DataFrame, labels:pd.DataFrame, evaluator:callable):
    """
    This function iteratively evaluates the model's performance by starting
    with all features and then, in each round, removing the single feature
    whose exclusion yields the highest evaluation score. The process continues
    until no features remain. The best performing set is the set of features
    that follow the highest recorded accuracy.
    :param model: sklearn compatible model
    :param features: dataframe of features to evaluate
    :param labels: dataframe of labels to evaluate
    :param evaluator: evaluator function that takes a model, features and labels
    :return: list of tuples of features elimination order and the accuracy of
             the features
    """
    column_names = features.columns.tolist()
    drop_order: [(str, int)] = []

    for _ in range(len(column_names)):
        max_score = 0
        print(f"\n Considering: {column_names}")
        for column_name in column_names:
            print(f"Evaluating without {column_name}: ", end='')
            cur_features = features.drop(column_name, axis=1)
            score = evaluator(model, cur_features, labels)
            print(f"{score} ", end='')
            if score > max_score:
                max_score = score
                drop_column = column_name
                print("MAX", end='')
            print("\n", end='')
        print("Excluding", drop_column, max_score)
        drop_order.append((drop_column, max_score))
        features = features.drop(drop_column, axis=1)
        column_names.remove(drop_column)

    return drop_order



if __name__ == "__main__":
    """
    Example script demonstrating the use of backward_feature_elimination()
    Uses /data/train.csv to initialize a KNN classifier and writes
    elimination order along with the accuracies associated with each
    feature elimination to the output file.
    """
    from preprocessor.data_preprocessor import DataPreprocessor
    from utils import always_one, euclidian
    from knn import KNearestNeighbors
    from src.cross_val import loocv
    from pprint import pprint

    # Read in training data
    train_data = pd.read_csv(r'data/train.csv')
    train_df = pd.DataFrame(train_data)
    train_dp = DataPreprocessor(train_df)

    # Get training features and labels
    train_features = train_dp.get_features('Survived')
    train_labels = train_dp.get_labels('Survived')

    params = {
        "n_neighbors": 5,
        "weights": always_one,
        "metric": euclidian,
    }

    classifier = KNearestNeighbors(**params)

    drop_order = backward_feature_elimination(classifier, train_features, train_labels, loocv)
    pprint(drop_order)

    data = "feature,accuracy\n"
    for feature, accuracy in drop_order:
        data += f"{feature},{accuracy}\n"

    with open("./eval_bfe_xval_out.csv", "w") as f:
        f.write(data)
