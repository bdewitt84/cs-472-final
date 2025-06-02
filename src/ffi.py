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


def forward_feature_inclusion(model, features:pd.DataFrame, label:pd.DataFrame, evaluator:callable):
    """
    This function iteratively evaluates the model's performance by starting
    with an empty set of features and then, in each round, adding the single
    feature whose inclusion yields the highest evaluation score. The process
    continues until all available features have been added. The best performing
    set of features is the set up to and including the highest recorded sore.
    :param model:
    :param features:
    :param label:
    :param evaluator:
    :return:
    """
    column_names = features.columns.tolist()
    columns = [features[column_name] for column_name in column_names]
    df = pd.DataFrame()
    inclusion_order:(str, int) = []
    for _ in range(len(columns)):
        max_score = 0
        print(f"\nConsidering : {column_names}")
        for column_name in column_names:
            print(f"Evaluating with {column_name}: ", end='')
            current_df = pd.concat([df, features[column_name]], axis=1)
            accuracy = evaluator(model, current_df, label)
            print(f"{accuracy} ", end='')
            if accuracy > max_score:
                max_score = accuracy
                max_column = column_name
                print("MAX", end='')
            print("\n", end='')
        print("Including: ", max_column, max_score)
        column_names.remove(max_column)
        df = pd.concat([df, features[max_column]], axis=1)
        inclusion_order.append((max_column, max_score))

    return inclusion_order



if __name__ == "__main__":

    from preprocessor.data_preprocessor import DataPreprocessor
    from utils import always_one, euclidian
    from knn import KNearestNeighbors
    from src.cross_val import llocv
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

    inclusion_order = forward_feature_inclusion(classifier, train_features, train_labels, llocv)
    pprint(inclusion_order)

    data = "feature,accuracy\n"
    for feature, accuracy in inclusion_order:
        data += f"{feature},{accuracy}\n"

    with open("./eval_ffi_xval_out.csv", "w") as f:
        f.write(data)
