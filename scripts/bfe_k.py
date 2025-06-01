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
    column_names = features.columns.tolist()
    drop_order: [(str, int)] = []

    for _ in range(len(column_names)):
        max_accuracy = 0
        print(f"\n Considering: {column_names}")
        for column_name in column_names:
            print(f"Evaluating without {column_name}: ", end='')
            cur_features = features.drop(column_name, axis=1)
            score = evaluator(model, cur_features, labels)
            print(f"{score} ", end='')
            if score > max_accuracy:
                max_accuracy = score
                drop_column = column_name
                print("MAX", end='')
            print("\n", end='')
        print("Excluding", drop_column, max_accuracy)
        drop_order.append((drop_column, max_accuracy))
        features = features.drop(drop_column, axis=1)
        column_names.remove(drop_column)

    return drop_order



if __name__ == "__main__":

    from preprocessor.data_preprocessor import DataPreprocessor
    from src.utils import always_one, euclidian
    from src.knn import KNearestNeighbors
    from src.cross_val import llocv
    from pprint import pprint

    # Read in training data
    train_data = pd.read_csv(r'data/train.csv')
    train_df = pd.DataFrame(train_data)
    train_dp = DataPreprocessor(train_df)

    # Get training features and labels
    train_features = train_dp.get_features('Survived')
    train_labels = train_dp.get_labels('Survived')

    for k in range(11, 18, 2):
        params = {
            "n_neighbors": k,
            "weights": always_one,
            "metric": euclidian,
        }

        classifier = KNearestNeighbors(**params)

        drop_order = backward_feature_elimination(classifier, train_features, train_labels, llocv)
        pprint(drop_order)

        data = "feature,accuracy\n"
        for feature, accuracy in drop_order:
            data += f"{feature},{accuracy}\n"

        with open(f"./eval_bfe_xval_{k}_out.csv", "w") as f:
            f.write(data)
