import pandas

def llocv(model, examples: pandas.DataFrame, labels: pandas.DataFrame):
    num_examples = len(examples)
    correct = 0

    for i in range(num_examples):
        train_features = examples.drop(examples.index[i])
        test_feature = examples.iloc[[i]]
        train_labels = labels.drop(labels.index[i])
        test_label = labels.iloc[[i]]

        model.fit(train_features, train_labels)
        pred = model.predict(test_feature)

        if pred.flatten()[0] == test_label.values.flatten()[0]:
            correct += 1

    score = correct / num_examples
    return score


if __name__ == '__main__':
    import pandas as pd
    from preprocessor.data_preprocessor import DataPreprocessor
    from knn import KNearestNeighbors

    from utils import always_one, euclidian
    from util.helper import accuracy_score

    # Read in training data
    train_data = pd.read_csv(r'data/train.csv')
    train_df = pd.DataFrame(train_data)
    train_dp = DataPreprocessor(train_df)

    # Get training features and labels
    train_features = train_dp.get_features('Survived')
    train_labels = train_dp.get_labels('Survived')

    train_df.to_numpy()

    params = {
        "n_neighbors": 5,
        "weights": always_one,
        "metric": euclidian,
    }

    classifier = KNearestNeighbors(**params)

    print(llocv(classifier, train_features, train_labels))
