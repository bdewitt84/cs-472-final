import pandas as pd


def correlation(features:pd.DataFrame, labels:pd.DataFrame):
    """Calculates Pearson correlation between features and labels."""
    feature_names = features.columns.tolist()
    corrs:[(str, int)] = []
    for feature in feature_names:
        corr = features[feature].corr(labels)
        corrs.append((feature, corr))
    corrs.sort(key=lambda x: abs(x[1]), reverse=True)
    return corrs

if __name__ == '__main__':
    """Example script for correlation()
    outputs feature and Pearson correlation to csv"""
    from preprocessor.data_preprocessor import DataPreprocessor
    from pprint import pprint

    # Read in training data
    train_data = pd.read_csv(r'data/train.csv')
    train_df = pd.DataFrame(train_data)
    train_dp = DataPreprocessor(train_df)

    # Get training features and labels
    train_features = train_dp.get_features('Survived')
    train_labels = train_dp.get_labels('Survived')

    corrs = correlation(train_features, train_labels)
    pprint(corrs)


    data = "feature,correlation\n"
    for feature, correlation in corrs:
        data += f"{feature},{correlation}\n"

    with open("./eval_corr_xval_out.csv", "w") as f:
        f.write(data)

