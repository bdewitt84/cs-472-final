#./preprocessor/data_preprocessor.py

"""
Data preprocessor for cs-472-file.
Provides functions for common data preprocessing tasks
"""

# Imports
import pandas as pd
import seaborn as sns # For histogram in plot_column_distribution
import matplotlib.pyplot as plt
from util.helper import safe_write

# Metadata
__author__ = "Brett DeWitt"
__date__ = "2025-5-8"


class DataPreprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()


    def drop_features(self, cols: list[str]):
        """Drops columns from dataframe

        :param cols: list of columns to drop
        :return: None
        """
        self.df.drop(cols, axis=1, inplace=True)


    def impute_missing_with_median(self, cols: list[str]):
        """Imputes missing values with median of columns values

        :param cols: list of columns in which to impute missing values
        :return: None
        """
        for col in cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())


    def impute_missing_with_mode(self, cols: list[str]):
        """ Imputes missing values with mode of columns values

        :param cols: list of columns in which to impute missing values
        :return: None
        """
        for col in cols:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])


    def map_categorical(self, col: str, mapping: dict):
        """Replaces values in column with values defined in mapping

        :param col: column to map
        :param mapping: key value pair mapping
        :return: None
        """
        self.df[col] = self.df[col].map(mapping)


    def one_hot_encode(self, cols: list[str], drop_original=True, prefix=True):
        """Encodes columns into one-hot encoded columns

        :param cols: List of columns to encode
        :param drop_original: Drops the original column if True
        :param prefix: Prefixes encoded columns with original column name if True
        :return: None
        """
        for col in cols:
            dummies = pd.get_dummies(self.df[col], prefix=col if prefix else None, dtype=float)
            self.df = pd.concat([self.df, dummies], axis=1)
            if drop_original:
                self.df.drop(col, axis=1, inplace=True)


    def scale_numeric_features(self, cols: list[str], method='minmax'):
        """Scales numeric columns into range [0, 1]

        :param cols: List of columns to scale
        :param method: Method to use for scaling
        :return: None
        """
        for col in cols:
            if method == 'zscore':
                mean = self.df[col].mean()
                std = self.df[col].std()
                self.df[col] = (self.df[col] - mean) / std
            elif method == 'minmax':
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                self.df[col] = (self.df[col] - min_val) / (max_val - min_val)
            else:
                raise ValueError(f"Unknown scaling method '{method}'")


    def get_features(self, label_col: str):
        """Takes name of label column and returns list of features

        :param label_col: Name of label column
        :return: list of features
        """
        return self.df.drop(label_col, axis=1)


    def get_labels(self, label_col: str):
        """Takes name of label column and returns that column

        :param label_col: Name of label column
        :return: Label column
        """
        return self.df[label_col]


    def plot_column_distribution(self, col: str, bins=30):
        """Plots column distribution with matplotlib

        :param col: Column to plot
        :param bins: Number of bins to plot
        :return: None
        """
        plt.figure(figsize=(8, 6))
        sns.histplot(self.df[col], bins=bins, kde=True, color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.show()


    def get_dataframe(self):
        """Returns the current working dataframe

        :return: Dataframe
        """
        return self.df.copy()


    def to_csv(self):
        """Returns the current working dataframe in CSV format

        :return: String
        """
        return self.df.to_csv(index=False)


    def shuffle_rows(self, random_state=42):
        """Shuffles rows of dataframe

        :param random_state: Seed with which to initialize randomizer
        :return: None
        """
        self.df = self.df.sample(frac=1, random_state=random_state)
        self.df.reset_index(drop=True)


    def split_train_dev_test(self, train: int, dev: int, test: int):
        """Splits data into train, dev, and test sets.

        :param train: proportion of train set
        :param dev: proportion of dev set
        :param test: proportion of test set
        :return: train, dev, test pandas DataFrames
        """

        if train < 0 or dev < 0 or test < 0:
            raise ValueError(f"Train, dev, and test must be non-negative integers")

        total = train+dev+test

        if total == 0:
            raise ValueError(f"Sum of train, dev and test must be greater than 0")

        num_rows = len(self.df)
        train_pct = train / total
        train_end = int(num_rows * train_pct)

        dev_pct = dev / total
        dev_end = train_end + int(num_rows * dev_pct)

        train_df = self.df[:train_end]
        dev_df = self.df[train_end:dev_end]
        test_df = self.df[dev_end:]

        return train_df, dev_df, test_df


if __name__ == '__main__':
    df_raw = pd.read_csv('../kaggle-titanic.csv')
    processor = DataPreprocessor(df_raw)

    # Shuffle features
    processor.shuffle_rows(42)

    # Drop irrelevant features
    processor.drop_features(['Name', 'Ticket', 'Cabin', 'PassengerId'])

    # Impute missing values
    processor.impute_missing_with_median(['Age'])
    processor.impute_missing_with_mode(['Embarked'])

    # Map categorical features
    processor.map_categorical('Sex', {'male': 0, 'female': 1})

    # One-hot encode categorical features
    processor.one_hot_encode(['Embarked', 'Pclass'])

    # Scale numeric features
    processor.scale_numeric_features(['Age', 'Fare'], method='minmax')

    # Get features and labels
    features = processor.get_features('Survived')
    labels = processor.get_labels('Survived')

    # Plot distribution
    processor.plot_column_distribution('Age')

    # Split data
    train, dev, test = processor.split_train_dev_test(7,2,1)

    # Check processed DataFrames
    print(train.head())
    print(train.info())

    print(dev.head())
    print(dev.info())

    print(test.head())
    print(test.info())

    # Save processed data
    safe_write('./train.csv', train.to_csv(index=False))
    safe_write('./dev.csv', dev.to_csv(index=False))
    safe_write('./test.csv', test.to_csv(index=False))
