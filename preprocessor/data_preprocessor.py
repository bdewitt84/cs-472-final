#./preprocessor/data_preprocessor.py

"""
Data preprocessor for cs-472-file.
Provides functions for common data preprocessing tasks
"""

# Imports
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Metadata
__author__ = "Brett DeWitt"
__date__ = "2025-5-8"


class DataPreprocessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()


    def drop_features(self, cols: list):
        self.df.drop(cols, axis=1, inplace=True)


    def impute_missing_with_median(self, cols: list):
        for col in cols:
            self.df[col] = self.df[col].fillna(self.df[col].median())


    def impute_missing_with_mode(self, cols: list):
        for col in cols:
            self.df[col] = self.df[col].fillna(self.df[col].mode()[0])


    def map_categorical(self, col: str, mapping: dict):
        self.df[col] = self.df[col].map(mapping)


    def one_hot_encode(self, cols: list, drop_original=True, prefix=True):
        for col in cols:
            dummies = pd.get_dummies(self.df[col], prefix=col if prefix else None)
            self.df = pd.concat([self.df, dummies], axis=1)
            if drop_original:
                self.df.drop(col, axis=1, inplace=True)


    def scale_numeric_features(self, cols: list, method='minmax'):
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
        return self.df.drop(label_col, axis=1)


    def get_labels(self, label_col: str):
        return self.df[label_col]


    def plot_column_distribution(self, col: str, bins=30):
        plt.figure(figsize=(8, 6))
        sns.histplot(self.df[col], bins=bins, kde=True, color='skyblue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.show()


    def get_dataframe(self):
        return self.df.copy()


    def to_csv(self):
        return self.df.to_csv(index=False)


    def shuffle_rows(self, random_state=42):
        self.df = self.df.sample(frac=1, random_state=random_state)
        self.df.reset_index(drop=True)



if __name__ == '__main__':
    df_raw = pd.read_csv('../kaggle-titanic.csv')
    processor = DataPreprocessor(df_raw)

    # Drop irrelevant features
    processor.drop_features(['Name', 'Ticket', 'Cabin', 'PassengerId'])

    # Impute missing values
    processor.impute_missing_with_median(['Age'])
    processor.impute_missing_with_mode(['Embarked'])

    # Map categorical features
    processor.map_categorical('Sex', {'male': 0, 'female': 1})

    # One-hot encode categorical features
    processor.one_hot_encode(['Embarked'])

    # Scale numeric features
    processor.scale_numeric_features(['Age', 'Fare'], method='minmax')

    # Get features and labels
    features = processor.get_features('Survived')
    labels = processor.get_labels('Survived')

    # Plot distribution
    processor.plot_column_distribution('Age')

    # Check processed DataFrame
    print(processor.get_dataframe().head())
    print(processor.get_dataframe().info())
