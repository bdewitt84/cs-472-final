#./preprocessor/kaggle-test-parse.py
"""
Parses the test data for the kaggle Titanic competition.
Derived from preprocessor/data_preprocessor.py by Brett DeWitt
Author: Makani Buckley
Date: June 8, 2025
"""

# Imports
import pandas as pd
import seaborn as sns # For histogram in plot_column_distribution
import matplotlib.pyplot as plt
from util.helper import safe_write
from preprocessor.data_preprocessor import DataPreprocessor

if __name__ == '__main__':
    # Import the data
    df_raw = pd.read_csv('data/kaggle-titanic-test.csv')
    processor = DataPreprocessor(df_raw)

    # Shuffle features
    processor.shuffle_rows(42)

    # Drop irrelevant features (Note: Keep Passenger ID)
    processor.drop_features(['Name', 'Ticket', 'Cabin'])

    # Impute missing values
    processor.impute_missing_with_median(['Age'])
    processor.impute_missing_with_mode(['Embarked'])
    processor.impute_missing_with_median(['Fare'])

    # Map categorical features
    processor.map_categorical('Sex', {'male': 0, 'female': 1})

    # One-hot encode categorical features
    processor.one_hot_encode(['Embarked', 'Pclass'])

    # Scale numeric features
    processor.scale_numeric_features(['Age', 'Fare'], method='minmax')

    # Save processed data
    safe_write('data/kaggle-test-proc.csv', processor.df.to_csv(index=False))
