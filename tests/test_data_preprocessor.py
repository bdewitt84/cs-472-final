import pytest
import pandas as pd
import numpy as np
from preprocessor.data_preprocessor import DataPreprocessor  # Replace with actual module name

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': [1, 2, np.nan, 4],
        'B': [1, 2, 2, np.nan],
        'C': ['male', 'female', 'female', 'male'],
        'D': ['X', 'Y', 'X', 'Z'],
        'label': [0, 1, 0, 1]
    })

def test_drop_features(sample_df):
    processor = DataPreprocessor(sample_df)
    processor.drop_features(['A', 'B'])
    assert 'A' not in processor.df.columns
    assert 'B' not in processor.df.columns

def test_impute_missing_with_median(sample_df):
    processor = DataPreprocessor(sample_df)
    median_A = sample_df['A'].median()
    processor.impute_missing_with_median(['A'])
    assert processor.df['A'].isnull().sum() == 0
    assert processor.df.loc[2, 'A'] == median_A

def test_impute_missing_with_mode(sample_df):
    processor = DataPreprocessor(sample_df)
    mode_B = sample_df['B'].mode()[0]
    processor.impute_missing_with_mode(['B'])
    assert processor.df['B'].isnull().sum() == 0
    assert processor.df.loc[3, 'B'] == mode_B

def test_map_categorical(sample_df):
    processor = DataPreprocessor(sample_df)
    processor.map_categorical('C', {'male': 0, 'female': 1})
    assert set(processor.df['C'].unique()) <= {0, 1}

def test_one_hot_encode(sample_df):
    processor = DataPreprocessor(sample_df)
    processor.one_hot_encode(['D'])
    for val in sample_df['D'].unique():
        assert f'D_{val}' in processor.df.columns
    assert 'D' not in processor.df.columns

def test_scale_numeric_features_minmax(sample_df):
    processor = DataPreprocessor(sample_df)
    processor.impute_missing_with_median(['A'])  # Ensure no NaNs
    processor.scale_numeric_features(['A'], method='minmax')
    assert processor.df['A'].min() == pytest.approx(0)
    assert processor.df['A'].max() == pytest.approx(1)

def test_scale_numeric_features_zscore(sample_df):
    processor = DataPreprocessor(sample_df)
    processor.impute_missing_with_median(['A'])  # Ensure no NaNs
    processor.scale_numeric_features(['A'], method='zscore')
    mean = processor.df['A'].mean()
    std = processor.df['A'].std()
    assert mean == pytest.approx(0, abs=1e-7)
    assert std == pytest.approx(1, abs=1e-7) or std == pytest.approx(0, abs=1e-7)  # 0 if no variance

def test_get_features(sample_df):
    processor = DataPreprocessor(sample_df)
    features = processor.get_features('label')
    assert 'label' not in features.columns

def test_get_labels(sample_df):
    processor = DataPreprocessor(sample_df)
    labels = processor.get_labels('label')
    assert labels.equals(sample_df['label'])

def test_get_dataframe_returns_copy(sample_df):
    processor = DataPreprocessor(sample_df)
    df_copy = processor.get_dataframe()
    df_copy['A'] = 999
    assert processor.df['A'].iloc[0] != 999  # original should remain unchanged
