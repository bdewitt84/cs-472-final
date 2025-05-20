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


def test_shuffle_rows(sample_df):
    processor = DataPreprocessor(sample_df)

    pre_shuffle_df = processor.df.copy()
    processor.shuffle_rows(42)

    # Order should change
    assert not pre_shuffle_df.equals(processor.df)

    # Data should remain unchanged, regardless of order
    pd.testing.assert_frame_equal(
        pre_shuffle_df.sort_values(by=pre_shuffle_df.columns.tolist()).reset_index(drop=True),
        processor.df.sort_values(by=processor.df.columns.tolist()).reset_index(drop=True)
    )


def test_get_dataframe_returns_copy(sample_df):
    processor = DataPreprocessor(sample_df)
    df_copy = processor.get_dataframe()
    df_copy['A'] = 999
    assert processor.df['A'].iloc[0] != 999  # original should remain unchanged


def test_split_train_dev_test():
    """Tests that the split is proportional to the number of samples in the data."""

    df = pd.DataFrame({
        'A': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        'B': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })

    preprocessor = DataPreprocessor(df)
    train, dev, test = preprocessor.split_train_dev_test(7, 2, 1)

    assert len(train) == 7
    assert len(dev) == 2
    assert len(test) == 1


def test_split_train_dev_test_odd():
    """Tests that the split consumes all the samples when the number of samples
    does not split evenly"""

    df = pd.DataFrame({
        'A': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
        'B': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
    })

    preprocessor = DataPreprocessor(df)
    train, dev, test = preprocessor.split_train_dev_test(7, 2, 1)

    assert len(train) + len(dev) + len(test) == 11


def test_split_train_dev_test_complete():
    """Tests that the split preserves the integrity of the data"""

    df = pd.DataFrame({
        'A': [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
        'B': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        'label': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]
    })

    preprocessor = DataPreprocessor(df)

    train, dev, test = preprocessor.split_train_dev_test(7, 2, 1)

    combined_indices = set(train.index) | set(dev.index) | set(test.index)
    assert len(combined_indices) == 11

    recombined_dataframe = pd.concat([train, dev, test]).sort_index()
    pd.testing.assert_frame_equal(recombined_dataframe, preprocessor.df)
