from preprocessor.data_preprocessor import DataPreprocessor
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, LeaveOneOut
import os

if __name__ == "__main__":
    # Define the output directory for results
    output_dir = './'
    os.makedirs(output_dir, exist_ok=True)

    # Data Loading and Preprocessing
    print("Loading and preprocessing data")
    train_data = pd.read_csv(r'./data/train.csv')
    dev_data = pd.read_csv(r'./data/dev.csv')
    train_df = pd.DataFrame(train_data)
    dev_df = pd.DataFrame(dev_data)
    train_plus_dev_df = pd.concat([train_df, dev_df])
    train_plus_dev_dp = DataPreprocessor(train_plus_dev_df)

    X_train_plus_dev = train_plus_dev_dp.get_features('Survived')
    y_train_plus_dev = train_plus_dev_dp.get_labels('Survived')

    # Define Hyperparameter Grid
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 3, 5, 7, 10, 15],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': [None, 'sqrt', 'log2'],
        'splitter': ['best', 'random']
    }
    print(f"\nDefined hyperparameter grid with {len(param_grid['criterion']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) * len(param_grid['min_samples_leaf']) * len(param_grid['max_features']) * len(param_grid['splitter'])} total combinations.")


    # Initialize Model and LOOCV
    dt_classifier = DecisionTreeClassifier(random_state=42)
    cv_strategy = LeaveOneOut()

    # Set up GridSearch
    grid_search = GridSearchCV(
        estimator=dt_classifier,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )

    print("\nStarting GridSearchCV with LOOCV")
    grid_search.fit(X_train_plus_dev, y_train_plus_dev)

    # Display Best Results
    print("\nBest Model")
    print("hyperparameters: ", grid_search.best_params_)
    print(f"LOOCV accuracy: {grid_search.best_score_:.4f}")

    # Extract results
    results_df = pd.DataFrame(grid_search.cv_results_)

    # Identify parameter columns
    param_cols = [col for col in results_df.columns if col.startswith('param_')]

    # Columns to keep for output
    report_cols = param_cols + [
        'mean_test_score',
        'std_test_score',
        'mean_train_score',
        'std_train_score',
        'rank_test_score',
        'mean_fit_time',
        'std_fit_time',
        'mean_score_time',
        'std_score_time'
    ]

    # Sort results by rank_test_score (1 is best)
    results_df_sorted = results_df[report_cols].sort_values(by='rank_test_score').reset_index(drop=True)

    # Save Results to CSV ---
    output_filename = os.path.join(output_dir, 'dt_loocv_grid_search_train_dev.csv')
    results_df_sorted.to_csv(output_filename, index=False)

    # Top N Results
    print("\nTop 10 Hyperparameter Combinations (LOOCV)")
    print(results_df_sorted.head(10).to_string()) # .to_string() displays more rows/columns
