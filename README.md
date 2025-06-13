# cs-472-final
Final Project  
Machine Learning  
University of Oregon    
Spring 2025

## Authors
Makani Buckley  
Brett DeWitt

## Description

Final project for CS 472 - Machine Learning, UO Spring '25.  
- Implements KNN for use with the [Titanic dataset](https://www.kaggle.com/datasets/yasserh/titanic-dataset) from [Kaggle](https://www.kaggle.com/)  
- Implements DataPreprocessor: a shim for Pandas which streamlines data pre-processing  
- Implements various filter and wrapper methods to aid in feature selection, including Forward Feature Inclusion, Backward Feature Elimination, and Pearson Correlation Coefficient calculation  
- Automation scripts for feature selection and hyperparameter tuning
- Sklearn Decision Tree and Perceptron for comparative analysis
- See [File Structure](#file-structure) for details

## Installation instructions

Install repository using
```
git clone https://github.com/bdewitt84/cs-472-final.git
```
Navigate to the repository with
```
cd cs-472-final
```

Then install package dependencies using
```
pip install -r requirements.txt
```
Install local packages using
```
pip install -e .
```

## Usage

Train and Validate a K-Nearest Neighbors model on the titanic data in data/ using
```
python scripts/train_validate.py
```

## File Structure

```
**Authorship Legend:**

| Code | Author(s)        |
|------|------------------|
| BD   | Brett DeWitt     |
| MB   | Makani Buckley   |
| BOTH | Brett DeWitt, Makani Buckley |

---

.
├── README.md
├── **data/** # **Raw and processed datasets for the project**
│   ├── dev.csv                              # Processed Titanic development data **(BOTH)**
│   ├── kaggle-test-proc.csv                 # Processed Kaggle test data **(MB)**
│   ├── kaggle-titanic-test.csv              # Raw Kaggle test data **(MB)**
│   ├── kaggle-titanic.csv                   # Raw Titanic data **(BD)**
│   ├── test.csv                             # Processed Titanic test data **(BOTH)**
│   └── train.csv                            # Processed Titanic training data **(BOTH)**
├── description.txt                          # Describes the project's file structure
├── **models/** # **Pickled Python objects of trained and untrained machine learning models** **(MB)**
│   ├── dt_trained.pkl                       # Trained Decision Tree model
│   ├── model_bfe.pkl                        # Untrained KNN model using BFE-selected features (Best KNN Model)
│   ├── model_bfe_trained.pkl                # Trained KNN model using BFE-selected features (Best KNN Model)
│   ├── model_ffi.pkl                        # Untrained KNN model using FFI-selected features
│   ├── model_ffi_trained.pkl                # Trained KNN model using FFI-selected features
│   ├── model_full.pkl                       # Untrained KNN model using all features
│   ├── model_full_trained.pkl               # Trained KNN model using all features
│   ├── perceptron_trained.pkl               # Trained Perceptron model
│   └── perceptron_untrained.pkl             # Untrained Perceptron model
├── **output/** # **Results and outputs from project scripts** **(BD)**
│   ├── dt_loocv_grid_search_train.csv       # Decision Tree grid search results on the 'train' set
│   ├── dt_loocv_grid_search_train_dev.csv   # Decision Tree grid search results on combined 'train' and 'dev' sets
│   ├── dt_predictions.csv                   # Predictions of the best Decision Tree model on Kaggle test data **(MB)**
│   ├── eval_bfe_xval_11_out.csv             # BFE evaluation results at k=11
│   ├── eval_bfe_xval_13_out.csv             # BFE evaluation results at k=13
│   ├── eval_bfe_xval_15_out.csv             # BFE evaluation results at k=15
│   ├── eval_bfe_xval_17_out.csv             # BFE evaluation results at k=17
│   ├── eval_bfe_xval_out.csv                # Optimal BFE set evaluated with LOOCV
│   ├── eval_corr_xval_out.csv               # Pearson Correlation Coefficient results of features with 'Survival' target
│   ├── eval_ffi_11_xval_out.csv             # FFI evaluation results at k=11
│   ├── eval_ffi_13_xval_out.csv             # FFI evaluation results at k=13
│   ├── eval_ffi_15_xval_out.csv             # FFI evaluation results at k=15
│   ├── eval_ffi_17_xval_out.csv             # FFI evaluation results at k=17
│   ├── eval_ffi_1_xval_out.csv              # FFI evaluation results at k=1
│   ├── eval_ffi_9_xval_out.csv              # FFI evaluation results at k=9
│   ├── hyper_parameter_search.csv           # KNN hyperparameter search performance (all features) **(MB)**
│   ├── hyper_parameter_search_bfe.csv       # KNN hyperparameter search performance (BFE-selected features) **(MB)**
│   ├── hyper_parameter_search_ffi.csv       # KNN hyperparameter search performance (FFI-selected features) **(MB)**
│   ├── knn_predictions.csv                  # Predictions of the best KNN model on Kaggle test data **(MB)**
│   └── perceptron_predictions.csv           # Predictions of the best Perceptron model on Kaggle test data **(MB)**
├── **preprocessor/** # **Scripts for data preprocessing**
│   ├── __init__.py
│   ├── data_preprocessor.py
│   └── kaggle-test-parse.py                 # Preprocessing script for Kaggle test data **(MB)**
├── requirements.txt                         # Python libraries and modules required for the project
├── **scripts/** # **Various scripts for hyperparameter tuning and model evaluation**
│   ├── bfe_k.py                             # BFE evaluation at a range of K values **(BD)**
│   ├── eval_k.py                            # KNN evaluation at a range of K values **(BD)**
│   ├── ffi_k.py                             # FFI evaluation at a range of K values **(BD)**
│   ├── train_validate.py                    # Script to train and validate KNN on 'train' and 'dev' data (all features) **(MB)**
│   ├── train_validate_bfe.py                # Script to train and validate KNN on 'train' and 'dev' data (BFE features) **(BD)**
│   ├── train_validate_distance.py           # Script to train and validate KNN, testing different metric functions **(MB)**
│   ├── train_validate_ffi.py                # Script to train and validate KNN on 'train' and 'dev' data (FFI features) **(MB)**
│   ├── train_validate_total.py              # Script to train a best KNN on the combined 'train' and 'dev' data **(MB)**
│   └── train_validate_weights.py            # Script to train and validate KNN, testing different weights functions **(MB)**
├── setup.py                                 # Setup script for installing the project as a Python module
├── **skl/** # **Scikit-learn model implementations and scripts**
│   ├── dtree.py                             # Scikit-learn Decision Tree with example script **(BD)**
│   ├── dtree_grid_search.py                 # Grid search for Scikit-learn Decision Tree hyperparameters **(BD)**
│   ├── knn.py                               # Scikit-learn KNN with example script **(BD)**
│   ├── perceptron.py                        # Script to train/test Perceptron and search for optimal hyperparameters **(MB)**
│   ├── total_train_dt.py                    # Script to train Decision Tree on combined 'train' and 'dev' data and save a pickled copy **(MB)**
│   └── total_train_perceptron.py            # Script to train Perceptron on combined 'train' and 'dev' data and save a pickled copy **(MB)**
├── **src/** # **Core implementation source files for the project**
│   ├── __init__.py
│   ├── bfe.py                               # Backward Feature Elimination implementation **(BD)**
│   ├── correlation.py                       # Pearson Correlation Coefficient implementation with example usage **(BD)**
│   ├── cross_val.py                         # LOOCV implementation **(BD)**
│   ├── evaluate.py                          # Script to test best KNN, Perceptron, and Decision Tree on the test data **(MB)**
│   ├── ffi.py                               # Forward Feature Inclusion implementation **(BD)**
│   ├── grid_search.py                       # Script for KNN hyperparameter grid search **(MB)**
│   ├── kaggle-evaluate.py                   # Script to test best KNN, Perceptron, and Decision Tree on the Kaggle test data **(MB)**
│   ├── knn.py                               # Custom KNN model implementation **(MB)**
│   └── utils.py                             # Various utility functions (e.g., metric, weights functions) **(MB)**
├── **tests/** # **Unit tests for project components**
│   ├── test_data_preprocessor.py            # Unit tests for DataPreprocessor **(BD)**
│   ├── test_helper.py                       # Unit tests for helper functions **(BD)**
│   └── test_knn.py                          # Unit tests for KNN implementation **(MB)**
└── **util/** # **General utility functions**
    ├── __init__.py
    └── helper.py                            # Helper functions **(BD)** 
```
