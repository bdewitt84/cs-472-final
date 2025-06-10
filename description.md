```
.
├── README.md
├── data # A folder containing the raw data for this project (Author: Makani Buckley and Brett DeWitt)
│   ├── dev.csv # The processed titanic development data (Author: Makani Buckley and Brett DeWitt)
│   ├── kaggle-test-proc.csv # The processed kaggle test data (Author: Makani Buckley)
│   ├── kaggle-titanic-test.csv # The raw kaggle test data (Author: Makani Buckley)
│   ├── kaggle-titanic.csv # The raw titanic data (Author: Brett DeWitt)
│   ├── test.csv # The processed titanic test data (Author: Makani Buckley and Brett DeWitt)
│   └── train.csv # The processed titanic train.data (Author: Makani Buckley and Brett DeWitt)
├── description.txt # Describes the file structure of the project
├── models # A folder of the machine learning models of the project as pickled python objects (Author: Makani Buckley)
│   ├── dt_trained.pkl # The trained decision tree (Author: Makani Buckley)
│   ├── model_bfe.pkl # The untrained KNN using BFE Features (Best KNN Model) (Author: Makani Buckley)
│   ├── model_bfe_trained.pkl # The trained KNN using BFE features (Best KNN Model) (Author: Makani Buckley)
│   ├── model_ffi.pkl # The untrained KNN using FFI features (Author: Makani Buckley)
│   ├── model_ffi_trained.pkl # The trained KNN using FFI features (Author: Makani Buckley)
│   ├── model_full.pkl # The untrained KNN using Full Features (Author: Makani Buckley)
│   ├── model_full_trained.pkl # The trained KNN using Full features (Author: Makani Buckley)
│   ├── perceptron_trained.pkl # The trained Perceptron (Author: Makani Buckley)
│   └── perceptron_untrained.pkl # The untrained Perceptron (Author: Makani Buckley)
├── output # A folder containing the output of our project scripts (Author: Brett DeWitt)
│   ├── dt_loocv_grid_search_train.csv # DataTree grid search on 'train' set (Author: Brett DeWitt)
│   ├── dt_loocv_grid_search_train_dev.csv # DataTree grid search on combined 'train' and 'dev' sets (Author: Brett DeWitt)
│   ├── dt_predictions.csv # The predictions of the best DT on the kaggle test data (Author: Makani Buckley)
│   ├── eval_bfe_xval_11_out.csv # BFE evaluated at k=11 (Author: Brett DeWitt)
│   ├── eval_bfe_xval_13_out.csv # BFE evaluated at k=13 (Author: Brett DeWitt)
│   ├── eval_bfe_xval_15_out.csv # BFE evaluated at k=15 (Author: Brett DeWitt)
│   ├── eval_bfe_xval_17_out.csv # BFE evaluated at k=17 (Author: Brett DeWitt)
│   ├── eval_bfe_xval_out.csv # Optimal BFE set evaluated with LOOCV
│   ├── eval_corr_xval_out.csv # Pearson Correlation Coefficient of the datasets features with target 'Survival'
│   ├── eval_ffi_11_xval_out.csv # FFI evaluated at K=11 (Author: Brett DeWitt)
│   ├── eval_ffi_13_xval_out.csv # FFI evaluated at K=13 (Author: Brett DeWitt)
│   ├── eval_ffi_15_xval_out.csv # FFI evaluated at K=15 (Author: Brett DeWitt)
│   ├── eval_ffi_17_xval_out.csv # FFI evaluated at K=17 (Author: Brett DeWitt)
│   ├── eval_ffi_1_xval_out.csv  # FFI evaluated at K=1 (Author: Brett DeWitt)
│   ├── eval_ffi_9_xval_out.csv # FFI evaluated at K=9 (Author: Brett DeWitt)
│   ├── hyper_parameter_search.csv # The performance of each hyperparameter combination of KNN (Full Features) (Author: Makani Buckley)
│   ├── hyper_parameter_search_bfe.csv # The performance of each hyperparameter combination of KNN (BFE Features) (Author: Makani Buckley)
│   ├── hyper_parameter_search_ffi.csv # The performance of each hyperparameter combination of KNN (FFI Features) (Author: Makani Buckley)
│   ├── knn_predictions.csv # The predictions of the best KNN on the kaggle test data (Author: Makani Buckley)
│   └── perceptron_predictions.csv # The predictions of the best Perceptron on the kaggle test data (Author: Makani Buckley)
├── preprocessor
│   ├── __init__.py
│   ├── data_preprocessor.py
│   └── kaggle-test-parse.py # A script for preprocessing the kaggle test data (Author: Makani Buckley)
├── requirements.txt # The python libraries and modules required for the project
├── scripts
│   ├── bfe_k.py # BFE evaluated at a range of K values (Author: Brett DeWitt)
│   ├── eval_k.py # Evaluations of KNN at a range of K values (Author: Brett DeWitt)
│   └── ffi_k.py # FFI evaluated at a range of K values (Author: Brett DeWitt)
├── setup.py # The setup script required for installing the project as a python module
├── skl # A folder of the scikit-learn models (Author: Brett DeWitt)
│   ├── dtree.py # Sklearn DecisionTree with example script (Author: Brett DeWitt)
│   ├── dtree_grid_search.py # Gridsearch of Sklearn DecisionTree hyperparameters (Author: Brett DeWitt)
│   ├── knn.py  # Sklearn KNN with example script (Author: Brett DeWitt)
│   ├── perceptron.py # A script to train and test percepton on the train and development sets and to search for optimal hyperparameters (Author: Makani Buckley)
│   ├── total_train_dt.py # A script to train a decision tree on the combined train and development data and save a pickled copy of it (Author: Makani Buckley)
│   └── total_train_perceptron.py # A script to train a perceptron on the combined train and development data and save a pickled copy of it (Author: Makani Buckley)
├── src # The source files for this project
│   ├── __init__.py
│   ├── bfe.py # Backward Feature Elimination implementation (Author: Brett DeWitt)
│   ├── correlation.py # Pandas' Pearson Correlation Coefficient with example usage (Author: Brett DeWitt)  
│   ├── cross_val.py # LOOCV implementation (Author: Brett DeWitt)
│   ├── evaluate.py # A script to test the best KNN, Perceptron, and Decision Tree model on the test data (Author: Makani Buckley)
│   ├── ffi.py # Forward Feature Includion implementation (Author: Brett DeWitt)
│   ├── grid_search.py # A script defining grid search for finding the best hyperparameter options for KNN (Author: Makani Buckley)
│   ├── kaggle-evaluate.py # A script to test the best KNN, Perceptron, and Decision Tree model on the kaggle test data (Author: Makani Buckley)
│   ├── knn.py # A script implementing the a KNN model in python (Author: Makani Buckley)
│   ├── train_validate.py # A script to train and validate KNN on the train and development data (Full featrures) (Author: Makani Buckley)
│   ├── train_validate_bfe.py # A script to train and validate KNN on the train and development data (BFE features)
│   ├── train_validate_distance.py # A script to train and validate KNN on the train and development data, testing different metric functions (Author: Makani Buckley)
│   ├── train_validate_ffi.py # A script to train and validate KNN on the train and development data (FFI features) (Author: Makani Buckley)
│   ├── train_validate_total.py # A script to train a best KNN on the combined train and development data (Author: Makani Buckley)
│   ├── train_validate_weights.py # A script to train and validate KNN on the train and development data, testing different weights functions (Author: Makani Buckley)
│   └── utils.py # A script definint various useful utility functions such as metric and weights functions (Author: Makani Buckley)
├── tests
│   ├── test_data_preprocessor.py # Unit tests for DataPreprocessor (Author: Brett DeWitt)
│   ├── test_helper.py # Unit tests for helper functions (Author: Brett DeWitt)
│   └── test_knn.py # Unit tests for KNN implementation (Author: Makani Buckley)
└── util
    ├── __init__.py
    └── helper.py # Helper functions (Author: Brett DeWitt) 
```
