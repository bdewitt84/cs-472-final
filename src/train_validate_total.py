#src/train_validate_total.py
'''
Cross-validate best models on total of train and development data
Derived from skl/knn.py by Brett Dewitt.
Author: Makani Buckley
Date: June 4, 2024
'''

# Imports
import pandas as pd
from preprocessor.data_preprocessor import DataPreprocessor
import numpy as np
import pickle

from src.knn import KNearestNeighbors
from src.utils import always_one, inverse, euclidian, negative, hump, decay, normalized_negative, minkowski
from src.grid_search import grid_search
from src.cross_val import llocv

if __name__ == "__main__":

    # Read in training data
    train_data = pd.read_csv(r'data/train.csv')
    train_df = pd.DataFrame(train_data)
    train_dp = DataPreprocessor(train_df)

    # Get training features and labels
    train_features = train_dp.get_features('Survived')
    train_labels = train_dp.get_labels('Survived')

    # Read in val data
    val_data = pd.read_csv('data/dev.csv')
    val_df = pd.DataFrame(val_data)
    val_dp = DataPreprocessor(val_df)

    # Get val features and labels
    val_features = val_dp.get_features('Survived')
    val_labels = val_dp.get_labels('Survived')

    # Get concatenated data
    cat_df = pd.concat([train_df, val_df])
    cat_dp = DataPreprocessor(cat_df)

    # Get bfe reduced data
    bfe_df = cat_df.copy()
    bfe_dp = DataPreprocessor(bfe_df)
    bfe_dp.drop_features(["SibSp", "Fare", "Pclass_3", "Embarked_S"])
   
    # Get ffi reduced data
    ffi_df = cat_df.copy()
    ffi_dp = DataPreprocessor(ffi_df)
    ffi_dp.drop_features(["Pclass_1", "Fare", "Pclass_2", "Embarked_Q"])

   
    # Get concatenated features and labels
    cat_features = cat_dp.get_features('Survived')
    cat_labels = cat_dp.get_labels('Survived')
    
    # Get bfe features and labels
    bfe_features = bfe_dp.get_features('Survived')
    bfe_labels = bfe_dp.get_labels('Survived')
    
    # Get ffi features and labels
    ffi_features = ffi_dp.get_features('Survived')
    ffi_labels = ffi_dp.get_labels('Survived')
    
    with open("models/model_full.pkl", "rb") as f:
        print("Loading full feature classifier")
        classifier = pickle.load(f)

        print("Training classifier")
        classifier.fit(cat_features, cat_labels)

        print("Accuracy: ", llocv(classifier, cat_features, cat_labels))

        print("Saving classifier")
        with open("models/model_full_trained.pkl", "wb") as g:
            pickle.dump(classifier, g)

        print("Done") 

    with open("models/model_bfe.pkl", "rb") as f:
        print("Loading bfe feature classifier")
        classifier = pickle.load(f)

        print("Training classifier")
        classifier.fit(bfe_features, bfe_labels)

        print("Accuracy: ", llocv(classifier, bfe_features, bfe_labels))

        print("Saving classifier")
        with open("models/model_bfe_trained.pkl", "wb") as g:
            pickle.dump(classifier, g)

        print("Done") 

    with open("models/model_ffi.pkl", "rb") as f:
        print("Loading ffi feature classifier")
        classifier = pickle.load(f)

        print("Training classifier")
        classifier.fit(ffi_features, ffi_labels)

        print("Accuracy: ", llocv(classifier, ffi_features, ffi_labels))

        print("Saving classifier")
        with open("models/model_ffi_trained.pkl", "wb") as g:
            pickle.dump(classifier, g)

        print("Done") 

