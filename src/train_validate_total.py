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
   
    # Get concatenated features and labels
    cat_features = cat_dp.get_features('Survived')
    cat_labels = cat_dp.get_labels('Survived')
    
    with open("model_full.pkl", "rb") as f:
        print("Loading full feature classifier")
        classifier = pickle.load(f)

        print("Training classifier")
        classifier.fit(cat_features, cat_labels)

        print("Accuracy: ", llocv(classifier, cat_features, cat_labels))

        print("Saving classifier")
        with open("model_full_trained.pkl", "wb") as g:
            pickle.dump(classifier, g)

        print("Done") 

    with open("model_bfe.pkl", "rb") as f:
        print("Loading bfe feature classifier")
        classifier = pickle.load(f)

        print("Training classifier")
        classifier.fit(cat_features, cat_labels)

        print("Accuracy: ", llocv(classifier, cat_features, cat_labels))

        print("Saving classifier")
        with open("model_bfe_trained.pkl", "wb") as g:
            pickle.dump(classifier, g)

        print("Done") 

    with open("model_ffi.pkl", "rb") as f:
        print("Loading ffi feature classifier")
        classifier = pickle.load(f)

        print("Training classifier")
        classifier.fit(cat_features, cat_labels)

        print("Accuracy: ", llocv(classifier, cat_features, cat_labels))

        print("Saving classifier")
        with open("model_ffi_trained.pkl", "wb") as g:
            pickle.dump(classifier, g)

        print("Done") 

