#./skl/total_train_perceptron.py

"""Sci-kit Learn Perceptron model used for reference and comparison in cs-472-final"""


# Imports
from sklearn.linear_model import Perceptron
import pandas as pd
from preprocessor.data_preprocessor import DataPreprocessor
import numpy as np
import pickle

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
    
    with open("perceptron_untrained.pkl", "rb") as f:
        print("Loading classifier")
        classifier = pickle.load(f)
        print("Training classifier")
        classifier.fit(cat_features, cat_labels)
        print("Accuracy: ", classifier.score(cat_features, cat_labels))
        print("Saving classifier")
        with open("perceptron_trained.pkl", "wb") as g:
            pickle.dump(classifier, g)
        print("Done") 
