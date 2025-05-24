'''
Implements the K-Nearest Neighbors Model for Machine Learning
Author: Makani Buckley
Date Added: May 21, 2025
'''

import numpy as np
from pandas import DataFrame

class KNearestNeighbors:

    '''
    Initializes the K-Nearest-Neighbor Model
    Arguments:
        - n_neighbors: the number of nearest-neighbors to predict from
        - weights: a function which accepts an array of distances, and
          returns an array of the same shape containing weights for voting
        - metric: a function taking some samples a and b,
          represented by feature vectors, and returning a
          scalar distance between them
        - weights_parametesrs: additional parameters to pass to the weights
          function
        - metric_parameters: additional parameters to pass to the metric
          function
    '''
    def __init__ (self, n_neighbors, weights, metric, weights_parameters={},
        metric_parameters={}):
        self.n_neighbors = n_neighbors 
        self.data = None
        self.weights = weights
        self.metric = metric
        self.weights_parameters = weights_parameters
        self.metric_parameters = metric_parameters

    '''
    Fit the model to some training samples
    Arguments
        - X: the data frame of feature vectors
        - Y: the data frame of class associated to the 
          feature vectors of X
    '''
    def fit (self, X, Y):
        self.data = (X, Y) 

    '''
    Predicts the class of a sample
    Arguments:
        - z: the feature vector [x1, ..., xn] of the
          sample to predict
    '''
    def predict (self, z):
        # 1. iterate over all samples to find k-nearest neighbors
        #    Need: k, feature vectors of samples, distance function

        # If data is None throw an error
        if self.data is None:
            raise ValueError('Training data is not initialized')

        # Get features as a numpy array
        features = self.data[0].to_numpy()
        print(features)
        
        # Calculate distances of all samples
        dist_sample = []
        for i, x in enumerate(features): 
            dist = self.metric(z, x, **self.metric_parameters)
            dist_sample += [[dist, i]]

        # Put distance, sample pairs into numpy array
        np_dist_sample = np.array(dist_sample)
        
        # Sort the distance, sample pairs by distance
        sorted_np_dist_sample = np_dist_sample[np_dist_sample[:, 0].argsort()]
        
        # Get the nearest k neighbors as indices
        k_nearest_neighbors = sorted_np_dist_sample[:self.n_neighbors, 1]

        # 2. k-nearest neighbors vote on class of x
        #    Need: k-nearest neighbors, classes of neigbors,

        # Get classes as a numpy array
        classes = self.data[1].to_numpy().flatten()
        print(classes)

        # Get neigbor classes 
        k_classes = np.array([classes[int(i)] for i in k_nearest_neighbors])

        # Get class categories and set up a poll for counting votes
        class_poll = {x: 0 for x in sorted(set(k_classes))}

        # Get neighbor weights
        k_weights = self.weights(
            sorted_np_dist_sample[:self.n_neighbors, 0],
            **self.weights_parameters
        )

        # K Neighbors go to the polls
        for w, y in zip(k_weights, k_classes):
            class_poll[y] += w

        # Get the class with maximum vote
        pred = None
        max = float('-inf')
        for c in class_poll:
            vote = class_poll[c]
            if max < vote:
                pred = c
                max = vote 

        # 3. return predicted class of x

        return pred
