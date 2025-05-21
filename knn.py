import numpy as np

class KNearestNeighbor:

    '''
    Initializes the K-Nearest-Neighbor Model
    Arguments:
        - k: the number of nearest-neighbors to predict from
        - data: a list of samples with form [x1, ..., xn, y]
          where x1, ..., xn are features and y is the class
        - distance: a function taking some samples a and b,
          represented by feature vectors, and returning a
          scalar distance between them
    def __init__ (self, k, data, distance, vote):
        self.k = k
        self.data = data
        self.distance = distance

    '''
    Predicts the class of a sample
    Arguments:
        - x: the feature vector [x1, ..., xn] of the
          sample to predict
    def predict (self, x):
        # 1. iterate over all samples to find k-nearest neighbors
        #    Need: k, feature vectors of samples, distance function
        # 2. k-nearest neighbors vote on class of x
        #    Need: k-nearest neighbors, classes of neigbors,
        # 3. return predicted class of x
