import numpy as np

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
    '''
    def __init__ (self, n_neighbors, weights, metric):
        self.n_neighbors = n_neighbors 
        self.data = None
        self.weights = weights
        self.metric = metric

    '''
    Fit the model to some training samples
    Arguments
        - X: the matrix of feature vectors
        - Y: the vector of class associated to the 
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
        
        # Calculate distances of all samples
        dist_sample = []
        for i, x in enumerate(self.data[0]): 
            dist = self.metric(z, x)
            dist_sample += [dist, i]

        # Put distance, sample pairs into numpy array
        np_dist_sample = np.array(dist_sample)
        
        # Sort the distance, sample pairs by distance
        sorted_np_dist_sample = np_dist_sample[np_dist_sample[:, 0].argsort()]
        
        # Get the nearest k neighbors
        k_nearest_neighbors = sorted_np_dist_sample[:self.n_neighbors, 1]
        print(k_nearest_neighbors)

        # 2. k-nearest neighbors vote on class of x
        #    Need: k-nearest neighbors, classes of neigbors,

        # 3. return predicted class of x

        return 0 # return 0 by default until implemented
