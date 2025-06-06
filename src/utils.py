import numpy as np

'''
Miscellaneous utility functions and classes supporting the project.
Author: Makani Buckley
Date Added: May 21, 2025
'''

from math import sqrt, exp, pow

'''
Compute the euclidian distance between two vectors
Arguments
    - a, b: two vectors of length n each
'''
def euclidian(a, b):
    if len(a) != len(b):
        raise ValueException("Vectors are of different size")

    sum = 0
    for ai, bi in zip(a, b):
        sum += (ai - bi) * (ai - bi) # compute (ai - bi) ^ 2

    return sqrt(sum)

'''
Compute the minkowski distance between two vectors
Arguments
    - a, b: two vectors of length n each
    - p: the parameter for minkowski distance to exponentiate by
'''

def minkowski(a, b, p):
    if len(a) != len(b):
        raise ValueException("Vectors are of different size")

    sum = 0
    for ai, bi in zip(a, b):
        sum += pow(abs(ai - bi), p) # compute (ai - bi) ^ p

    return pow(sum, 1/p)

'''
Take a list and returns a list of 1's of the same length
Arguments
   - a: a list
'''
def always_one(a):
    return np.ones(len(a))

'''
Apply the element wise inverse on a list
Arguments:
    - X: a list of numbers
'''
def inverse(X):
    return [(float('inf') if x==0 else 1/x) for x in X]

'''
Apply the element wise negative on a list
Arguments:
    - X: a list of numbers
'''
def negative(X):
    return [-x for x in X]

'''
A function in the shape of a hump applied element wise to a list
Arguments:
    - X: a list of numbers
'''
def hump(X):
    return [exp(-(x*x)) for x in X]

'''
Exponential decay applied element wise to a list
Arguments:
    - X: a list of numbers
'''
def decay(X):
    return [exp(-x) for x in X]

'''
Negative applied element wise to a list but normalized to squash output between
[0, 1]
Arguments
    - X: a list of numbers
'''
def normalized_negative(X):
    _max = max(X)
    if _max == 0:
        return [0] * len(X)
    return [-x/_max + 1 for x in X]

'''
Measure accuracy as a porportion of correct predictions
Arguments
    - y_pred: the classes predicted by the model for the samples
    - y_true: the true classes of the samples
'''
def accuracy_score(y_pred, y_true):
    if len(y_pred) != len(y_true):
        raise ValueException("The y_pred and y_true have different lengths")

    correct = [a == b for a, b in zip(y_pred, y_true)]
    return sum(correct) / len(y_pred)
