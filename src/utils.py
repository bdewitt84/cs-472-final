import numpy as np

'''
Miscellaneous utility functions and classes supporting the project.
Author: Makani Buckley
Date Added: May 21, 2025
'''

from math import sqrt

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
