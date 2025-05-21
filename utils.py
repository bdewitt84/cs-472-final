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
