'''
Unit Tests for our implementation of K Nearest Neigbhor
Author: Makani Buckley
Date: May 24, 2025
'''
from src.utils import always_one, euclidian
from src.knn import KNearestNeighbors
from pandas import DataFrame

def test_and():
    model = KNearestNeighbors(1, always_one, euclidian)
    X = DataFrame({"A": [0,0,1,1], "B": [0,1,0,1]})
    Y = DataFrame({"C": [0,0,0,1]})
    model.fit(X, Y)
    assert all(a == b for a, b in zip([0,0,0,1], model.predict(X)));

def test_or():
    model = KNearestNeighbors(1, always_one, euclidian)
    X = DataFrame({"A": [0,0,1,1], "B": [0,1,0,1]})
    Y = DataFrame({"C": [0,1,1,1]})
    model.fit(X, Y)
    assert all(a == b for a, b in zip([0,1,1,1], model.predict(X)));

def test_xor():
    model = KNearestNeighbors(1, always_one, euclidian)
    X = DataFrame({"A": [0,0,1,1], "B": [0,1,0,1]})
    Y = DataFrame({"C": [0,1,1,0]})
    model.fit(X, Y)
    assert all(a == b for a, b in zip([0,1,1,0], model.predict(X)));

