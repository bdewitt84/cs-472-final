from src.utils import always_one, euclidian
from src.knn import KNearestNeighbors

def test_and():
    model = KNearestNeighbors(1, always_one, euclidian)
    X = [[0,0],[0,1],[1,0],[1,1]]
    Y = [0, 0, 0, 1]
    model.fit(X, Y)
    assert model.predict([0,0]) == 0;
    assert model.predict([0,1]) == 0;
    assert model.predict([1,0]) == 0;
    assert model.predict([1,1]) == 1;

def test_or():
    model = KNearestNeighbors(1, always_one, euclidian)
    X = [[0,0],[0,1],[1,0],[1,1]]
    Y = [0, 1, 1, 1]
    model.fit(X, Y)
    assert model.predict([0,0]) == 0;
    assert model.predict([0,1]) == 1;
    assert model.predict([1,0]) == 1;
    assert model.predict([1,1]) == 1;

def test_xor():
    model = KNearestNeighbors(1, always_one, euclidian)
    X = [[0,0],[0,1],[1,0],[1,1]]
    Y = [0, 1, 1, 0]
    model.fit(X, Y)
    assert model.predict([0,0]) == 0;
    assert model.predict([0,1]) == 1;
    assert model.predict([1,0]) == 1;
    assert model.predict([1,1]) == 0;

