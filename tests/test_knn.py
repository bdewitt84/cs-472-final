import src.utils
from src.knn import KNearestNeighbors

def test_and():
    model = KNearestNeighbors(5, utils.always_one, utils.euclidian)
    X = [[0,0],[0,1],[1,0],[1,1]]
    Y = [0, 0, 0, 1]
    model.fit(X, Y)
    print(model.predict([0,0]))

