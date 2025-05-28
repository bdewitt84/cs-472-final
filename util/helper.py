#./util/helper.py

"""Helper functions for cs-472-final"""


# Imports
from pathlib import Path


# Metadata
__author__ = "Brett DeWitt"
__date__ = "2025-05-20"


def safe_write(path, data):
    """
    Writes 'data' to file at 'path' in text mode.
    :raises FileExistsError: if 'path' exists
    """
    path_obj = Path(path)
    if path_obj.exists():
        raise FileExistsError("path must not point to existing file")
    with open(path_obj, 'w') as file:
        file.write(data)


def accuracy_score(y_pred, y_true):
    if len(y_pred) != len(y_true):
        raise ValueError("The y_pred and y_true have different lengths")

    correct = [a == b for a, b in zip(y_pred, y_true)]
    return sum(correct) / len(y_pred)