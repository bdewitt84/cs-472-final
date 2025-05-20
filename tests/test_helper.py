#./test/test_helper.py

r"""Unit tests for ./util/helper.py"""


# Imports
import os
from pytest import raises
from tempfile import TemporaryDirectory

from util.helper import safe_write


# Metadata
__author__ = "Brett DeWitt"
__date__ = "2025-05-20"


def test_safe_write_valid():
    temp_dir = TemporaryDirectory()
    path = temp_dir.name + '/test_safe_write_valid.tmp'
    norm_path = os.path.normpath(path)
    data = 'dummy data'

    safe_write(norm_path, data)

    # File exists
    assert os.path.exists(norm_path)

    # Correct data was written
    with open(norm_path, 'r') as f:
       assert f.read() == data

    # Cleanup
    temp_dir.cleanup()


def test_safe_write_invalid():
    temp_dir = TemporaryDirectory()
    path = temp_dir.name + '/test_safe_write_invalid.tmp'
    norm_path = os.path.normpath(path)
    data = 'dummy data'
    with open(norm_path, 'w') as f:
        f.write(data)

    with raises(FileExistsError) as e:
        # Exception was raised when trying to write to existing file
        safe_write(norm_path, data)

    # Cleanup
    temp_dir.cleanup()
