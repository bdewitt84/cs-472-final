from setuptools import setup, find_packages

setup(
    name='cs_472_final',
    version='0.1',
    description='A python package for our CS472 final project',
    author='Makani Buckley',
    author_email='makanicb25@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
    ],
)
