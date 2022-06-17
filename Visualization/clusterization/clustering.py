from sklearn.cluster import KMeans
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import Birch

def kmeans_method(features, kmeans_kwargs):
    return KMeans(**kmeans_kwargs).fit(features)

def birch_method(data, clustersNumber):
    return Birch(n_clusters=clustersNumber).fit(data)