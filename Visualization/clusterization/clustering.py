import sklearn
from sklearn.cluster import KMeans
from numpy import unique
from numpy import where
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import OPTICS
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth

def kmeans_method(features, kmeans_kwargs):
    return KMeans(**kmeans_kwargs).fit(features)

def gaussian_mixture(data, clustersNumber):
    return GaussianMixture(n_components=clustersNumber).fit(data)

def DBSCAN_method(data):
    return DBSCAN(eps=5.85, min_samples=6).fit(data)

def optics_method(data):
    return OPTICS(min_samples=10).fit(data)

def meanshift_method(data):
    bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=500)
    return MeanShift(bandwidth=bandwidth).fit(data)