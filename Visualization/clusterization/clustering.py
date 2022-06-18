from sklearn.cluster import KMeans
import numpy as np
from sklearn.cluster import Birch
from numpy import unique
from numpy import where
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import OPTICS
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def kmeans_method(features, kmeans_kwargs):
    return KMeans(**kmeans_kwargs).fit(features)

def gaussian_mixture(data, clustersNumber):
    return GaussianMixture(n_components=clustersNumber).fit(data)

def DBSCAN_find_eps(data):
    nn_model = NearestNeighbors(n_neighbors=2)
    nn_model.fit(data)
    distances, indices = nn_model.kneighbors(data)
    distances = np.sort(distances, axis=0)
    distances = distances[:,1]
    plt.plot(distances)
    plt.show()

def DBSCAN_method(data, eps):
    return DBSCAN(eps=eps, min_samples=6).fit(data)

def optics_method(data):
    return OPTICS(min_samples=10).fit(data)


def meanshift_method(data):
    bandwidth = estimate_bandwidth(data, quantile=0.5)
    return MeanShift(bandwidth=bandwidth).fit(data)