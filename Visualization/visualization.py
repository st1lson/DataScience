import pandas as pd
import plotly.express as px
from clusterization.clustering import spectral_clustering
from clusterization.clustersNumber import elbow_method
from clusterization.clustersNumber import silhouette_method
from clusterization.clustersNumber import gap_statistics
from clusterization.clustering import kmeans_method
from clusterization.clustersNumber import draw_dendogram
from clusterization.clustering import DBSCAN_method
from clusterization.clustering import optics_method
from clusterization.clustering import meanshift_method

data = pd.read_csv('data/encoded_data.csv', encoding='utf-8')
kmeans_kwargs = {
    'init': 'random',
    'n_clusters': 4,
    'n_init': 10,
    'max_iter': 300,
    'random_state': 42,
}

meanshift_method(data)