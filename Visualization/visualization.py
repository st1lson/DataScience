import pandas as pd
from clusterization.clustering import elbow_method
from clusterization.clustering import silhouette_method
from clusterization.clustering import gap_statistics

data = pd.read_csv('data/preprocessed.csv', encoding='utf-8')
kmeans_kwargs = {
    'init': 'random',
    'n_init': 10,
    'max_iter': 300,
    'random_state': 42,
}

elbow_method(data, kmeans_kwargs)
silhouette_method(data, kmeans_kwargs)
gap_statistics(data)