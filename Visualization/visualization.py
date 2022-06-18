import pandas as pd
import plotly.express as px
from sklearn.cluster import dbscan
from clusterization.clustersNumber import elbow_method
from clusterization.clustersNumber import silhouette_method
from clusterization.clustersNumber import gap_statistics
from clusterization.clustering import kmeans_method
from clusterization.clustersNumber import draw_dendogram
from clusterization.clustering import DBSCAN_method
from clusterization.clustering import optics_method
from clusterization.clustering import meanshift_method
from clusterization.clustering import gaussian_mixture
from PCA.PCA import PCA_cleaning

data = pd.read_csv('data/encoded_data.csv', encoding='utf-8')
kmeans_kwargs = {
    'init': 'random',
    'n_clusters': 4,
    'n_init': 10,
    'max_iter': 300,
    'random_state': 42,
}


# data = pd.read_csv('/home/vlad/Projects/DataScience/data/preprocessed.csv', sep=',', encoding='utf8').drop(labels='Unnamed: 0', axis=1)
# result = PCA_cleaning(data)
# new_columns = [('Component ' + str(i)) for i in range(1, result.shape[1] + 1)]
# data[new_columns] = result

clusters = optics_method(data)
#y_pred = clusters.predict(data)

#data clustering


def visualize(data):
    fig = px.scatter_3d(
        data, color=clusters.labels_[data.index], x='Component 1', y='Component 2', z='Component 3',
    )
    fig.update_traces(marker=dict(size=1))
    fig.show()

visualize(data)

#print(clusters.labels_)