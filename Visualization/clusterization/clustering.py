import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

max_kernels = 10

def draw_graph(data, range, label):
    plt.figure(figsize=(10, 8))
    plt.plot(range, data)
    plt.xticks(range)
    plt.xlabel('Number of Clusters')
    plt.ylabel(label)
    plt.grid(linestyle='--')
    plt.show()

def elbow_method(features: DataFrame, kmeans_kwargs: dict):
    sse = []
    for k in range(1, max_kernels + 1):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(features)
        sse.append(kmeans.inertia_)
    kl = KneeLocator(range(1, max_kernels + 1), sse, curve='convex', direction='decreasing')
    draw_graph(sse, range(1, max_kernels + 1), "Elbow method")
    
    return kl.elbow

def silhouette_method(features: DataFrame, kmeans_kwargs: dict):
    silhouette_coefficients = []
    for k in range(2, max_kernels + 1):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(features)
        score = silhouette_score(features, kmeans.labels_)
        silhouette_coefficients.append(score)

    draw_graph(silhouette_coefficients, range(2, max_kernels + 1), "Silhouette coefficients")

def gap_statistics(data):
    def optimalK(data, nrefs=3, maxClusters=15):
        gaps = np.zeros((len(range(1, maxClusters)),))
        resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
        for gap_index, k in enumerate(range(1, maxClusters)):
            refDisps = np.zeros(nrefs)
            for i in range(nrefs):
                randomReference = np.random.random_sample(size=data.shape)
            
                km = KMeans(k)
                km.fit(randomReference)
                
                refDisp = km.inertia_
                refDisps[i] = refDisp
            km = KMeans(k)
            km.fit(data)
            
            origDisp = km.inertia_
            gap = np.log(np.mean(refDisps)) - np.log(origDisp)
            gaps[gap_index] = gap
            
            resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)
        return (gaps.argmax() + 1, resultsdf)

    score_g, df = optimalK(data, nrefs=5, maxClusters=30)
    plt.plot(df['clusterCount'], df['gap'], linestyle='--', marker='o', color='b')
    plt.xlabel('K')
    plt.ylabel('Gap Statistic')
    plt.title('Gap Statistic vs. K')