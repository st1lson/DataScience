# performing preprocessing part
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

sc = StandardScaler()

data = []

def PCA_cleaning(data):
    new_data = sc.fit_transform(data)

    pca = PCA(n_components=3)
    new_data = pca.fit_transform(new_data)
    return new_data
