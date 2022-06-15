# performing preprocessing part
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

sc = StandardScaler()

def PCA_cleaning(data):
    new_data = sc.fit_transform(data)

    pca = PCA(n_components=3)
    new_data = pca.fit_transform(new_data)
    return new_data

data = pd.read_csv('/home/vlad/Projects/DataScience/data/preprocessed.csv', sep=',', encoding='utf8').drop(labels='Unnamed: 0', axis=1)
new_data = PCA_cleaning(data)
new_columns = [('Component ' + str(i)) for i in range(1, new_data.shape[1] + 1)]
data[new_columns] = new_data
print(data)