import pandas as pd
import torch
import numpy as np

from AEModel.inference import Inference
from PCA.PCA import PCA_cleaning

inf = Inference("/home/vlad/Projects/DataScience/FactorAnalyze/AEModel/AEMode.pt")
data = pd.read_csv('/home/vlad/Projects/DataScience/data/preprocessed.csv', sep=',', encoding='utf8').drop(labels='Unnamed: 0', axis=1)
numeric_data = data[['age', 'mil_rank', 'edu_lvl']]
torch_tensor = torch.tensor(data.drop(['age', 'mil_rank', 'edu_lvl'], axis=1).values.astype(np.float32))
encoded = inf.encode_data(torch_tensor)
data = inf.make_dataset(encoded, numeric_data)
print(data)

result = PCA_cleaning(data)
new_columns = [('Component ' + str(i)) for i in range(1, result.shape[1] + 1)]
data[new_columns] = result
print(data[new_columns])

data[new_columns].to_csv("encoded_data.csv")