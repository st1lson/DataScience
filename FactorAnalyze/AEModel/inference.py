from base64 import encode
from dataclasses import make_dataclass
import torch
import pandas as pd
import numpy as np

from AEModel.model import AE

class Inference():
    def __init__(self, path):
        self.model = self.load_model(path)

    def load_model(self, path):
        model = torch.jit.load(path)
        model.eval()
        return model

    def encode_data(self, data):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        data = data.to(device)
        self.model = self.model.to(device)
        encoded = self.model.encoder(data)
        return encoded

    def make_dataset(self, encoded, numeric_data):
        with torch.no_grad():
            encoded = pd.DataFrame(encoded.cpu().numpy())
        return pd.merge(encoded, numeric_data, left_index=True, right_index=True)
