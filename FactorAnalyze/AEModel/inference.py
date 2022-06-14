import torch
from model import AE

class Inference():
    def __init__(self, path):
        self.model = self.load_model(path)

    def load_model(self, path):
        model = torch.jit.load(path)
        model.eval()
        return model

    def encode_data(self, data):
        encoded = self.model.encode(data)
        return encoded