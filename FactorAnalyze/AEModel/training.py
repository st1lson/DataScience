import torch
from model import AE
import pandas as pd
import numpy as np
from torchsummary import summary

# Model Initialization
model = AE(121)
data = pd.read_csv('/home/vlad/Projects/DataScience/data/preprocessed.csv', sep=',', encoding='utf8').drop(labels='Unnamed: 0', axis=1)
torch_tensor = torch.tensor(data.values.astype(np.float32))

# print(data)
# print(torch_tensor)

def train(model, data, epochs = 10):
# Validation using MSE Loss function
  loss_function = torch.nn.MSELoss(size_average=False)
    
  # Using an Adam Optimizer with lr = 0.1
  optimizer = torch.optim.Adam(model.parameters(),
                              lr = 1e-1)

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = model.to(device)
  data = data.to(device)
  for epoch in range(epochs):
    sum = 0
    for (i, soldier) in enumerate(data):
      optimizer.zero_grad()  
      # Output of Autoencoder
      reconstructed = model(soldier)
      # Calculating the loss function
      loss = loss_function(soldier, reconstructed)
      sum += loss.item()
      # The gradients are set to zero,
      # the the gradient is computed and stored.
      # .step() performs parameter update     
      if(i == 100):
        print(loss.item())
        print(soldier)
        print(torch.round(reconstructed))
        print(reconstructed)
      loss.backward()
      optimizer.step()
    print('Epoch', epoch, '-', sum / data.shape[0])
  return model

def save_model(model, path):
    script_module = torch.jit.script(model)
    script_module.save(path)

model = train(model, torch_tensor)
save_model(model, 'AEMode.pt')