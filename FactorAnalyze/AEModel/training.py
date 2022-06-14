import torch
from model import AE


# Model Initialization
model = AE()
data = [] #get data from somewhere

def train(model, data):
# Validation using MSE Loss function
  loss_function = torch.nn.MSELoss()
    
  # Using an Adam Optimizer with lr = 0.1
  optimizer = torch.optim.Adam(model.parameters(),
                              lr = 1e-1,
                              weight_decay = 1e-8)


  epochs = 20
  for epoch in range(epochs):
      for soldier in data:
          
        # Output of Autoencoder
        reconstructed = model(soldier)
          
        # Calculating the loss function
        loss = loss_function(reconstructed, soldier)
          
        # The gradients are set to zero,
        # the the gradient is computed and stored.
        # .step() performs parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

  return model

def save_model(model, path):
    script_module = torch.jit.script(model)
    script_module.save(path)

model = train(model, data)
save_model(model, 'AEMode.pt')