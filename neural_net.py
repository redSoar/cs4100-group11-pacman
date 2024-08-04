import torch.nn as nn

class NeuralNetwork(nn.Module):
  def __init__(self, action_space_size):
    super().__init__()
    self.flatten = nn.Flatten()
    self.layers = nn.Sequential(
      nn.Conv2d(3, 3, 3),
      nn.ReLU(),
      nn.MaxPool2d(3, 3),
      nn.Conv2d(3, 3, 3),
      nn.ReLU(),
      nn.MaxPool2d(3, 3),
      nn.Flatten(),
      nn.Linear(1248, 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, action_space_size),
      nn.ReLU(),
      nn.Softmax(dim=0),
    )

  def forward(self, x):
    x = x.transpose(1,3)
    logits = self.layers(x)
    return logits
