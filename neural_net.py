import torch.nn as nn

class NeuralNetwork(nn.Module):
  def __init__(self, action_space_size):
    super().__init__()
    self.flatten = nn.Flatten()
    input_height=160
    input_width=250
    self.layers = nn.Sequential(
      nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(3, 3),
      nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(3, 3),
      nn.Flatten(),
      nn.Linear(32 * (input_height//9) * (input_width//9), 128),
      nn.ReLU(),
      nn.Linear(128, 128),
      nn.ReLU(),
      nn.Linear(128, action_space_size),
    )

  def forward(self, x):
    x = x.transpose(1,3)
    logits = self.layers(x)
    return logits
