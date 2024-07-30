from constants import Constants
from neural_net import NeuralNetwork
import torch
import torch.nn as nn
from collections import deque
import random

class Agent:
  def __init__(self, env, device):
    self.env = env
    obs_space_size = env.observation_space.shape[0]
    action_space_size = env.action_space.n

    self.device = device
    print(f"Using {self.device} device")

    self.memory = deque([], maxlen=Constants.MEMORY_CAPACITY)

    # Set up networks
    self.target_network = NeuralNetwork(obs_space_size, action_space_size).to(self.device)
    self.policy_network = NeuralNetwork(obs_space_size, action_space_size).to(self.device)

    # Copy target network to policy network to start
    self.policy_network.load_state_dict(self.target_network.state_dict())

    self.optimizer = torch.optim.AdamW(
      self.policy_network.parameters(),
      lr=Constants.LEARNING_RATE,
      amsgrad=True
    )

    self.epsilon = Constants.EPSILON_MAX

  def predict(self, state):
    self.epsilon = max(self.epsilon * Constants.EPSILON_DECAY, Constants.EPSILON_MIN)

    if random.random() < self.epsilon:
      # Perform random action
      return torch.tensor(
        [[self.env.action_space.sample()]],
        device=self.device,
        dtype=torch.long
      )
    else:
      # Perform best action using the policy network
      return self.policy_network(state).max(1).indices.view(1, 1)

  def optimize_model(self):
    if len(self.memory) < Constants.BATCH_SIZE:
        return
    
    transitions = random.sample(self.memory, Constants.BATCH_SIZE)

    q_values = []
    updated_values = []

    for state, action, next_state, reward in transitions:
      if next_state is None:
        continue
      q_s_a = self.policy_network(state).gather(1, action)
      current_value_fn = self.target_network(next_state).max(1).values
      updated_value_fn = reward + Constants.GAMMA * current_value_fn

      q_values.append(q_s_a)
      updated_values.append(updated_value_fn.unsqueeze(1))
      
    loss = nn.CrossEntropyLoss()(
      torch.tensor(q_values, device=self.device, requires_grad=True),
      torch.tensor(updated_values, device=self.device, requires_grad=True)
    )

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
    