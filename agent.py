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

    states = []
    actions = []
    next_states = []
    rewards = []

    for state, action, next_state, reward in transitions:
      states.append(state)
      actions.append(action)
      next_states.append(next_state)
      rewards.append(reward)

    states_tensor = torch.cat(states)
    actions_tensor = torch.cat(actions)
    next_states_tensor = torch.cat([sprime for sprime in next_states if sprime is not None])
    rewards_tensor = torch.cat(rewards)

    non_terminal_next_state_indices = torch.tensor(tuple([sprime is not None for sprime in next_states]), device=self.device, dtype=torch.bool)

    next_state_values = torch.zeros(Constants.BATCH_SIZE, device=self.device)
    with torch.no_grad():
        next_state_values[non_terminal_next_state_indices] = self.target_network(next_states_tensor).max(1).values
    
    q_values = self.policy_network(states_tensor).gather(1, actions_tensor)
    updated_v_values = rewards_tensor + Constants.GAMMA * next_state_values

    criterion = nn.SmoothL1Loss()
    loss = criterion(q_values, updated_v_values.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100)
    self.optimizer.step()
