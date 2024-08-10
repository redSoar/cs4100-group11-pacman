from constants import Constants
from neural_net import NeuralNetwork
import torch
import torch.nn as nn
from collections import deque
import random

class Agent:
  def __init__(self, env, device, train):
    self.env = env
    action_space_size = env.action_space.n

    self.device = device
    print(f"Using {self.device} device")

    self.train = train

    self.memory = deque([], maxlen=Constants.MEMORY_CAPACITY)

    # Set up networks
    self.target_network = NeuralNetwork(action_space_size).to(self.device)
    self.policy_network = NeuralNetwork(action_space_size).to(self.device)

    # Copy target network to policy network to start
    self.policy_network.load_state_dict(self.target_network.state_dict())

    self.optimizer = torch.optim.AdamW(
      self.policy_network.parameters(),
      lr=Constants.LEARNING_RATE,
      amsgrad=True
    )

    self.epsilon = Constants.EPSILON_MAX

  def predict(self, state):
    if self.train and random.random() < self.epsilon:
      # Perform random action
      return torch.tensor(
        [[self.env.action_space.sample()]],
        device=self.device,
        dtype=torch.long
      )
    else:
      # Perform best action using the policy network
      return self.policy_network(state).max(1).indices.view(1, 1)
    
  def update_parameters(self):
    self.epsilon = max(self.epsilon * Constants.EPSILON_DECAY, Constants.EPSILON_MIN)
    # Partial update of target network
    target_net_state_dict = self.target_network.state_dict()
    policy_net_state_dict = self.policy_network.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * Constants.TAU + target_net_state_dict[key] * (1 - Constants.TAU)
    self.target_network.load_state_dict(target_net_state_dict)

  def optimize_model(self):
    if not self.train or len(self.memory) < Constants.BATCH_SIZE:
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

    # Keep track of which states result in a terminal state.
    # These should not contribute to the V_opt used in the update of the Q value
    non_terminal_next_state_indices = torch.tensor(tuple([sprime is not None for sprime in next_states]), device=self.device, dtype=torch.bool)

    next_state_values = torch.zeros(Constants.BATCH_SIZE, device=self.device)
    with torch.no_grad():
        next_state_values[non_terminal_next_state_indices] = self.target_network(next_states_tensor).max(1).values
    
    # Pass states to the network together, to generate gradient values
    q_values = self.policy_network(states_tensor).gather(1, actions_tensor)

    # Update Q value
    updated_v_values = rewards_tensor + Constants.GAMMA * next_state_values

    criterion = nn.SmoothL1Loss()
    loss = criterion(q_values, updated_v_values.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()

    # Clip gradients
    torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 10)
    self.optimizer.step()

  def load_models(self):
    self.policy_network = torch.load('models/policy.pth')
    self.target_network = torch.load('models/target.pth')

  def save_models(self):
    torch.save(self.policy_network, 'models/policy.pth')
    torch.save(self.target_network, 'models/target.pth')
