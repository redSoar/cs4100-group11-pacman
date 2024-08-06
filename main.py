from collections import deque

from agent import Agent
from constants import Constants
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from itertools import count
import os

class FrameWrapper(gym.Wrapper):
  def __init__(self, env, num_frames = 4):
    super(FrameWrapper, self).__init__(env)
    self.num_frames = num_frames
    self.frames = deque(maxlen = self.num_frames)
    self.average_frame = None

  def reset(self):
    observation, info = self.env.reset()
    for i in range(self.num_frames):
      self.frames.append(observation)
    self.average_frame = self.frame_averaging()
    return self.frame_averaging(), info

  def step(self, action):
    observation, reward, terminated, truncated, info = self.env.step(action)
    self.frames.append(observation)
    self.average_frame = self.frame_averaging()
    return self.frame_averaging(), reward, terminated, truncated, info

  def frame_averaging(self):
    average_frame = np.mean(np.stack(self.frames), axis=0).astype(np.uint8)
    return average_frame

def show_frame(frame):
  plt.figure(figsize=(8,8))
  plt.imshow(frame)
  plt.axis('off')
  plt.show()

# Initialize the environment
env = gym.make("ALE/Pacman-v5", render_mode="human")
env = FrameWrapper(env)
# env = gym.make("ALE/Pacman-v5")
device = (
  "cuda"
  if torch.cuda.is_available()
  else "mps"
  if torch.backends.mps.is_available()
  else "cpu"
)

# Initialize the agent
agent = Agent(env, device)

if os.path.exists('models'):
  print('Loading previous models...')
  agent.load_models()
else:
  print('No previous models found.')
  os.mkdir('models')

for episode in range(Constants.NUM_EPISODES):
  print(f'Starting episode {episode}')

  state, _ = env.reset()
  state = torch.tensor(
    state,
    dtype=torch.float32,
    device=device
  ).unsqueeze(0)

  if episode % 10 == 0:
    print('Saving models...')
    agent.save_models()

  for time_step in count():
    action = agent.predict(state)
    observation, reward, terminated, truncated, _ = env.step(action.item())
    reward = torch.tensor([reward], device=device)
    done = terminated or truncated

    if terminated:
      next_state = None
    else:
      next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    # Store the transition in memory
    agent.memory.append((state, action, next_state, reward))

    # Move to the next state
    state = next_state

    # Perform one step of the optimization (on the policy network)
    agent.optimize_model()

    # Soft update of the target network's weights
    # θ′ ← τ θ + (1 −τ )θ′
    target_net_state_dict = agent.target_network.state_dict()
    policy_net_state_dict = agent.policy_network.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key]*Constants.TAU + target_net_state_dict[key]*(1-Constants.TAU)
    agent.target_network.load_state_dict(target_net_state_dict)

    if done:
      break

  show_frame(env.average_frame)