from agent import Agent
from constants import Constants
from frame_wrapper import FrameWrapper
import gymnasium as gym
import torch
from itertools import count
import json 
import os

# Initialize the environment
env = gym.make("ALE/Pacman-v5")
env = FrameWrapper(env)

device = (
  "cuda"
  if torch.cuda.is_available()
  else "mps"
  if torch.backends.mps.is_available()
  else "cpu"
)

# Initialize the agent
agent = Agent(env, device)

# Prepare to save reward data
data_to_add = {}
reward_data_filename = "reward_data.json"

if os.path.exists(reward_data_filename):
    with open(reward_data_filename, 'r') as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            # If the file is empty or invalid, start with an empty dict
            print("file is invalid")
            data = {}
else:
    data = {}

data_to_add = data

# Load model parameters, if present
if os.path.exists('models'):
  print('Loading previous models...')
  agent.load_models()
else:
  print('No previous models found.')
  os.mkdir('models')

# Reinforcement learning episode loop
for episode in range(Constants.NUM_EPISODES):
  print(f'Starting episode {episode}')

  with open(reward_data_filename, 'w') as file:
    json.dump(data, file, indent=4)

  state, _ = env.reset()
  state = torch.tensor(
    state,
    dtype=torch.float32,
    device=device
  ).unsqueeze(0)

  if episode != 0:
     data_to_add[episode] = sum(score)

  if episode % 10 == 0:
    print('Saving models...')
    agent.save_models()

  score = []
  for time_step in count():
    action = agent.predict(state)
    observation, reward, terminated, truncated, _ = env.step(action.item())

    # Add integer reward from this step to the reward list
    score.append(reward)

    reward_tensor = torch.tensor([reward], device=device)
    done = terminated or truncated

    if terminated:
      next_state = None
    else:
      next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    # Store the transition in memory
    agent.memory.append((state, action, next_state, reward_tensor))

    # Move to the next state
    state = next_state

    # Perform one step of the optimization (on the policy network)
    agent.optimize_model()

    # Partial update of target network
    target_net_state_dict = agent.target_network.state_dict()
    policy_net_state_dict = agent.policy_network.state_dict()
    for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * Constants.TAU + target_net_state_dict[key] * (1 - Constants.TAU)
    
    agent.target_network.load_state_dict(target_net_state_dict)

    if done:
      break
