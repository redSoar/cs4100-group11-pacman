from agent import Agent
from constants import Constants
import gymnasium as gym
import torch
from itertools import count

# Initialize the environment
env = gym.make("CartPole-v1", render_mode="human")
device = (
  "cuda"
  if torch.cuda.is_available()
  else "mps"
  if torch.backends.mps.is_available()
  else "cpu"
)

# Initialize the agent
agent = Agent(env, device)

for episode in range(Constants.NUM_EPISODES):
  print(f'Starting episode {episode}')

  state, _ = env.reset()
  state = torch.tensor(
    state,
    dtype=torch.float32,
    device=device
  ).unsqueeze(0)

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
