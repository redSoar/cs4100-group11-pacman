import numpy as np
from collections import deque
import gymnasium as gym

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
