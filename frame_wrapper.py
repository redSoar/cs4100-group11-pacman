import numpy as np
from collections import deque
import gymnasium as gym

# wrapper class for editing the observation returned by the environment
class FrameWrapper(gym.Wrapper):
  # initializes the number of frames and creates a deque for storing the current frames
  def __init__(self, env, num_frames = 4):
    super(FrameWrapper, self).__init__(env)
    self.num_frames = num_frames
    self.frames = deque(maxlen = self.num_frames)
    self.average_frame = None

  # takes the observation from reset() and uses it 4 times to allow
  # subsequent step() calls to average 4 frames
  def reset(self):
    observation, info = self.env.reset()
    for i in range(self.num_frames):
      self.frames.append(observation)
    self.average_frame = self.frame_averaging()
    return self.frame_averaging(), info

  # takes the observation from step(), uses it for the average and returns the new observation
  def step(self, action):
    observation, reward, terminated, truncated, info = self.env.step(action)
    self.frames.append(observation)
    self.average_frame = self.frame_averaging()
    return self.frame_averaging(), reward, terminated, truncated, info

  # takes the pixel values of the 4 frames in the deque and averages the values
  def frame_averaging(self):
    average_frame = np.mean(np.stack(self.frames), axis=0).astype(np.uint8)
    return average_frame
