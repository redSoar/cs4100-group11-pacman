# PacmanDQN

![trained](https://github.com/user-attachments/assets/aee840f6-590e-4aa0-b67b-7879a2ad553d)

## Overview

We have designed and trained a Deep Q Neural Network using Pytorch to play OpenAI Gymnasium's Atari Pacman environment.

We aimed to explore the complexity of emergent behaviors by implementing and combining various AI techniques, such as convolutional neural networks, deep learning, and reinforcement learning.

## Requirements

Python 3.11 specifically (3.12 will NOT work with the Atari environment for unknown reasons)

All required packages are listed in the requirements.txt file

## How to Run

```sh
python3.11  main.py
```

The variable `train` can be set to `True` to train the model, or `False` to view its performance. Model parameters are stored in the `model/` directory.

Note: Do not uncomment the call to show_frame() at the end of main.py unless you wish to generate a PNG visualizing how we processed 4 frames worth of pixels.

## Performance

The model showed signs of improvement over the course of 4000 episodes, as shown in the videos below.

Episode 0:

https://github.com/user-attachments/assets/eae4b3ca-30d0-45f5-8cb0-895f982bb146

Episode 4173

https://github.com/user-attachments/assets/8caabc24-20a0-4b8a-a6a3-b52737ad34ed
