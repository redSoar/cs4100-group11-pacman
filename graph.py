import os
import json
import matplotlib.pyplot as plt
import numpy as np

reward_data_filename = "reward_data.json"
xpoints = []
ypoints = []
max_score = 0
with open(reward_data_filename, 'r') as file:
    try:
        data = json.load(file)
    except json.JSONDecodeError:
        print("file is invalid")

for reward in data:
    if data[reward] > max_score:
        max_score = data[reward]
    xpoints.append(int(reward))
    ypoints.append(data[reward])

print(max_score)
window_size = 200
 
i = 0
moving_averages_xpoints = []
moving_averages_ypoints = []
 
while i < len(xpoints) - window_size + 1:
   
    windowx = xpoints[i : i + window_size]
    windowy = ypoints[i : i + window_size]
 
    window_averagex = round(sum(windowx) / window_size, 2)
    window_averagey = round(sum(windowy) / window_size, 2)
     
    moving_averages_xpoints.append(window_averagex)
    moving_averages_ypoints.append(window_averagey)
     
    i += 1
plt.plot(np.array(xpoints), np.array(ypoints), label = "Rewards")
plt.plot(np.array(moving_averages_xpoints), np.array(moving_averages_ypoints), label = "Moving Average")
plt.legend()
plt.show()

plt.plot(np.array(moving_averages_xpoints), np.array(moving_averages_ypoints), label = "Moving Average")
plt.legend()
plt.show()
