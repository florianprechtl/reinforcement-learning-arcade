import re
import sys
import numpy as np
import collections
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
	print("Awaiting filename of log file as first parameter.")
	exit()

p = re.compile('([+-]?([0-9]*[.])?[0-9]+)')
f = open(sys.argv[1], 'r')

total_steps = []
average_steps = []
total_rewards = []

last_steps = collections.deque(maxlen=10000)

for line in f:
	if line.startswith('* A'):
		(steps, _), (reward, _), (steps_p_s, _) = p.findall(line)
		total_steps.append(float(steps))
		total_rewards.append(float(reward))
		last_steps.append(float(steps))
		average_steps.append(np.mean(last_steps).item())

f.close()

plt.clf()
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.plot(total_steps)
plt.plot(average_steps)
plt.show()

plt.close()
plt.clf()
plt.xlabel("Episode")
plt.ylabel("Rewards")
plt.plot(total_rewards)
plt.show()
