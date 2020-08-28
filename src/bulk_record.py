#!/usr/bin/python3

import os
import re

model_folder = "../models/FlappyBird-v0/DQN/1587200629.8618479"

# Read all *.mdl files from the supplied dir
file_list = [os.path.join(model_folder, f) for f in os.listdir(model_folder) 
			if os.path.isfile(os.path.join(model_folder, f)) and f.endswith(".mdl")]

# Sort the string properly ascending by numbers (10000 comes AFTER 1000)
file_list.sort(key=lambda f: int(re.sub('\D', '', f)))

counter = 0

for file in file_list:
	os.system('py -3 main.py -e 1 -s 50000 -S 1337 -E "FlappyBird-v0" -f {} -a dqn -r -R'.format(file))
	os.rename('video.avi', 'video_{0:03d}.avi'.format(counter))
	counter += 1
