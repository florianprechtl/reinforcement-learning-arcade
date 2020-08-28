#!/usr/bin/python3
#
# https://www.toptal.com/deep-learning/pytorch-reinforcement-learning-tutorial

import gym
import time
import numpy as np
import collections
from dqn import DQN


# Const parameters
SEED = 1337	# int(time.time())
STEPS = 200
EPISODES = 300
TRAINING = False
RENDERING = True

env = gym.make('CartPole-v1')
env.seed(SEED)
agent = DQN(env.observation_space.shape[0], env.action_space.n, TRAINING, SEED)

if not TRAINING:
	agent.load_model("models/0.001_0.95_1.0_0.01_0.995.mdl")

"""
Description:
	A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts upright, and the goal is to prevent it from falling over by increasing and reducing the cart's velocity.

Source:
	This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson
	
Observation: 
	Type: Box(4)
	Num	Observation                 Min         Max
	0	Cart Position             -4.8            4.8
	1	Cart Velocity             -Inf            Inf
	2	Pole Angle                 -24 deg        24 deg
	3	Pole Velocity At Tip      -Inf            Inf

Actions:
	Type: Discrete(2)
	Num	Action
	0	Push cart to the left
	1	Push cart to the right

	Note: The amount the velocity that is reduced or increased is not fixed; it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

Reward:
	Reward is 1 for every step taken, including the termination step
	
Starting State:
	All observations are assigned a uniform random value in [-0.05..0.05]
	
Episode Termination:
	Pole Angle is more than 12 degrees
	Cart Position is more than 2.4 (center of the cart reaches the edge of the display)
	Episode length is greater than 200
	
Solved Requirements:
	Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
"""

# Average buffers for logging
rewards = collections.deque(maxlen=100)
steps = collections.deque(maxlen=100)

for episode in range(EPISODES):
	avg_reward = 0
	state = env.reset()
	
	for step in range(STEPS):
		if RENDERING:
			env.render()
		
		# Feed the network and get back an appropriate action
		action = agent.get_action(state)
		
		# Pass the action to the environment
		next_state, reward, done, info = env.step(action)
		
		# Custom reward for better training
		#reward = 4.7 - abs(next_state[0])
		#reward = 3 - abs(next_state[1])
		#reward = 4.7 - abs(next_state[2])
		#reward = 3 - abs(next_state[3])
		
		if TRAINING:
			# Train the network
			agent.update(state, action, reward, next_state, done)
		
		state = next_state
		
		# object next_state: 	environment dependant state of the game, for CartPole it is [Cart Position, Cart Velocity, Pole Angle, Pole Velocity At Tip]
		# float reward:			amount of reward achieved by previous action, not normalized, goal: increase
		# boolean done:			indicates whether the environment has to be reset because the game ended (invalid state, lost last life, ...)
		# dict info:			used for debugging purpose, not allowed for final network to be used
		
		avg_reward += reward
		
		# End the episode early when we are done
		if done:
			break
	
	rewards.append(avg_reward / (step + 1))
	steps.append(step + 1)
	
	print("Episode {} finished after {} steps, avg. reward: {:.2f}, avg. steps: {:.2f}".format(episode + 1, step + 1, np.mean(rewards), np.mean(steps)))

if TRAINING:
	agent.save_model()

env.close()
