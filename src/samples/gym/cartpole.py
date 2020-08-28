#!/usr/bin/python3

import gym

# Print all available environments
#print(gym.envs.registry.all())

env = gym.make('CartPole-v1')

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
	Solved Requirements
	Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
"""



# Fixed non-negative integers
print(env.action_space)

# n-dimensional box with upper and lower bounds per dimension
print(env.observation_space)


# Train exactly 20 episodes
for episode in range(20):
	observation = env.reset()
	
	# Do a maximum of 100 steps for each episode
	for step in range(100):
		env.render()
		
		# Here we should actually feed the network and get back an appropriate action
		print(observation)
		action = env.action_space.sample()
		
		# Pass the action to the environment
		observation, reward, done, info = env.step(action)
		
		# object observation: 	environment dependant state of the game, for CartPole it is [Cart Position, Cart Velocity, Pole Angle, Pole Velocity At Tip]
		# float reward:			amount of reward achieved by previous action, not normalized, goal: increase
		# boolean done:			indicates whether the environment has to be reset because the game ended (invalid state, lost last life, ...)
		# dict info:			used for debugging purpose, not allowed for final network to be used
		
		# End the episode early when we are done
		if done:
			break
	
	print("Episode {} finished after {} steps\n".format(episode + 1, step + 1))

env.close()