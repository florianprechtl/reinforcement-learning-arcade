import random
import numpy as np
import agents.agent as agent


class Agent(agent.IAgent):
	# A random action Agent
	
	NAME = "RND"
	
	def __init__(self, input_size, output_size, training_mode, is_conv, load_filename, seed):
		np.random.seed(seed)
		self.output_size = output_size
	
	def init_network(self):
		pass
	
	def save_model(self, filename):
		pass
	
	def copy_from(self, model):
		pass
	
	def get_action(self, state):
		return random.choice(np.arange(self.output_size))
	
	def update(self, state, action, reward, next_state, done):
		pass
