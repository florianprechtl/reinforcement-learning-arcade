import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import torchvision.transforms as T
from replayBuffer import ReplayBuffer

class NN(nn.Module):
	# A simple neural network
	
	def __init__(self, input_size, output_size, seed):
		super(NN, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.seed = torch.manual_seed(seed)
		self.fc_in = nn.Linear(input_size, 64)		# input layer -> hidden layer 1 (64)
		self.fc_h1 = nn.Linear(64, 32)				# hidden layer 1 (64) -> hidden layer 2 (32)
		self.fc_out = nn.Linear(32, output_size)	# hidden layer 2 (32) -> output layer
	
	def forward(self, x):
		# Forwards a state through the NN to get an action
		
		x = self.fc_in(x)	# input layer -> hidden layer 1
		x = F.relu(x)
		
		x = self.fc_h1(x)	# hidden layer 1 -> hidden layer 2
		x = F.relu(x)
		
		x = self.fc_out(x)	# hidden layer 2 -> output layer
		return x

class DQN(object):
	# A Deep-Q Network
	#	Double DQN: Use 2 networks to decouple the action selection from the target Q value generation (Reduces the overestimation of q values, more stable learning)
	#		Implemented in DQN.update
	# 	Dueling DQN: Only learn when actions have an effect on the rewards (Good if there is a gamestate when any input is accepted)
	#		Implemented in NN.forward
	#	PER Prioritized Experience Replay: Very important memories may occur rarely thus needing prioritization
	#		Implemented in ReplayBuffer
	
	# Hyperparameters
	ALPHA = 0.001			# learning rate
	ALPHA_DECAY = 0.01		# [UNUSED] for Adam
	GAMMA = 0.95			# discount factor (default: 0.95)
	EPSILON_MAX = 1.0		# epsilon greedy threshold
	EPSILON_MIN = 0.01
	EPSILON_DECAY = 0.995
	TAU = 0.002				# target update factor for double DQN
	
	DOUBLE_DQN = False
	
	def __init__(self, input_size, output_size, training_mode, seed):
		self.seed = random.seed(seed)
		self.epsilon = self.EPSILON_MAX
		self.training_mode = training_mode
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.memory = ReplayBuffer(self.device, seed=seed)
		self.nn = NN(input_size, output_size, seed).to(self.device)
		if self.DOUBLE_DQN:
			self.target_nn = NN(input_size, output_size, seed).to(self.device)
		self.optimizer = optim.Adam(self.nn.parameters(), lr=self.ALPHA, amsgrad=False)
		self.loss_func = nn.MSELoss()
	
	def save_model(self, filename=None):
		if filename==None:
			filename = "models/{}_{}_{}_{}_{}.mdl".format(self.ALPHA, self.GAMMA, self.EPSILON_MAX, self.EPSILON_MIN, self.EPSILON_DECAY)
		torch.save(self.nn.state_dict(), filename)
	
	def load_model(self, filename):
		state_dict = torch.load(filename, map_location=self.device)
		self.nn.load_state_dict(state_dict)
	
	def get_action(self, state):
		# Extract an output tensor (action) by forwarding the state into the NN
		if self.training_mode and random.random() < self.epsilon:
			return random.choice(np.arange(self.nn.output_size))
		
		state = torch.tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)
		
		self.nn.eval()				# Set NN to eval mode
		with torch.no_grad():		# Disable autograd engine
			value = self.nn(state)	# Forward state to NN
		self.nn.train()				# Set NN back to train mode
		
		# Return the action with the highest tensor value
		action_max_value, index = torch.max(value, 1)
		
		return index.item()
	
	def update(self, states, actions, rewards, next_states, dones):
		# Stores the experience in the replay buffer
		self.memory.add_experience(states, actions, rewards, next_states, dones)
		
		# Wait for the replay memory to be filled with a few experiences first before learning from it
		if(not self.memory.can_pick()):
			return
		
		# Extract a random batch of experiences
		states, actions, rewards, next_states, dones = self.memory.sample()
		
		# Normalize rewards / gradient step size
		#rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
		
		# Calculate current Q-Value from current states
		curr_Q = self.nn(states).gather(1, actions)
		
		if self.DOUBLE_DQN:
			# Double DQN
			next_Q = self.target_nn(next_states)	#.detach()
			max_next_Q = torch.max(next_Q, 1)[0].unsqueeze(1)
			#next_Q = self.nn(next_states).detach().max(1)[1].unsqueeze(1)
			#max_next_Q = self.target_nn(next_states)[next_Q].unsqueeze(1)
		else:
			# Vanilla DQN
			next_Q = self.nn(next_states)	#.detach()
			max_next_Q = torch.max(next_Q, 1)[0].unsqueeze(1)
		
		# Calculate Q-Target values and the loss
		expected_Q = (1 - dones) * (max_next_Q * self.GAMMA) + rewards
		loss = self.loss_func(curr_Q, expected_Q)
		
		# Minimize the loss
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		
		if self.DOUBLE_DQN:
			# Slowly update target NN from NN using factor TAU
			for target_param, param in zip(self.target_nn.parameters(), self.nn.parameters()):
				target_param.data.copy_(self.TAU * param + (1 - self.TAU) * target_param)
		
		# Update exploration rate
		self.epsilon = max(self.EPSILON_MIN, self.epsilon*self.EPSILON_DECAY)
