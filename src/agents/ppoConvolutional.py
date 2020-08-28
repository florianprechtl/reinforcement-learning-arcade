# CartPole mit PPO:

# https://github.com/Ashboy64/rl-reimplementations/tree/master/Reimplementations/PPO
# https://github.com/adik993/ppo-pytorch
# https://github.com/4kasha/CartPole_PPO/blob/master/train.ipynb

# die update funktion erstellt ne batch mit zb 64 durchgeführten schritten,
# dann wendet man darauf die policy an (bewertung wie gut das war) und dann
# passt die policy die gewichte des nn an


import tensorflow as tf
import numpy as np
import gym
import time
import sys
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch

# Helper/Utility Class
class Probability_Distribution():
    def __init__(self, logits):
        self.logits = logits

    def neglogp(self, x):
        one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
        return tf.nn.softmax_cross_entropy_with_logits(logits = self.logits, labels = one_hot_actions)

    def sample(self):
        u = tf.random_uniform(shape = tf.shape(self.logits), dtype = tf.float32)
        return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)

    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis = -1, keep_dims = True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis = -1, keep_dims = True)
        p0 = ea0/z0
        return tf.reduce_sum(p0 * (tf.log(z0) - a0), axis = -1)

    def logp(self, x):
        return - self.neglogp(x)




class NN(nn.Module):
	# A simple neural network
	
	def __init__(self, input_size, hidden1_size, hidden2_size, output_size, seed):
		super(NN, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.seed = torch.manual_seed(seed)
		#
		self.fc_in = nn.Linear(input_size, hidden1_size)		# input layer -> hidden layer 1 (256)
		self.fc_h1 = nn.Linear(hidden1_size, hidden2_size)		# hidden layer 1 (256) -> hidden layer 2 (128)
		self.fc_out = nn.Linear(hidden2_size, output_size)		# hidden layer 2 (128) -> output layer
	
	def forward(self, x):
		# Forwards a state through the NN to get an action
		
		x = self.fc_in(x)	# input layer -> hidden layer 1
		x = F.relu(x)
		
		x = self.fc_h1(x)	# hidden layer 1 -> hidden layer 2
		x = torch.tanh(x)	#x = F.relu(x)
		
		x = self.fc_out(x)	# hidden layer 2 -> output layer
		return x

class ConvolutionNN(nn.Module):
	# A simple convolutional neural network
	
	def __init__(self, input_size, hidden1_size, hidden2_size, output_size, seed):
		super(ConvolutionNN, self).__init__()
		self.input_size = input_size
		self.output_size = output_size
		self.seed = torch.manual_seed(seed)
		#
		"""
		#self.conv1 = nn.Conv2d(input_size, 32, kernel_size=8, stride=4)			# (64, 40, 2) -> (15*9*32)
		#self.bn1 = nn.BatchNorm2d(32)
		self.conv1 = nn.Conv2d(input_size, 64, kernel_size=4, stride=2)
		self.bn1 = nn.BatchNorm2d(64)
		self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1)
		self.bn2 = nn.BatchNorm2d(32)
		self.fc1 = nn.Linear(150304, 512)										# self.fc4 = nn.Linear(15 * 9 * 32, 512)		# W: 64/4 = 16-1 = 15, H: 40/4 = 10-1 = 9
		self.head = nn.Linear(512, output_size)
		"""
		self.features = nn.Sequential(
			nn.Conv2d(input_size[0], 32, kernel_size=8, stride=4),
			nn.ReLU(),
			nn.Conv2d(32, 64, kernel_size=4, stride=2),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, stride=1),
			nn.ReLU()
		)
		
		size_x = input_size[1]
		size_y = input_size[2]
		
		last_layer = None
		for layer in self.features:
			if type(layer) is nn.Conv2d:
				size_x = ((size_x - layer.kernel_size[0]) // layer.stride[0]) + 1
				size_y = ((size_y - layer.kernel_size[1]) // layer.stride[1]) + 1
				last_layer = layer
		
		self.fc = nn.Sequential(
			nn.Linear(size_x*size_y*last_layer.out_channels, 256),
			nn.ReLU(),
			nn.Linear(256, output_size)
		)
	
	def forward(self, x):
        # Forwards a state through the NN to get an action

            print(x.shape)
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)

            return x




class Agent(object):
    NAME = "PPO"


    buffer_s = []
    buffer_a = []
    buffer_r = []
    stepCounter = 0

    # Hyperparameters
    LAYER_H1_SIZE = 256
    LAYER_H2_SIZE = 128
    EP_MAX = 200 #200
    EP_LEN = 200 #200
    A_LR = 0.0001  # learning rate (0.001)
    C_LR = 0.0002  # learning rate (0.001)
    GAMMA = 0.9  # discount factor (0.95)?
    EPSILON = 0.2
    BATCH_SIZE = 32  # size of one mini-batch to sample
    A_UPDATE_STEP = 10
    C_UPDATE_STEP = 10
    # S_DIM = 4
    MEMORY_SIZE = 65536		# size of the replay 

    def __init__(self, input_size, output_size, training_mode, is_conv, load_filename, seed):
        super(Agent, self).__init__()
        self.input_size = input_size  # Input size is the number of features
        self.output_size = output_size  # Output size is the number of possible actions
        self.sess = tf.Session()
        self.is_conv = is_conv
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.seed = torch.manual_seed(seed)

        # Create the model / NN
        ModelType = NN
        if is_conv:
            ModelType = ConvolutionNN

        self.nn = ModelType(input_size, self.LAYER_H1_SIZE, self.LAYER_H2_SIZE, output_size, seed).to(self.device)
        self.target_nn = ModelType(input_size, self.LAYER_H1_SIZE, self.LAYER_H2_SIZE, output_size, seed).to(self.device)
        # self.states_placeholder = tf.placeholder(tf.float32, [None, self.S_DIM], 'state')
        # Outsource my dimension?
        self.states_placeholder = tf.placeholder(tf.float32, [1, None, 80, 80], 'state')
        self.actions_placeholder = tf.placeholder(tf.int32, [None, 1], 'action')
        # self.dicounted_rewards_placeholder = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.dicounted_rewards_placeholder = tf.placeholder(tf.float32, [32, 1, 80, 1], 'discounted_r')
        # self.advantages_placeholder = tf.placeholder(tf.float32, [None, 1], 'advantage')
        self.advantages_placeholder = tf.placeholder(tf.float32, [32, 128, 80, 1], 'advantage')

        # critic
        self.value = self.build_critic()
        self.advantage = self.dicounted_rewards_placeholder - self.value
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(self.C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self.build_policy('pi', trainable=True)
        oldpi, oldpi_params = self.build_policy('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(), axis=0)
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                ratio = tf.exp(pi.logp(self.actions_placeholder) - oldpi.logp(self.actions_placeholder))
                surr = ratio * self.advantages_placeholder
            self.aloss = -tf.reduce_mean(tf.minimum(
                surr,
                tf.clip_by_value(ratio, 1.-self.EPSILON, 1.+self.EPSILON)*self.advantages_placeholder))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(self.A_LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    # todo
    def save_model(self, filename):
        pass

    def build_policy(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.states_placeholder, 100, tf.nn.relu, trainable=trainable)
            l2 = tf.layers.dense(l1, 100, tf.nn.relu, trainable=trainable)
            pd = Probability_Distribution(tf.layers.dense(l2, self.output_size, trainable=trainable))
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return pd, params

    def build_critic(self):
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.states_placeholder, 100, tf.nn.relu)
            val = tf.layers.dense(l1, 1)
            return val

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.states_placeholder: s})
        return a

    # if s has too many dimensions, s is shrinked
    # ...i think it just returns an empty array of the same dimension/size as s(?)
    def get_v(self, s, dim):
        if dim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.value, {self.states_placeholder: s})[0, 0]


    # Return the best according for the current observation 'state'.
    def get_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)
        # state = state[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.states_placeholder: state})
        return a.all()


    def update(self, state, action, reward, next_state, done):
        # saving the new state, action and reward to the buffers
        self.buffer_s.append(state)
        self.buffer_a.append(action)
        self.buffer_r.append(reward)
        state = next_state
        state = torch.tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)
        self.curDim = next_state.ndim
        next_state = torch.tensor(next_state, dtype=torch.float).to(self.device).unsqueeze(0)


        # update ppo:
        # happens only if: next step completes batchsize
        # OR if: next step ends episode
        if (self.stepCounter+1) % self.BATCH_SIZE == 0 or self.stepCounter == self.EP_LEN-1:

            # v_s_ is the next state(describes the games state)
            v_s_ = self.get_v(next_state, self.curDim)

            # rewards are weighted. 
            discounted_r = []


            # iterating through the rewards of the episode in reverse order
            # all rewards are added, but older rewards are less relevant
            # (all old rewards) * 0,9(GAMMA) + new reward
            # for r in buffer_r[::-1]: WAS LIKE THIS BEFORE: wrong way?
            for r in self.buffer_r[::-1]:  # maybe wrong "fix"
                v_s_ = r + self.GAMMA * v_s_    # DONES hinzufügen, siehe flo
                discounted_r.append(v_s_)
            discounted_r.reverse()


             # getting the previous state, action and the weighted rewards from the buffer
            last_state, last_action, last_reward = np.vstack(self.buffer_s), np.vstack(self.buffer_a), np.array(discounted_r)[:, np.newaxis]


            last_state = torch.tensor(last_state, dtype=torch.float).to(self.device).unsqueeze(0)

            # resetting the buffers
            self.buffer_s, self.buffer_a, self.buffer_r = [], [], []

            # updating with the previous state, previous action, weighted rewards and with the current state and done
            self.inner_update(last_state, last_action, last_reward, next_state, done)
        
        
        self.stepCounter = self.stepCounter + 1

        # no more steps in the episode if "done"
        if done:
            self.buffer_s, self.buffer_a, self.buffer_r = [], [], []
            self.stepCounter = 0


    # Update ppo
    def inner_update(self, state, action, reward, next_state, done):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.states_placeholder: state, self.dicounted_rewards_placeholder: reward})

        # update actor
        [self.sess.run(self.atrain_op, {self.states_placeholder: state, self.actions_placeholder: action, self.advantages_placeholder: adv}) for _ in range(self.A_UPDATE_STEP)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.states_placeholder: state, self.dicounted_rewards_placeholder: reward}) for _ in range(self.C_UPDATE_STEP)]

        self.A_LR = max(0.0, self.A_LR - 0.000000125)
        self.C_LR = max(0.0, self.C_LR - 0.0000025)
