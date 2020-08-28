import gym
import os
import math
import pickle
import random
import numpy as np
from collections import namedtuple

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import utils.replay_buffers as mem

import agents.agent as agent


SavedAction = namedtuple("SavedAction", ["log_prob", "value"])


class ActorCriticNetwork(nn.Module):
    def __init__(
        self, input_dims, output_dims, device, seed, alpha=0.0001, fc1_dims=256, fc2_dims=256
    ):
        super(ActorCriticNetwork, self).__init__()
        self.seed = T.manual_seed(seed)
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.output_dims = output_dims

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.prob_value = nn.Linear(self.fc2_dims, self.output_dims)
        self.state_value = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.RMSprop(self.parameters(), lr=alpha)
        self.device = device
        self.to(device)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        prob_value = F.softmax(self.prob_value(x))
        state_value = self.state_value(x)

        return prob_value, state_value


class ConvolutionalActorCriticNetwork(nn.Module):
    def __init__(self, input_dims, output_dims, device, seed, alpha=0.0001):
        super(ConvolutionalActorCriticNetwork, self).__init__()
        self.seed = T.manual_seed(seed)
        self.input_dims = input_dims
        self.output_dims = output_dims

        self.features = nn.Sequential(
            nn.Conv2d(input_dims[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        size_x = input_dims[1]
        size_y = input_dims[2]

        last_layer = None
        for layer in self.features:
            if type(layer) is nn.Conv2d:
                size_x = (
                    (size_x - layer.kernel_size[0]) // layer.stride[0]) + 1
                size_y = (
                    (size_y - layer.kernel_size[1]) // layer.stride[1]) + 1
                last_layer = layer

        self.fc = nn.Sequential(
            nn.Linear(size_x*size_y*last_layer.out_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )

        self.flatten = nn.Flatten()

        self.state_value = nn.Linear(512, 1)
        self.prob_value = nn.Linear(512, self.output_dims)

        self.optimizer = optim.RMSprop(self.parameters(), lr=alpha)
        self.device = device
        self.to(device)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc(x)
        state_value = self.state_value(x)
        prob_value = F.softmax(self.prob_value(x), dim=-1)
        return prob_value, state_value


class Agent(agent.IAgent):
    # Advantage Actor Critic
    # Saves all probability distributions (actor_values), rewards and state values (critic_values)
    # After the episode is done the losses are calculated and backpropagated
    # The loss of the actor uses the discounted factor method
    # Real accumulation of all steps that lead to the outcome (different to DAC Model)
    NAME = "AAC"

    ALPHA = 0.0001
    GAMMA = 0.99
    LAYER_H1_SIZE = 256
    LAYER_H2_SIZE = 128

    EPSILON_MAX = 1.0  # epsilon greedy threshold
    EPSILON_MIN = 0.01
    # amount of steps to reach half-life (0.99 ~~ 400 steps)
    EPSILON_DECAY = 1500

    def __init__(
        self, input_size, output_size, training_mode, is_conv, load_filename, seed
    ):

        self.epsilon = self.EPSILON_MAX
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu:0")
        self.seed = seed
        self.training_mode = training_mode
        self.is_conv = is_conv
        self.step = 0
        self.input_size = input_size
        self.output_size = output_size
        self.load_filename = load_filename

        # Handle loading of previously saved models
        if load_filename:
            with open(load_filename, "rb") as f:
                self.ALPHA = pickle.load(f)
                self.GAMMA = pickle.load(f)
                self.LAYER_H1_SIZE = pickle.load(f)
                self.LAYER_H2_SIZE = pickle.load(f)
                self.EPSILON_MIN = pickle.load(f)
                self.EPSILON_MAX = pickle.load(f)
                self.EPSILON_DECAY = pickle.load(f)
                self.epsilon = pickle.load(f)
                self.step = pickle.load(f)
                self.seed = pickle.load(f)

    def init_network(self):
        # Initiate ActorCriticNetwork
        if self.is_conv:
            self.actor_critic = ConvolutionalActorCriticNetwork(
                input_dims=self.input_size,
                output_dims=self.output_size,
                device=self.device,
                seed=self.seed,
                alpha=self.ALPHA
            )
        else:
            self.actor_critic = ActorCriticNetwork(
                input_dims=self.input_size,
                output_dims=self.output_size,
                device=self.device,
                seed=self.seed,
                alpha=self.ALPHA,
                fc1_dims=self.LAYER_H1_SIZE,
                fc2_dims=self.LAYER_H2_SIZE,
            )
        # Load previously saved model
        if self.load_filename:
            self.load_model(self.load_filename)

    def save_model(self, filename):
        # Save actor and critic model in separate files
        filename_network = filename + ".mdl.actor-critic.pth"
        T.save(self.actor_critic.state_dict(), filename_network)

        # Save Hyperparamters
        with open(filename + ".mdl", "wb") as f:
            pickle.dump(self.ALPHA, f)
            pickle.dump(self.GAMMA, f)
            pickle.dump(self.LAYER_H1_SIZE, f)
            pickle.dump(self.LAYER_H2_SIZE, f)
            pickle.dump(self.EPSILON_MIN, f)
            pickle.dump(self.EPSILON_MIN, f)
            pickle.dump(self.EPSILON_DECAY, f)
            pickle.dump(self.epsilon, f)
            pickle.dump(self.step, f)
            pickle.dump(self.seed, f)

    def load_model(self, filename):
        # Load old trained actor model
        filename_network = filename + ".actor-critic.pth"
        state_dict_actor = T.load(filename_network, map_location=self.device)
        self.actor_critic.load_state_dict(state_dict_actor)

    def copy_from(self, model):
        pass

    def get_action(self, state):
        state = T.Tensor(state).to(self.device)
        # Calculate prob and state value of current state
        prob_value, state_value = self.actor_critic.forward(state)

        action_probs = T.distributions.Categorical(prob_value)
        # Choose action to take
        action = action_probs.sample()

        if self.training_mode and T.rand(1) < self.epsilon:
            action = (
                action +
                T.randint(0, self.actor_critic.output_dims, (1,)).item()
            ) % self.actor_critic.output_dims

        # Update exploration rate
        if self.epsilon > 0:
            self.epsilon = self.EPSILON_MIN + (
                self.EPSILON_MAX - self.EPSILON_MIN
            ) * math.exp(-1.0 * self.step / self.EPSILON_DECAY)
            self.step += 1

        # Save probability distribution, action and state_value for later use in finnished_episode function
        self.actor_critic.saved_actions.append(
            SavedAction(action_probs.log_prob(action), state_value)
        )

        return action.item()

    def update(self, state, action, reward, next_state, done):
        # Save reward for later use in finnished_episode function
        self.actor_critic.rewards.append(reward)

        if done:
            self.finnish_episode()

    def finnish_episode(self):
        total_discounted_reward = 0
        saved_actions = self.actor_critic.saved_actions
        policy_losses = []  # List to save actor (policy) loss
        value_losses = []  # List to save critic (value) loss
        returns = []  # List to save the true values

        for r in self.actor_critic.rewards[::-1]:
            # Calculate the discounted value
            total_discounted_reward = r + self.GAMMA * total_discounted_reward
            returns.insert(0, total_discounted_reward)

        returns = T.Tensor(returns)
        eps = np.finfo(np.float32).eps.item()  # small epsilon
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for (log_prob, value), total_discounted_reward in zip(saved_actions, returns):
            # Calculate advantage
            advantage = total_discounted_reward - value.item()
            # Calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)
            # Calculate critic (value) loss using L1 smooth loss
            value_losses.append(advantage ** 2)

        # Reset gradients
        self.actor_critic.optimizer.zero_grad()

        # Sum up all the values of policy_losses and value_losses
        loss = T.stack(policy_losses).sum() + T.stack(value_losses).sum()

        # Perform backprop
        loss.backward()
        self.actor_critic.optimizer.step()

        # Reset rewards and action buffer
        del self.actor_critic.rewards[:]
        del self.actor_critic.saved_actions[:]
