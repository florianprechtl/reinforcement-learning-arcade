import gym
import os
import math
import pickle
import numpy as np

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import agents.agent as agent


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
        self.q = nn.Linear(self.fc2_dims, self.output_dims)
        self.v = nn.Linear(self.fc2_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(device)

    def forward(self, observation):
        x = F.relu(self.fc1(observation))
        x = F.relu(self.fc2(x))
        q = F.softmax(self.q(x))
        v = self.v(x)

        return q, v


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
    # Discrete Actor Critic
    # Update the policy after every action

    # The discounted reward is not calculated by accumulation of the discounted factors (accumulation of all steps within one episode)
    # but by adding together the current reward and the critic rating (forecast of the reward of the next state)
    NAME = "DAC"

    ALPHA = 0.0001
    GAMMA = 0.85
    LAYER_H1_SIZE = 256
    LAYER_H2_SIZE = 128

    def __init__(
        self, input_size, output_size, training_mode, is_conv, load_filename, seed
    ):

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu:0")
        self.is_conv = is_conv
        self.log_probs = None
        self.input_size = input_size
        self.output_size = output_size
        self.load_filename = load_filename
        self.seed = seed

        # Handle loading of previously saved models
        if load_filename:
            with open(load_filename, "rb") as f:
                self.ALPHA = pickle.load(f)
                self.GAMMA = pickle.load(f)
                self.LAYER_H1_SIZE = pickle.load(f)
                self.LAYER_H2_SIZE = pickle.load(f)
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
        self.log_probs = action_probs.log_prob(action)
        return action.item()

    def update(self, state, action, reward, next_state, done):
        # Reset gradients
        self.actor_critic.optimizer.zero_grad()

        # Load states on device
        state = T.Tensor(state).to(self.device)
        next_state = T.Tensor(next_state).to(self.device)
        # Calculate the state values (approximated reward) of the current and the next state
        _, critic_value = self.actor_critic.forward(state)
        _, next_critic_value = self.actor_critic.forward(next_state)

        # Calculate discounted reward and advantage
        reward = T.tensor(reward, dtype=T.float).to(self.device)
        discounted_reward = reward + self.GAMMA * \
            next_critic_value * (1 - int(done))
        advantage = discounted_reward - critic_value

        # Calculate loss
        actor_loss = -self.log_probs * advantage
        critic_loss = advantage ** 2

        # Sum up loss and execute backpropagation
        (actor_loss + critic_loss).backward()
        self.actor_critic.optimizer.step()
