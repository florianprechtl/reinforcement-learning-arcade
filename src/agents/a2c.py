import gym
import os
import math
import pickle
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import agents.agent as agent


class Actor(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, seed):
        super(Actor, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.seed = seed

        self.fc_in = nn.Linear(input_size, hidden1_size)
        self.fc_h1 = nn.Linear(hidden1_size, hidden2_size)
        self.fc_out = nn.Linear(hidden2_size, output_size)

    def forward(self, x):
        x = self.fc_in(x)
        x = F.relu(x)
        x = self.fc_h1(x)
        x = F.relu(x)
        x = self.fc_out(x)
        x = x.squeeze(0)
        distribution = Categorical(F.softmax(x, dim=-1))
        return distribution


class Critic(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, seed):
        super(Critic, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.seed = seed
        critic_reward_size = 1

        self.fc_in = nn.Linear(input_size, hidden1_size)
        self.fc_h1 = nn.Linear(hidden1_size, hidden2_size)
        self.fc_out = nn.Linear(hidden2_size, critic_reward_size)

    def forward(self, x):
        x = self.fc_in(x)
        x = F.relu(x)
        x = self.fc_h1(x)
        x = F.relu(x)
        x = self.fc_out(x)
        return x


class ConvolutionalActor(nn.Module):
    # A simple convolutional neural network

    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, seed):
        super(ConvolutionalActor, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.seed = seed

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
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.squeeze(0)
        x = Categorical(F.softmax(x, dim=-1))
        return x


class ConvolutionalCritic(nn.Module):
    # A simple convolutional neural network

    def __init__(self, input_size, hidden1_size, hidden2_size, output_size, seed):
        super(ConvolutionalCritic, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.seed = seed

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
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Agent(agent.IAgent):
    # A Actor critic etwork
    # A2C: Todo: Explanation

    NAME = "A2C"

    # Hyperparameters
    ALPHA = 0.001  # learning rate of the actor
    BETA = 0.001  # learning rate of the critic
    GAMMA = 0.99  # discount factor for rewards of previous steps

    EPSILON_MAX = 1.0  # epsilon greedy threshold
    EPSILON_MIN = 0.02
    EPSILON_DECAY = 30000  # amount of steps to reach half-life (0.99 ~~ 400 steps)

    # Layer Sizes
    H_ACTOR_LAYER_1_SIZE = 256
    H_ACTOR_LAYER_2_SIZE = 128
    H_CRITIC_LAYER_1_SIZE = 256
    H_CRITIC_LAYER_2_SIZE = 128

    def __init__(
        self, input_size, output_size, training_mode, is_conv, load_filename, seed
    ):
        self.step = 0  # Total steps
        self.epsilon = self.EPSILON_MAX
        self.input_size = input_size  # Input size is the number of features
        self.output_size = output_size  # Output size is the number of possible actions
        self.load_filename = load_filename
        self.seed = torch.manual_seed(seed)  # Seed if needed
        # Todo: Differentiate between train and tets mode
        self.training_mode = training_mode
        self.is_conv = is_conv
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Handle loading of previously saved models
        if self.load_filename:
            with open(self.load_filename, "rb") as f:
                self.ALPHA = pickle.load(f)
                self.BETA = pickle.load(f)
                self.GAMMA = pickle.load(f)
                self.EPSILON_DECAY = pickle.load(f)
                self.EPSILON_MAX = pickle.load(f)
                self.EPSILON_MIN = pickle.load(f)
                self.H_ACTOR_LAYER_1_SIZE = pickle.load(f)
                self.H_ACTOR_LAYER_2_SIZE = pickle.load(f)
                self.H_CRITIC_LAYER_1_SIZE = pickle.load(f)
                self.H_CRITIC_LAYER_2_SIZE = pickle.load(f)
                self.step = pickle.load(f)
                self.epsilon = pickle.load(f)

        # Initialize empty values of episode
        self.resetMemory()

    def init_network(self):
        # Initialize actor and critic
        # Decide between actor types
        ActorType = Actor
        CriticType = Critic
        if self.is_conv:
            ActorType = ConvolutionalActor
            CriticType = ConvolutionalCritic

        # Initialize actor
        self.actor = ActorType(
            self.input_size,
            self.H_ACTOR_LAYER_1_SIZE,
            self.H_ACTOR_LAYER_2_SIZE,
            self.output_size,
            self.seed,
        ).to(self.device)

        # Initialize critic
        self.critic = CriticType(
            self.input_size,
            self.H_CRITIC_LAYER_1_SIZE,
            self.H_CRITIC_LAYER_2_SIZE,
            self.output_size,
            self.seed,
        ).to(self.device)

        # Initialize Optimizers of actor and critic with separate learning rates
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.ALPHA)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.BETA)

        # Load previously saved model
        if self.load_filename:
            self.load_model(self.load_filename)

    def resetMemory(self):
        # Reset values of last episode
        self.log_probs = []
        self.critic_ratings = []
        self.rewards = []
        self.dones = []

    def addExperienceToMemory(self, log_prob, critic_rating, reward, done):
        # Add values of this step to our array of values of the entire episode
        # Those values are used to update the actor and critic after the episode is done
        self.log_probs.append(log_prob)
        self.critic_ratings.append(critic_rating)
        self.rewards.append(
            torch.tensor([reward], dtype=torch.float, device=self.device)
        )
        self.dones.append(
            torch.tensor([1 - done], dtype=torch.float, device=self.device)
        )

    def save_model(self, filename):
        # Save actor and critic model in separate files

        filename_actor = filename + ".mdl.actor.pth"
        filename_critic = filename + ".mdl.critic.pth"

        torch.save(self.actor.state_dict(), filename_actor)
        torch.save(self.critic.state_dict(), filename_critic)

        # Save Hyperparamters
        with open(filename + ".mdl", "wb") as f:
            pickle.dump(self.ALPHA, f)
            pickle.dump(self.BETA, f)
            pickle.dump(self.GAMMA, f)
            pickle.dump(self.EPSILON_DECAY, f)
            pickle.dump(self.EPSILON_MAX, f)
            pickle.dump(self.EPSILON_MIN, f)
            pickle.dump(self.H_ACTOR_LAYER_1_SIZE, f)
            pickle.dump(self.H_ACTOR_LAYER_2_SIZE, f)
            pickle.dump(self.H_CRITIC_LAYER_1_SIZE, f)
            pickle.dump(self.H_CRITIC_LAYER_2_SIZE, f)
            pickle.dump(self.step, f)
            pickle.dump(self.epsilon, f)

    def load_model(self, filename):
        # For loading the models of the actor and the critic you just have to pass in
        # the filename without the appending actor or critic
        # e.g. real filenames:
        #           model_249.mdl.actor.pth
        #           model_249.mdl.critic.pth
        # Pass in:  model_249.mdl

        # Load old trained actor model
        actor_filename = filename + ".actor.pth"
        critic_filename = filename + ".critic.pth"

        state_dict_actor = torch.load(actor_filename, map_location=self.device)
        self.actor.load_state_dict(state_dict_actor)

        # Load old trained critic model
        state_dict_critic = torch.load(critic_filename, map_location=self.device)
        self.critic.load_state_dict(state_dict_critic)

    def copy_from(self, model):
        pass

    def get_action(self, state):
        # Explore (random action)

        if self.training_mode and torch.rand(1) < self.epsilon:
            action = torch.randint(0, self.actor.output_size, (1,)).item()
            return action

        # Load state on device
        state = torch.tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)
        # Get distribution of probabilities according to the actors current policy
        dist = self.actor(state)
        # Choose one action out of the distribution according to their probabilities
        action = dist.sample()
        # Get value of action and return to main function
        action = action.cpu().numpy()
        return action

    def update(self, state, action, reward, next_state, done):
        # Todo: Understand when to use which of the upper cases of loading variables to the device
        state = torch.tensor(state, dtype=torch.float).to(self.device).unsqueeze(0)
        action = torch.tensor(action, dtype=torch.float).to(self.device).unsqueeze(0)

        # Get critic rating based on current policy
        critic_rating = self.critic(state)
        # Get distribution of probabilities according to the actors current policy
        dist = self.actor(state)
        # Get log of the probability of the current action
        log_prob = dist.log_prob(action).unsqueeze(0)
        # Add computated values of this step to the memory of the episode
        self.addExperienceToMemory(log_prob, critic_rating, reward, done)

        # Critic and actor do only get updated after the episode is done
        # That means that the networks do only get updated if they fail before the last step is reached
        if done:
            next_state = (
                torch.tensor(next_state, dtype=torch.float).to(self.device).unsqueeze(0)
            )
            next_value = self.critic(next_state)
            discounted_rewards = self.discount_rewards(
                next_value, self.rewards, self.dones, gamma=self.GAMMA
            )

            log_probs = torch.cat(self.log_probs)
            discounted_rewards = torch.cat(discounted_rewards).detach()
            critic_ratings = torch.cat(self.critic_ratings)

            advantage = discounted_rewards - critic_ratings

            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            critic_loss.backward(retain_graph=True)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            if len(self.critic_ratings) > 500:
                self.resetMemory()

        # Update exploration rate
        self.epsilon = self.EPSILON_MIN + (
            self.EPSILON_MAX - self.EPSILON_MIN
        ) * math.exp(-1.0 * self.step / self.EPSILON_DECAY)
        self.step += 1

    def discount_rewards(self, next_value, rewards, dones, gamma=0.99):
        # Set initial reward to discount
        reward_to_discount = next_value
        # Initialize empty array of discounted rewards
        discounted_rewards = []

        # Iterate through each step of the episode
        # This is done in reverse, because the oldest steps lead to the end result and were more important than newer steps
        # Todo: Explain this in a better way
        for step in reversed(range(len(rewards))):
            # Add discounted reward of previous step (discounted by gamma and dependent on the factor done of the step)
            # to the reward of the current step
            reward_to_discount = (
                rewards[step] + gamma * reward_to_discount * dones[step]
            )
            discounted_rewards.insert(0, reward_to_discount)
        return discounted_rewards
