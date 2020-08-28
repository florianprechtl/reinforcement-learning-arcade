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
import torch
import pickle

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


class Agent(object):
    NAME = "PPO"


    buffer_s = []
    buffer_a = []
    buffer_r = []
    stepCounter = 0

    # Hyperparameters
    EP_LEN = 200 #200
    A_LR = 0.0001  # learning rate (0.001)
    C_LR = 0.0002  # learning rate (0.001)
    GAMMA = 0.9  # discount factor (0.95)?
    EPSILON = 0.2
    BATCH_SIZE = 32  # size of one mini-batch to sample
    A_UPDATE_STEP = 10
    C_UPDATE_STEP = 10
    MEMORY_SIZE = 65536		# size of the replay 

    def __init__(self, input_size, output_size, training_mode, is_conv, load_filename, seed):
        super(Agent, self).__init__()
        self.input_size = input_size  # Input size is the number of features
        self.output_size = output_size  # Output size is the number of possible actions
        self.sess = tf.Session()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.seed = torch.manual_seed(seed)
        self.states_placeholder = tf.placeholder(tf.float32, [None, input_size], 'state')
        self.actions_placeholder = tf.placeholder(tf.int32, [None, 1], 'action')
        self.dicounted_rewards_placeholder = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantages_placeholder = tf.placeholder(tf.float32, [None, 1], 'advantage')

        
        # Handle loading of previously saved models
        if load_filename:
            pass
            # with open(load_filename, "rb") as f:
            #     self.A_UPDATE_STEP = pickle(f)
            #     self.C_UPDATE_STEP = pickle(f)
            #     self.A_LR = pickle(f)
            #     self.C_LR = pickle(f)
            #     self.GAMMA = pickle(f)
            #     self.EPSILON = pickle(f)
            #     self.MEMORY_SIZE = pickle(f)
            #     self.BATCH_SIZE = pickle.load(f)


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
        
        # # Load previously saved model
        # if load_filename:
        #     self.load_model(load_filename)



    def save_model(self, filename):
        # torch.save(self.nn.state_dict(), filename + ".mdl.pth")
        # with open(filename + ".mdl", "wb") as f:
        #     pickle.dump(self.A_UPDATE_STEP, f)
        #     pickle.dump(self.C_UPDATE_STEP, f)
        #     pickle.dump(self.A_LR, f)
        #     pickle.dump(self.C_LR, f)
        #     pickle.dump(self.GAMMA, f)
        #     pickle.dump(self.EPSILON, f)
        #     pickle.dump(self.MEMORY_SIZE, f)
        #     pickle.dump(self.BATCH_SIZE, f)

    def load_model(self, filename):
        # For loading the models of the actor and the critic you just have to pass in
        # the filename

        # Load old trained actor model
        load_filename = filename + ".mdl.pth"

        state_dict = torch.load(load_filename, map_location=self.device)
        self.load_state_dict(state_dict)


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
    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.value, {self.states_placeholder: s})[0, 0]


    # Return the best according for the current observation 'state'.
    def get_action(self, state):
        state = state[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.states_placeholder: state})
        return a


    def update(self, state, action, reward, next_state, done):
        # saving the new state, action and reward to the buffers
        self.buffer_s.append(state)
        self.buffer_a.append(action)
        self.buffer_r.append(reward)
        state = next_state

        # update ppo:
        # happens only if: next step completes batchsize
        # OR if: next step ends episode
        if (self.stepCounter+1) % self.BATCH_SIZE == 0 or self.stepCounter == self.EP_LEN-1:

            # v_s_ is the next state(describes the games state)
            v_s_ = self.get_v(next_state)

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
