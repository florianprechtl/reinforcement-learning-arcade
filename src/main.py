#!/usr/bin/python3

import os
import sys
import time
import torch
import json
import argparse
import importlib
import numpy as np
import collections
import utils.logger
import utils.env_wrapper
import matplotlib.pyplot as plt


def main():
    sys.stdout = utils.logger.Logger()

    # Handle the command line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--episodes", help="The total amount of episodes to train", type=int, default=300)
    parser.add_argument("-s", "--steps", help="The total amount of steps per episode", type=int, default=200)
    parser.add_argument("-E", "--environment", help="The gym environment name", default="CartPole-v1")
    parser.add_argument("-S", "--seed", help="The prng seed for all random actions", type=int, default=int(time.time()))
    parser.add_argument("-a", "--agent", help="The name of the agent to use", default="dqn")
    parser.add_argument("-A", "--argsext", help="The extended arguements for the environment (json)")
    parser.add_argument("-f", "--file", help="The path to the model which should be loaded")
    parser.add_argument("-t", "--train", help="Whether to train a network or not", action="store_true")
    parser.add_argument("-r", "--render", help="Whether to render the game or not", action="store_true")
    parser.add_argument("-H", "--headless", help="Whether to write logs to tensorboard or local plotting", action="store_true")
    parser.add_argument("-R", "--record", help="Whether to record the gameplay to a video file (.avi) or not", action="store_true")
    parser.add_argument("-g", "--g_index", help="The index of the combination that is executed by the gridsearch. Result will be saved in a Gridsearch folder", default=-1)
    parser.add_argument("-T", "--timeout", help="Timeout in ms after each episode", default=0)
    args = parser.parse_args()

    print("\n", " Current Configuration ".center(79, "-"))
    for arg in vars(args):
        print("|", "{} = {} ".format(arg, getattr(args, arg)).center(78, " ") + "|")
    print(" ".ljust(80, "-"), "\n")

    if args.record:
        args.render = True
        args.timeout = 1000

    if not args.train:
        if not args.file:
            print("The model filename cannot be empty")
            return

    # Define an inner function to backup our current training state
    def save_state():
        print("> Saving state...\n")
        filename = "model_" + str(episode + 1)
        dir = "./../models/{}/{}/{}/".format(args.environment, agent.NAME, start_time)
        if int(args.g_index) >= 0:
            dir = "./../models/{}/{}/{}/{}/".format(args.environment, agent.NAME, "GridSearch", start_time)
        os.makedirs(dir, exist_ok=True)
        agent.save_model(dir + filename)

        plt.figure(2)
        plt.title("Training..." if args.train else "Evaluating...")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.plot(total_rewards)
        plt.plot(smoothed_rewards)
        plt.pause(0.0001)
        plt.savefig(dir + filename + ".png")
        plt.close()

        with open(dir + filename + ".log", "w") as f:
            f.write(sys.stdout.data)

    # Initialize environment and agent
    start_time = time.time()
    env = utils.env_wrapper.EnvWrapper(args)
    agent_module = importlib.import_module("agents." + args.agent)
    agent = agent_module.Agent(env.get_input_size(), env.get_output_size(), args.train, env.is_conv(), args.file, args.seed)

    # Initialize network
    agent.init_network()
    print("* Running agent '{}' starting at {}".format(agent.NAME, time.ctime()))

    # Set training combination
    if int(args.g_index) >= 0:
        combination_index = int(args.g_index)
        with open("./gridsearch.json") as json_combi:
            data = json.load(json_combi)
            command = data["command"]
            combinations = data["combinations"]
            combination = combinations[combination_index]
            print("* Executed Parameter Combination: '{}'\n".format(combination))
            if (hasattr(combination, "ALPHA")):
                agent.ALPHA = float(combination["ALPHA"])
            if (hasattr(combination, "BETA")):
                agent.BETA = float(combination["BETA"])
            if (hasattr(combination, "LAYER_H1_SIZE")):
                agent.LAYER_H1_SIZE = int(combination["LAYER_H1_SIZE"])
            if (hasattr(combination, "LAYER_H2_SIZE")):
                agent.LAYER_H2_SIZE = int(combination["LAYER_H2_SIZE"])
            if (hasattr(combination, "GAMMA")):
                agent.GAMMA = float(combination["GAMMA"])
            if (hasattr(combination, "EPSILON_MAX")):
                agent.EPSILON_MAX = float(combination["EPSILON_MAX"])
            if (hasattr(combination, "EPSILON_MIN")):
                agent.EPSILON_MIN = float(combination["EPSILON_MIN"])
            if (hasattr(combination, "EPSILON_DECAY")):
                agent.EPSILON_DECAY = int(combination["EPSILON_DECAY"])
            if (hasattr(combination, "ALPHA_DECAY")):
                agent.ALPHA_DECAY = float(combination["ALPHA_DECAY"])
            if (hasattr(combination, "TAU")):
                agent.TAU = float(combination["TAU"])
            if (hasattr(combination, "TGT_UPDATE_RATE")):
                agent.TGT_UPDATE_RATE = int(combination["TGT_UPDATE_RATE"])
            if (hasattr(combination, "MEMORY_SIZE")):
                agent.MEMORY_SIZE = int(combination["MEMORY_SIZE"])
            if (hasattr(combination, "MEMORY_FILL")):
                agent.MEMORY_FILL = int(combination["MEMORY_FILL"])
            if (hasattr(combination, "BATCH_SIZE")):
                agent.BATCH_SIZE = int(combination["BATCH_SIZE"])
            if (hasattr(combination, "UPDATE_RATE")):
                agent.UPDATE_RATE = int(combination["UPDATE_RATE"])
            if (hasattr(combination, "DOUBLE_DQN")):
                agent.DOUBLE_DQN = bool(combination["DOUBLE_DQN"])
            if (hasattr(combination, "DUELING_DQN")):
                agent.DUELING_DQN = bool(combination["DUELING_DQN"])
            if (hasattr(combination, "PRIO_REPLAY")):
                agent.PRIO_REPLAY = bool(combination["PRIO_REPLAY"])

    # Average buffers for logging
    total_rewards = []
    smoothed_rewards = []
    avg_episode_reward = 0
    last_rewards = collections.deque(maxlen=100)
    last_steps = collections.deque(maxlen=100)

    try:
        # Main learning/playing loop, most important
        for episode in range(args.episodes):
            episode_start_time = time.time()
            episode_reward = 0
            state = env.reset()
            action = None

            for step in range(args.steps):
                # Handle rendering
                if args.render:
                    env.render(episode, action, episode_reward, avg_episode_reward)

                # Feed the network and get back an appropriate action
                action = agent.get_action(state)

                # Pass the action to the environment
                next_state, reward, done, info = env.step(action)

                if args.train:
                    # Train the network
                    agent.update(state, action, reward, next_state, done)

                    # Copy network from learning AI to cloned AI
                    if step % 1000 == 0:
                        env.copy_network(agent)

                state = next_state
                episode_reward += reward

                # End the episode early when we are done
                if done:
                    break

            # Pause the game after each episode for a short amount of time
            if args.timeout:
                for _ in range(int(int(args.timeout) / 16)):
                    env.render(episode, action, episode_reward, avg_episode_reward)

            episode_time = max(1, time.time() - episode_start_time)
            last_rewards.append(episode_reward)
            last_steps.append(step + 1)
            total_rewards.append(episode_reward)
            smoothed_rewards.append(np.mean(last_rewards).item())
            avg_episode_reward = np.mean(last_rewards)

            # Print statistics
            print(" Episode {} ".format(episode + 1).center(80, '*'))
            print("* Episode:  " + "Steps: " + "{}".format(step + 1).ljust(8, ' ')
                  + "Reward: " + "{:.2f}".format(episode_reward).ljust(8, ' ')
                  + "Last Reward: " + "{:.2f}".format(reward).ljust(8, ' '))
            print("* Average:  " + "Steps: " + "{:.2f}".format(np.mean(last_steps)).ljust(8, ' ')
                  + "Reward: " + "{:.2f}".format(avg_episode_reward).ljust(8, ' ')
                  + "Steps / sec: " + "{:.2f}".format((step + 1) / episode_time).ljust(8, ' '))
            print("".ljust(80, '*'), "\n")

            # Graph plotting
            if not args.headless:
                plt.clf()
                plt.title("Training..." if args.train else "Evaluating...")
                plt.xlabel("Episode")
                plt.ylabel("Reward")
                plt.plot(total_rewards)
                plt.plot(smoothed_rewards)
                plt.pause(0.0001)

            # Backup the agent and statistics every 1000th episode (except for the very last episode)
            if args.train and (episode + 1) % 1000 == 0 and (episode + 1) < args.episodes:
                save_state()

    except KeyboardInterrupt:
        pass

    # Make sure to save the final result
    if args.train:
        save_state()

    env.close()


if __name__ == "__main__":
    main()
