# HSP1 - Reinforcement Learning

A machine learning project using **reinforcement learning** to play arcade games.

## Media

We created 2 different kinds of videos:
- **Final Trained Agents:** Contains short videos (max 30 seconds) of the best results per game/environment
- **Training Progress:**    Presents the games/environments, observation data, pre-processing techniques and shows how the agents evolved over time

**NOTICE: Training Progress videos are too big for git, they have been uploaded to dropbox. Please download them as soon as possible!**
Training Progress download: https://www.dropbox.com/s/e7w9hn3xcp2d5u4/Final.zip?dl=1

## Installation

For the installation of the Project follow these steps:

1. **Install Anaconda** https://www.anaconda.com/distribution/ (Python 3.7)

2. **Create a new environement** conda create -n _hsp_ python=3.7 anaconda (The name _hsp_ can be replaced by any other name of your choice)

3. **Activate the environement** conda activate reinforcement

4. **Install Keras** pip install keras

5. **Install OpenCV** pip install opencv-python

6. **Install Tensorflow (GPU Version)** conda install tensorflow-gpu

7. **Install Pytorch** conda install pytorch torchvision -c pytorch

8. **Install Gym** pip install gym

9. **Further Dependencies**
   - Box2D
   - pygame
   - keyboard

## Execution

For the execution of the machine learning process use the following arguments and be sure that you have installed all required packages.

# Possible arguments

| Argument        | Abreviation | Type   | Default          | Explanation                                                     |
| --------------- | ----------- | ------ | ---------------- | --------------------------------------------------------------- |
| `--episodes`    | `-e`        | int    | 300              | The total amount of episodes to train                           |
| `--steps`       | `-s`        | int    | 200              | The total amount of steps per episod                            |
| `--environment` | `-E`        | string | "CartPole-v1"    | The gym environment name                                        |
| `--seed`        | `-S`        | int    | int(time.time()) | The prng seed for all random actions                            |
| `--agent`       | `-a`        | string | "dqn"            | The name of the agent to use                                    |
| `--argsext`     | `-A`        | string | -                | The extended arguements for the environment (json)              |
| `--file`        | `-f`        | string | -                | The path to the model which should be loaded                    |
| `--train`       | `-t`        | action | -                | Whether to train a network or not                               |
| `--render`      | `-r`        | action | -                | Whether to render the game or not                               |
| `--headless`    | `-H`        | action | -                | Whether to write logs to tensorboard or local plotting          |
| `--record`      | `-R`        | action | -                | Whether to record the gameplay to a video file (.avi) or not    |
| `--g_index`     | `-g`        | int    | -1               | The index of the combination that is executed by the gridsearch |
| `--timeout`     | `-T`        | int    | 0                | Timeout in ms after each episode                                |

# Start execution

Navigate with the command line to the src-folder (**Don't forget to activate your conda environement!**) and try out the following commands:

## Training agents

The following commands starts training in the CartPole environement with the DQN agent:

> `python main.py --train --render -E CartPole-v1 -a dqn`

During the execution the results of the episodes are plotted:

![Image description](../doc/gfx/model_300.png)

After finishing or interrupting the training the current state of the agent is saved to the directory _/models_ and can be loaded for further training or test purposes.

## Loading saved agents

Load a previously trained model in test mode:

> `python main.py --file ../models/CartPole-v1/A2C/1575402068.244839/model_122.mdl --render -E CartPole-v1 -a a2c`

Load a previously trained model and progress with the training:

> `python main.py --train --file ../models/CartPole-v1/A2C/1575402068.244839/model_122.mdl --render -E CartPole-v1 -a a2c`

## Passing extended arguements to the environment:

NOTE: Extended Arguements are documented for each environment which supports them in _env_wrapper.py_

Load a previously trained model in test mode and pass extended arguements to the environment (mode 1 means play against another agent):

> `python main.py -f "../models/PongCustom/DQN/ServerTraining_11\model_53000.mdl" --render -E "PongCustom" -a dqn -A "{ 'mode': 1, 'agent': 'dqn', 'model': '../models/PongCustom/DQN/ServerTraining_12/model_70000.mdl' }"`

Playing as human against one of the trained DQN agents on CustomPong:

> `python main.py -e 200 -s 50000 -E "PongCustom" -f "../models/PongCustom/DQN/ServerTraining_12/model_70000.mdl" -A "{'mode': 3}" -a dqn -r -H -T 1000`

## Gridsearch - Automated search of hyperparameters

To make life a bit easier it is possible to define gridsearch combinations which will be used as parameters for the execution of a predefined command. For the usage of this functionality you have define a json file that describes the desired execution. One example:

    {
    	"command": "python main.py --train --render -E CartPole-v1 -a a2c -S 1337 -e 10",
    	"combinations":[
    		{
    			"ALPHA":"0.001",
    			"BETA":"0.001"
    		},
    		{
    			"ALPHA":"0.002",
    			"BATCH_SIZE": "64",
    			"UPDATE_RATE": "2"
    		}
    	]
    }

Within the attribute command you can define the everything relevant for the execion if the agent and within combinations you can add as many combinations as you want with all common parameters supported an agant. The results will be saved to the folder _/models/Gridsearch_.

### Supported parameters

| Parameter     | Type    |     |     |     |     |     |     | Parameter         | Type   |
| ------------- | ------- | --- | --- | --- | --- | --- | --- | ----------------- | ------ |
| `ALPHA`       | `float` |     |     |     |     |     |     | `LAYER_H1_SIZE`   | `int`  |
| `BETA`        | `float` |     |     |     |     |     |     | `LAYER_H2_SIZE`   | `int`  |
| `GAMMA`       | `float` |     |     |     |     |     |     | `EPSILON_DECAY`   | `int`  |
| `TAU`         | `float` |     |     |     |     |     |     | `TGT_UPDATE_RATE` | `int`  |
| `EPSILON_MAX` | `float` |     |     |     |     |     |     | `MEMORY_SIZE`     | `int`  |
| `EPSILON_MIN` | `float` |     |     |     |     |     |     | `MEMORY_FILL`     | `int`  |
| `ALPHA_DECAY` | `float` |     |     |     |     |     |     | `BATCH_SIZE`      | `int`  |
| `DOUBLE_DQN`  | `bool`  |     |     |     |     |     |     | `UPDATE_RATE`     | `int`  |
| `DUELING_DQN` | `bool`  |     |     |     |     |     |     | `PRIO_REPLAY`     | `bool` |
