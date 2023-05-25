## Governed Reward Shaping for MARLS

This repository contains environments, governance wrappers and implemented models for different MARL system architectures.
And, the corresponding experiments and jupyter notebooks for result replication across different sparse environment practical use-cases.

## Table of Contents

* [OpenAI Gym Package Delivery Environment Description](#openai-gym-package-delivery-environment-description)
  * [Environment Action Space](#environment-action-space)
  * [Environment Observation Space](#environment-observation-space)
  * [Environment Dynamics](#environment-dynamics)
  * [Sparse Rewards Scheme](#sparse-rewards-scheme)
  * [Environment Parameters Details](#environment-parameters-details)
* [Developer Execution Instructions](#developer-execution-instructions)

### OpenAI Gym Package Delivery Environment Description

The multi-agent package delivery environments simulate _last mile_ crowd-sourced package delivery in the simple and abstract sequential game-like settings.
Here, the two agents _(either drones or vehicles)_ need to learn cooperation for delivering packages successfully in a highly sparse setting, where rewards are only given upon package sharing and goal delivery.
The environments provide both grid road and drone network separate layouts of flexible sizes which include hindrance obstacles.
For both environments, in implementation the agents operate sequentially and the observation space is completely observable.
But, practically even in such settings on very high grid dimensions the cooperation and package learning is hard with current existing techniques.

#### Environment Action Space
  
In the 2d grid road simulation environment, vehicles have the following four discrete action space:

`V={ Turn Left, Turn Right, Go Up, Go Down }`

These actions allow the two vehicles to move across the 2D grid road environment.

In the 2d grid road simulation environment, vehicles have the following six discrete action space:

`D={ Turn Left, Turn Right, Go Up Y-Axis, Go Down Y-Axis, Go Up Z-Axis, Go Down Z-Axis}`

These actions allow the two vehicles to move across the 3D grid space environment.
  
#### Environment Observation Space

There are observation versions for both the grid road and drone space environments, namely: i.) `Regular Observation Space`, ii.) `Goal Based Observation Space`.
For regular observation space, the observation space is of the same size as the road grid or space volume generated for the environment respectively.
In case of goal based observation space, the observation space comprises of an `OrderedDict` containing _{"observation", "achieved_goal", "desired_goal"}_ keys, where each key provides the Box observation space of the same size as the road grid or space volume generated for the environment respectively.
The second observation space is only generated when `her_goal` argument is `True` for utilizing the `HER` wrapper from _stable\_baselines_.

#### Environment Dynamics

These agents are specifically designed for the CTCE paradigm for system prototyping simplification.
The agents cannot collide and move outside the environment, and actions prompting such scenarios are non-responsive in both environments.
In the implementation the package is automatically picked up by the vehicle or drone when the agent reaches the package location.
In the intermediate stage the package is dropped when the gas/fuel of the first agent is finished.
And, for task completion stage the package is dropped automatically by the second agent when either it reaches the goal or in the worst case scenario its gas is finished.
Here, we focus specifically in a resource constraint setting where both agents does not have enough fuel to deliver the package, and both agents need to learn cooperation.
  
#### Sparse Rewards Scheme

The rewards are only assigned in two scenarios, namely: i.) `Package Sharing`, ii.) `Package Delivery`.
And, this highly sparse nature of the environment combined with sequential nature of the subtasks make this problem especially challenging.

#### Environment Parameters Details

For consistency purposes, both the 2D grid road and 3D grid space environments are parameterized by same arguments listed below:

* `size`: It specifies the size of the grid road and space volume for the respective environments.
* `randomize_world`: When `True` this argument randomly rotates the environment configuration during training and evaluation.
* `default_world`: When `True` this argument populates the simple default configuration with zero obstacles.
* `num_blockers`: This numeric argument populates specified obstacles randomly in the environments.
* `her_goal`: This argument changes the observation type into goal based environment observation style for compatibility with HER like algorithms.

### Developer Execution Instructions

For debugging the project and parallely interacting with the codebase follow below stated two steps:

* First, open the project in your IDE and set the project `PYTHONPATH` with command `export PYTHONPATH=${PYTHONPATH}:$/.`

* Second, simply run the command in your `conda` environment `/home/{**path**}/miniconda3/envs/gov-rs-marls/bin/python
/home/{**path**}/gov-rs-marls/govrs/envs/openai/road_env.py` to test specific module of your choice from the project while your `pwd` is `{some-base-path}/gov-rs-marls/`.
  * **Note:** In the above `python` command, `gov-rs-marls` is also the name of the `conda` environment in use, same as the `pwd` or repository name.
