import gym
import random
import numpy as np  
from gym import spaces
from collections import OrderedDict
from gym.envs.registration import EnvSpec

from gov_rek.envs.common.entities import *

class GridRoadEnv(gym.Env):

    def randomly_rotate_world(self, world_arr, goal_world_arr = None, her_goal = False):

        rot_world = world_arr
        world_two = np.rot90(rot_world)
        world_three = np.rot90(world_two)
        world_four = np.rot90(world_three)

        if her_goal:
            rot_goal_world = goal_world_arr
            goal_world_two = np.rot90(rot_goal_world)
            goal_world_three = np.rot90(goal_world_two)
            goal_world_four = np.rot90(goal_world_three)

        prob_rot_world = random.uniform(0, 1)

        if her_goal:
            if prob_rot_world > 0.25 and prob_rot_world <= 0.50:
                rot_world = world_two
                rot_goal_world = goal_world_two
            elif prob_rot_world > 0.5 and prob_rot_world <= 0.75:
                rot_world = world_three
                rot_goal_world = goal_world_three
            elif prob_rot_world > 0.75 and prob_rot_world <= 1:
                rot_world = world_four
                rot_goal_world = goal_world_four
            
            return rot_world, rot_goal_world
        else:
            if prob_rot_world > 0.25 and prob_rot_world <= 0.50:
                rot_world = world_two
            elif prob_rot_world > 0.5 and prob_rot_world <= 0.75:
                rot_world = world_three
            elif prob_rot_world > 0.75 and prob_rot_world <= 1:
                rot_world = world_four

        return rot_world
    
    def add_gas_constraint(self, gas_val, size):
        if (gas_val < size) or (gas_val > (2*(size-1)-1)):
            return random.randint(size, 2*(size-1))
        return gas_val

    def __init__(self, size, gas, randomize_world = False, \
                 default_world = True, num_blockers = 0, her_goal = False):
        # gas constraints limits for restricted env mobility
        # gas : {n_min: trunc(n/2), n_max: 2*(n-1)-1}
        self.size = size
        self.randomize_world = randomize_world
        self.num_blockers = num_blockers
        self.default_world = default_world
        self.grid_road = SimpleGridRoadWorld(self.size,
                                             self.default_world,
                                             self.num_blockers)
        self.world = self.grid_road.world
        self.her_goal = her_goal
        if self.her_goal:
            self.goal_world = self.grid_road.goal_world

        if self.randomize_world:
            self.world = self.randomly_rotate_world(self.world)
        elif self.randomize_world and self.her_goal:
            self.world, self.goal_world = self.randomly_rotate_world(self.world,
                                                                     self.goal_world,
                                                                     self.her_goal)


        self.action_space = spaces.Discrete(4)
        shape_0 = np.size(self.world, 0) # represents y-axis of the grid world
        shape_1 = np.size(self.world, 1) # represents x-axis of the grid world
        if self.her_goal:
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Box(low=0, high=5,
                                            shape=(shape_0 + 1, shape_1),
                                            dtype=np.int16),
                    "achieved_goal": spaces.Box(low=0, high=5,
                                            shape=(shape_0 + 1, shape_1),
                                            dtype=np.int16),
                    "desired_goal": spaces.Box(low=0, high=5,
                                            shape=(shape_0 + 1, shape_1),
                                            dtype=np.int16),
                }
            )
        else:
            self.observation_space = spaces.Box(low=0, high=5,
                                            shape=(shape_0 + 1, shape_1),
                                            dtype=np.int16)
        self.reward_range = (-10, 10)
        self.current_episode = 0
        self.success_episode = []
        # gas value constraint definition
        self.gas = self.add_gas_constraint(gas, self.size)
        # defining the driver agents in the environments
        # 3 integer value assigned instead of 0 while carrying the package
        self.agent_one = DriverAgent(1, self.gas, 0)
        # 3 integer value assigned instead of 0 while carrying the package
        self.agent_two = DriverAgent(2, int(self.gas*2.75), 0)
        self.spec = EnvSpec("GridRoadEnv-v0")

    def reset(self):
        # sequential game formulation:
        # each player agent moves one step when its their chance

        # instantiating agent 1 upon env reset
        self.agent_one = DriverAgent(1, self.gas, 0)
        # instantiating agent 2 upon env reset
        self.agent_two = DriverAgent(2, int(self.gas*2.75), 0)
        self.current_player = self.agent_one
        # 'P' represents playable game state,
        # 'W' represents package delivered state,
        # 'L' represents no delivery state
        self.state = 'P'
        self.current_step = 0
        self.max_step = int(self.gas*4.75)
        self.grid_road = SimpleGridRoadWorld(self.size,
                                             self.default_world,
                                             self.num_blockers)
        self.world = self.grid_road.world
        
        if self.her_goal:
            self.goal_world = self.grid_road.goal_world

        if self.randomize_world:
            self.world = self.randomly_rotate_world(self.world)

        if self.randomize_world:
            self.world = self.randomly_rotate_world(self.world)
        elif self.randomize_world and self.her_goal:
            self.world, self.goal_world = self.randomly_rotate_world(self.world,
                                                                     self.goal_world,
                                                                     self.her_goal)
            
        return self._next_observation()
    
    def _next_observation(self):
        obs = self.world
        data_to_add = [0] * np.size(self.world, 1)
        # adding current player's label in the obs, not permutation invariant
        data_to_add[0] = self.current_player.name
        obs = np.append(obs, [data_to_add], axis=0)
        # observation sample provided below for reference:
        # last row, represents 'data_to_add' vector
        # A 3x3 observation grid example for reference highlighted below
        # array([[1, 0, 0],
        #         [3, 0, 2],
        #         [0, 0, 4],
        #         [1, 0, 0]])

        if self.her_goal:
            goal_obs = self.goal_world
            goal_data_to_add = [0] * np.size(self.goal_world, 1)
            goal_data_to_add[0] = 2
            goal_obs = np.append(goal_obs, [goal_data_to_add], axis=0)
            return OrderedDict(
                [
                    ("observation", obs),
                    ("achieved_goal", obs),
                    ("desired_goal", goal_obs),
                ]
            )


        return obs

    def _take_action(self, action):
        # agent's name is matched to the array entries for index identification
        # 'current_player.name' should be updated alongside the array values
        current_pos = np.where(self.world == self.current_player.name)
        # sample current_pos example: (array([1]), array([0]))
        # the current agent must non-zero have gas in it
        if self.current_player.gas > 0:
            if action == 0:
                 # agent moving upwards
                next_pos = (current_pos[0] - 1, current_pos[1])

                if next_pos[0] >= 0 and int(self.world[next_pos]) == 0:
                    self.world[next_pos] = self.current_player.name
                    self.world[current_pos] = 0
                    # reducing the agent's gas by 1
                    self.current_player.gas = self.current_player.gas - 1

                elif next_pos[0] >= 0 and int(self.world[next_pos]) in (1, 2, 5):
                    # two agents and blocker object can't be at the same place
                    pass 

                elif next_pos[0] >= 0 and int(self.world[next_pos] == 3):
                    self.world[next_pos] = self.current_player.name
                    # package is also hidden now from other agent
                    self.current_player.package = 3 
                    self.world[current_pos] = 0
                    # reducing the agent's gas by 1
                    self.current_player.gas = self.current_player.gas - 1

                elif next_pos[0] >= 0 and int(self.world[next_pos] == 4):
                    # agent only allowed to transition at this position
                    # when it is having the package with itself
                    if self.current_player.package == 3:
                        # like 3, 4 numbers present at the goal
                        # but agent position only shown and represented
                        self.world[next_pos] = self.current_player.name
                        self.world[current_pos] = 0
                        # the episode ends at this state
                        self.state = 'W' 
                        # reducing the agent's gas by 1
                        self.current_player.gas = self.current_player.gas - 1
                    else:
                        pass


            elif action == 1:
                next_pos = (current_pos[0], current_pos[1] + 1)
                limit = np.size(self.world, 1)

                if next_pos[1] < limit and int(self.world[next_pos]) == 0:
                    self.world[next_pos] = self.current_player.name
                    self.world[current_pos] = 0
                    # reducing the agent's gas by 1
                    self.current_player.gas = self.current_player.gas - 1

                elif next_pos[1] < limit and int(self.world[next_pos]) in (1, 2, 5):
                    # two agents and blocker object can't be at the same place
                    pass

                elif next_pos[1] < limit and (int(self.world[next_pos]) == 3):
                    self.world[next_pos] = self.current_player.name
                    # package is also hidden now from other agent
                    self.current_player.package = 3 
                    self.world[current_pos] = 0
                    # reducing the agent's gas by 1
                    self.current_player.gas = self.current_player.gas - 1

                elif next_pos[1] < limit and int(self.world[next_pos] == 4):
                    # agent only allowed to transition at this position
                    # when it is having the package with itself
                    if self.current_player.package == 3:
                        # like 3, 4 numbers present at the goal
                        # but agent position only shown and represented
                        self.world[next_pos] = self.current_player.name
                        self.world[current_pos] = 0
                        # the episode ends at this state
                        self.state = 'W'
                        # reducing the agent's gas by 1
                        self.current_player.gas = self.current_player.gas - 1
                    else:
                        pass


            elif action == 2:
                next_pos = (current_pos[0] + 1, current_pos[1])
                limit = np.size(self.world, 0)

                if next_pos[0] < limit and int(self.world[next_pos]) == 0:
                    self.world[next_pos] = self.current_player.name
                    self.world[current_pos] = 0
                    # reducing the agent's gas by 1
                    self.current_player.gas = self.current_player.gas - 1

                elif next_pos[0] < limit and int(self.world[next_pos]) in (1, 2, 5):
                    # two agents and blocker object can't be at the same place
                    pass

                elif next_pos[0] < limit and (int(self.world[next_pos]) == 3):
                    self.world[next_pos] = self.current_player.name
                    # package is also hidden now from other agent
                    self.current_player.package = 3
                    self.world[current_pos] = 0
                    # reducing the agent's gas by 1
                    self.current_player.gas = self.current_player.gas - 1

                elif next_pos[0] < limit and int(self.world[next_pos] == 4):
                    # agent only allowed to transition at this position
                    # when it is having the package with itself
                    if self.current_player.package == 3:
                        # like 3, 4 numbers present at the goal
                        # but agent position only shown and represented
                        self.world[next_pos] = self.current_player.name
                        self.world[current_pos] = 0
                        # the episode ends at this state
                        self.state = 'W'
                        # reducing the agent's gas by 1
                        self.current_player.gas = self.current_player.gas - 1
                    else:
                        pass

            elif action == 3:
                next_pos = (current_pos[0], current_pos[1] - 1)

                if next_pos[1] >= 0 and int(self.world[next_pos]) == 0:
                    self.world[next_pos] = self.current_player.name
                    self.world[current_pos] = 0
                    # reducing the agent's gas by 1
                    self.current_player.gas = self.current_player.gas - 1

                elif next_pos[1] >= 0 and int(self.world[next_pos]) in (1, 2, 5):
                    # two agents and blocker object can't be at the same place
                    pass

                elif next_pos[1] >= 0 and (int(self.world[next_pos]) == 3):
                    self.world[next_pos] = self.current_player.name
                    # package is also hidden now from other agent
                    self.current_player.package = 3
                    self.world[current_pos] = 0
                    # reducing the agent's gas by 1
                    self.current_player.gas = self.current_player.gas - 1

                elif next_pos[1] >= 0 and int(self.world[next_pos] == 4):
                    # agent only allowed to transition at this position
                    # when it is having the package with itself
                    if self.current_player.package == 3:
                        # like 3, 4 numbers present at the goal
                        # but agent position only shown and represented
                        self.world[next_pos] = self.current_player.name
                        self.world[current_pos] = 0
                        # the episode ends at this state
                        self.state = 'W'
                        # reducing the agent's gas by 1
                        self.current_player.gas = self.current_player.gas - 1
                    else:
                        pass
            
        else:
            # if agent's gas is finished, it drops the package
            # and disappears from the current location
            if self.current_player.package == 3:
                self.world[current_pos] = self.current_player.package
                # agent dissappears from observation grid after package drop
            else:
                # if gas is finished, agent should dissappear
                self.world[current_pos] = 0 
        # if gas is empty for both agents
        # the episode stops at that instant
        if self.agent_one.gas == 0 and self.agent_two.gas == 0:
            self.state = 'L'


    def step(self, action):
        self._take_action(action)
        self.current_step += 1

        # note: to debug, uncomment the below statement
        # print(self.world) 

        if self.state == "W":
            # default reward value for default env
            # needs to be updated for governed env
            # based on agent interactions with reward shaped env
            reward = 2.5
            done = True
        elif self.state == 'L':
            reward = 0
            done = True
        elif self.state == 'P':
            # sparse reward encoding,
            # only rewarded when episode ends
            reward = 0 
            done = False

        if self.current_step >= self.max_step:
            print(f'New episode number {self.current_episode + 1}')
            done = True

        # agents object used for alternating the agent turns
        if self.current_player.name == 1:
            self.current_player = self.agent_two
        elif self.current_player.name == 2:
            self.current_player = self.agent_one

        if done:
            self.render_episode(self.state)
            self.current_episode += 1

        obs = self._next_observation()

        return obs, reward, done, {'state': self.state}, self.current_player

    def render_episode(self, win_or_lose):
        # storing the rendered episodes result in a file
        self.success_episode.append(
            'Success' if win_or_lose == 'W' else 'Failure')
        file = open('render.txt', 'a')
        file.write('----------------------------\n')
        file.write(f'Episode number {self.current_episode}\n')
        file.write(
            f'{self.success_episode[-1]} in {self.current_step} steps\n')
        file.close()
