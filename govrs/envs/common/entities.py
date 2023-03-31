import math
import random
import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod

class Entity(ABC):
    @abstractmethod
    def __init__(self):
        pass

class Agent(Entity):
    def __init__(self, name, gas, package):
        self.name = name
        self.gas = gas
        self.package = package

class DriverAgent(Agent):
    def __init__(self, name, gas, package):
        super(DriverAgent, self).__init__(name, gas, package)

class DroneAgent(Agent):
    def __init__(self, name, gas, package):
        super(DroneAgent, self).__init__(name, gas, package)

class World(Entity):
    def __init__(self, size, default_world, num_blockers = 0):
        self.size = size
        self.default_world = default_world
        self.num_blockers = num_blockers

class SimpleGridDroneWorld(World):
    def __init__(self, size, default_world = True, num_blockers = 0):
        super(SimpleGridDroneWorld, self).__init__(size, default_world, num_blockers)
        self.world = np.zeros((self.size, self.size, self.size),
                              dtype=int) # zyx indexing system
        self.goal_world = np.zeros((self.size, self.size, self.size),
                              dtype=int) # zyx indexing system
        
        self.world = self.populate_world(self.world, self.size,
                                         self.default_world, self.num_blockers)
        
        self.goal_world = deepcopy(self.world)
        self.goal_world = self.generate_goal_world(self.goal_world, self.size)

    def populate_world(self, world_arr, size, default_world, num_blockers):
        world_arr[0][0][0] = 1 # represents the first agent
        world_arr[size-1][size-1][size-1] = 4 # represents the final goal
        world_arr[0][size//2][size-1] = 2 # represents the second agent
        world_arr[0][size//2][0] = 3 # represents the package

        # masking the entire bottom field for drone movements
        floor_indices = np.where(world_arr[0,:,:] == 0)
        x_idx_flr, y_idx_flr = list(floor_indices[0]), list(floor_indices[1])
        for i in range(len(x_idx_flr)):
            world_arr[0][x_idx_flr[i]][y_idx_flr[i]] = 5


        if default_world == True: # num_blockers arguments irrelevant
            return world_arr
        else:
            if num_blockers > math.sqrt(size*size*size):
                num_blockers = int(math.sqrt(size*size*size))
            valid_indices = np.where(world_arr == 0)
            z_idx, y_idx, x_idx = list(valid_indices[0]), list(valid_indices[1]), \
                                    list(valid_indices[2])
            for i in range(num_blockers):
                blocker_ind = random.choice(range(len(x_idx)))
                # representation of the blocker object
                world_arr[z_idx[blocker_ind]][y_idx[blocker_ind]][x_idx[blocker_ind]] = 5
                x_idx.pop(blocker_ind)
                y_idx.pop(blocker_ind)
                z_idx.pop(blocker_ind)

        return world_arr
    
    def generate_goal_world(self, world_arr, size):
        agent_two_indices = np.where(world_arr == 2)
        world_arr[agent_two_indices[0][0]][agent_two_indices[1][0]] \
            [agent_two_indices[2][0]] = 0

        world_arr[size-1][size-1][size-1] = 2
        agent_one_indices = np.where(world_arr == 1) 
        package_indices = np.where(world_arr == 3)

        world_arr[agent_one_indices[0][0]][agent_one_indices[1][0]] \
            [agent_one_indices[2][0]] = 0
        world_arr[package_indices[0][0]][package_indices[1][0]] \
            [package_indices[2][0]] = 0

        return world_arr

class SimpleGridRoadWorld(World):
    def __init__(self, size, default_world = True, num_blockers = 0):
        super(SimpleGridRoadWorld, self).__init__(size, default_world, num_blockers)
        self.world = np.zeros((self.size, self.size), dtype=int)
        self.goal_world = np.zeros((self.size, self.size), dtype=int)

        self.world = self.populate_world(self.world, self.size,
                                         self.default_world, self.num_blockers)
        
        self.goal_world = deepcopy(self.world)
        self.goal_world = self.generate_goal_world(self.goal_world, self.size)

    def populate_world(self, world_arr, size, default_world, num_blockers):
        world_arr[0][0] = 1 # represents the first agent
        world_arr[size-1][size-1] = 4 # represents the final goal
        world_arr[size//2][size-1] = 2 # represents the second agent
        world_arr[size//2][0] = 3 # represents the package

        if default_world == True: # num_blockers arguments irrelevant
            return world_arr
        else:
            if num_blockers > math.sqrt(size*size)/2:
                num_blockers = int(math.sqrt(size*size)/2)
            valid_indices = np.where(world_arr == 0)
            y_idx, x_idx = list(valid_indices[0]), list(valid_indices[1])
            for i in range(num_blockers):
                blocker_ind = random.choice(range(len(x_idx)))
                world_arr[y_idx[blocker_ind]][x_idx[blocker_ind]] = 5 # represents the blocker object
                x_idx.pop(blocker_ind)
                y_idx.pop(blocker_ind)

        return world_arr

    def generate_goal_world(self, world_arr, size):
        agent_two_indices = np.where(world_arr == 2)
        world_arr[agent_two_indices[0][0]][agent_two_indices[1][0]] = 0

        world_arr[size-1][size-1] = 2
        agent_one_indices = np.where(world_arr == 1) 
        package_indices = np.where(world_arr == 3)

        world_arr[agent_one_indices[0][0]][agent_one_indices[1][0]] = 0
        world_arr[package_indices[0][0]][package_indices[1][0]] = 0

        return world_arr