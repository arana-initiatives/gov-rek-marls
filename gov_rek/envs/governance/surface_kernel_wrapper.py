# general import statements
import random
import numpy as np
from collections import OrderedDict
# environment related import statements
from gov_rek.envs.governance.utils import *
from gov_rek.envs.openai.drone_env import GridDroneEnv
from gov_rek.envs.governance.surface_kernels import *
from gov_rek.envs.governance.utils import *

class SurfaceKernelGovernanceWrapper(GridDroneEnv):

    def __init__(self, kernel_list, size, gas, randomize_world = False, \
                 default_world = True, num_blockers = 0, her_goal = False, greedy_fraction = 1.0, delay = True):
        # kernel_list: list[tuple(str, dict)] specifies the reward kernels that are used in the environment
        # for example: kernel list[[(kernel types, function arg dict)]}], the function arg dict specifies
        super(SurfaceKernelGovernanceWrapper, self).__init__(size, gas, randomize_world, \
                                                            default_world, num_blockers, her_goal, delay)
        self.kernel_list = kernel_list
        self.greedy_fraction = greedy_fraction
        self.gov_kern_agent_one = self.get_gov_reks({'world_map': self.world_start, 'agent_name': 1})
        # plot_surface_kernel(self.gov_kern_agent_one, title='Sample Kernel Plot Visualization')
        self.gov_kern_agent_two = self.get_gov_reks({'world_map': self.world_start, 'agent_name': 2})
        # plot_surface_kernel(self.gov_kern_agent_two, title='Sample Kernel Plot Visualization')
        self.preward_flag_agent_one = True
        self.preward_flag_agent_two = True
        self.prev_obs_agent_one = np.copy(self.world_start)
        self.prev_obs_agent_two = np.copy(self.world_start)

    def get_gov_reks(self, args_dict):
        # only global kernel support added for experiments
        kernel_signal = np.zeros((self.size, self.size))

        for (kernel_function, kernel_args) in self.kernel_list:
            func_args_dict = args_dict.copy()
            func_args_dict.pop('agent_name')
            if kernel_args is not None and "rotation_angles" in kernel_args:
                if kernel_function == 'ellipsoid_kernel':
                    kernel_signal = kernel_signal + normalize_rewards(ellipsoid_kernel(**func_args_dict), self.max_reward)
                    func_args_dict['world_map'] = kernel_signal
                    if kernel_args:
                        func_args_dict.update(kernel_args)
                    kernel_signal = rotate_surface(**func_args_dict)
                elif kernel_function == 'hyperboloid_kernel':
                    kernel_signal = kernel_signal + normalize_rewards(hyperboloid_kernel(**func_args_dict), self.max_reward)
                    func_args_dict['world_map'] = kernel_signal
                    if kernel_args:
                        func_args_dict.update(kernel_args)
                    kernel_signal = rotate_surface(**func_args_dict)
                elif kernel_function == 'elliptic_paraboloid_kernel':
                    kernel_signal = kernel_signal + normalize_rewards(elliptic_paraboloid_kernel(**func_args_dict), self.max_reward)
                    func_args_dict['world_map'] = kernel_signal
                    if kernel_args:
                        func_args_dict.update(kernel_args)
                    kernel_signal = rotate_surface(**func_args_dict)
                elif kernel_function == 'hyperbolic_paraboloid_kernel':
                    kernel_signal = kernel_signal + normalize_rewards(hyperbolic_paraboloid_kernel(**func_args_dict), self.max_reward)
                    func_args_dict['world_map'] = kernel_signal
                    if kernel_args:
                        func_args_dict.update(kernel_args)
                    kernel_signal = rotate_surface(**func_args_dict)
                elif kernel_function == 'torus_kernel':
                    kernel_signal = kernel_signal + normalize_rewards(torus_kernel(**func_args_dict), self.max_reward)
                    func_args_dict['world_map'] = kernel_signal
                    if kernel_args:
                        func_args_dict.update(kernel_args)
                    kernel_signal = rotate_surface(**func_args_dict)
                elif kernel_function == 'genus_ray_kernel':
                    kernel_signal = kernel_signal + normalize_rewards(torus_kernel(**func_args_dict), self.max_reward)
                    func_args_dict['world_map'] = kernel_signal
                    if kernel_args:
                        func_args_dict.update(kernel_args)
                    kernel_signal = rotate_surface(**func_args_dict)
                elif kernel_function == 'diagonal_gradient':
                    kernel_signal = kernel_signal + normalize_rewards(diagonal_gradient(**func_args_dict), self.max_reward)
                    func_args_dict['world_map'] = kernel_signal
                    if kernel_args:
                        func_args_dict.update(kernel_args)
                    kernel_signal = rotate_surface(**func_args_dict)
            else:
                if kernel_function == 'ellipsoid_kernel':
                    kernel_signal = kernel_signal + normalize_rewards(ellipsoid_kernel(**func_args_dict), self.max_reward)
                elif kernel_function == 'hyperboloid_kernel':
                    kernel_signal = kernel_signal + normalize_rewards(hyperboloid_kernel(**func_args_dict), self.max_reward)
                elif kernel_function == 'elliptic_paraboloid_kernel':
                    kernel_signal = kernel_signal + normalize_rewards(elliptic_paraboloid_kernel(**func_args_dict), self.max_reward)
                elif kernel_function == 'hyperbolic_paraboloid_kernel':
                    kernel_signal = kernel_signal + normalize_rewards(hyperbolic_paraboloid_kernel(**func_args_dict), self.max_reward)
                elif kernel_function == 'torus_kernel':
                    kernel_signal = kernel_signal + normalize_rewards(torus_kernel(**func_args_dict), self.max_reward)
                elif kernel_function == 'genus_ray_kernel':
                    kernel_signal = kernel_signal + normalize_rewards(torus_kernel(**func_args_dict), self.max_reward)
                    func_args_dict['world_map'] = kernel_signal
                    kernel_signal = kernel_signal + rotate_surface(**func_args_dict)
                elif kernel_function == 'diagonal_gradient':
                    kernel_signal = kernel_signal + normalize_rewards(diagonal_gradient(**func_args_dict), self.max_reward)
                    func_args_dict['world_map'] = kernel_signal
                    kernel_signal = kernel_signal + rotate_surface(**func_args_dict)
        
        return normalize_rewards(kernel_signal, self.max_reward)
    
    def governance_reward(self, observation, kernel, agent_name, prev_obs, greedy_fraction, her_goal):
        # based on her goal, extracting the observation from the ordered dict is needed
        # to make this governance wrapper compatible with the underlying road environment
        if her_goal:
            agent_ind_x, agent_ind_y, agent_ind_z = np.where(observation['observation'] == agent_name)
        else:
            agent_ind_x, agent_ind_y, agent_ind_z = np.where(observation == agent_name)
        
        prev_agent_ind_x, prev_agent_ind_y, prev_agent_ind_z = np.where(prev_obs == agent_name)

        if her_goal:
            prev_obs = observation['observation']
        else:
            prev_obs = observation # saving the previous observation state

        # if agent is not present in the observation grid, return 0 reward
        if agent_ind_x[0] >= kernel.shape[0] or agent_ind_y[0] >= kernel.shape[1] or agent_ind_z[0] >= kernel.shape[2]: 
            return 0, prev_obs
        # if agent is still at present at the previous grid position, return 0 reward
        if agent_ind_x[0] == prev_agent_ind_x[0] and agent_ind_y[0] == prev_agent_ind_y[0] and agent_ind_z[0] >= prev_agent_ind_z[0]: 
            return 0, prev_obs

        kernel_reward_val = kernel[agent_ind_x[0]][agent_ind_y[0]][agent_ind_z[0]]

        if agent_name == 1:
            self.gov_kern_agent_one[agent_ind_x[0]][agent_ind_y[0]][agent_ind_z[0]] = kernel[agent_ind_x[0]][agent_ind_y[0]][agent_ind_z[0]] * greedy_fraction
        if agent_name == 2:
            self.gov_kern_agent_two[agent_ind_x[0]][agent_ind_y[0]][agent_ind_z[0]] = kernel[agent_ind_x[0]][agent_ind_y[0]][agent_ind_z[0]] * greedy_fraction
        
        return round(kernel_reward_val, 2), prev_obs

    def reset(self):
        self.prev_obs_agent_one = np.copy(self.world_start)
        self.prev_obs_agent_two = np.copy(self.world_start)
        self.preward_flag_agent_one = True
        self.preward_flag_agent_two = True
        self.gov_kern_agent_one = self.get_gov_reks({'world_map': self.world_start, 'agent_name': 1})
        self.gov_kern_agent_two = self.get_gov_reks({'world_map': self.world_start, 'agent_name': 2})
        return super().reset()

    def step(self, action):
        next_state, reward, done, info, agent = super().step(action)
        reward_new = 0
        if agent.name == 1:
            reward_new, self.prev_obs_agent_one = self.governance_reward(next_state, self.gov_kern_agent_one, agent.name,
                                                                         self.prev_obs_agent_one, self.greedy_fraction, self.her_goal)
        elif agent.name == 2:
            reward_new, self.prev_obs_agent_two = self.governance_reward(next_state, self.gov_kern_agent_two, agent.name,
                                                                         self.prev_obs_agent_two, self.greedy_fraction, self.her_goal)

        if agent.package == 3 and self.preward_flag_agent_one == True and agent.name == 1:
            reward_new = reward_new + round(self.max_reward / 4, 2)
            self.preward_flag_agent_one = False
        elif agent.package == 3 and self.preward_flag_agent_two == True and agent.name == 2:
            reward_new = reward_new + round(self.max_reward / 4, 2)
            self.preward_flag_agent_two = False

        reward_new  = round(reward_new, 2) + reward
        return next_state, reward_new, done, info


def main():
    simple_obs_list = []
    simple_info_list = []
    env_simple = SurfaceKernelGovernanceWrapper(
                        kernel_list = [ ('hyperboloid_kernel', {"rotation_angles": (m.pi/4, 0, -m.pi/4)}),
                                ('diagonal_gradient', None)
                            ], size = 10, gas = 17, her_goal=True
                        )

    for e_ in range(3):
        obs = env_simple.reset()
        for i in range(200):
            action = random.randint(0, 4) # model_simple.predict(obs)
            obs, reward, done, info = env_simple.step(action)
            print(obs, reward, done, info) # , agent.name, agent.gas, agent.package, agent.picked)
            # print('\n')
            # simple_obs_list.append(obs)
            #simple_info_list.append(info)
            if done:
                print(info['state'])
                obs = env_simple.reset()
                # result_test.append(info['state'])
                break

if __name__ == '__main__':
    main()
