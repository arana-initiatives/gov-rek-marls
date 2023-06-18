from copy import deepcopy
# import statements
from gov_rek.envs.governance.utils import *
from gov_rek.envs.openai.road_env import GridRoadEnv
from gov_rek.envs.governance.planar_kernels import *

class PlanarKernelGovernanceWrapper(GridRoadEnv):

    def __init__(self, kernel_list, size, gas, randomize_world = False, \
                 default_world = True, num_blockers = 0, her_goal = False, greedy_fraction = 1.0):
        # kernel_list: list[tuple(str, dict)] specifies the reward kernels that are used in the environment
        # for example: kernel list[[(kernel types, function arg dict)]}], the function arg dict specifies
        super(PlanarKernelGovernanceWrapper, self).__init__(size, gas, randomize_world, \
                                                            default_world, num_blockers, her_goal)
        self.kernel_list = kernel_list
        self.greedy_fraction = greedy_fraction
        self.gov_kern_agent_one = self.get_gov_reks(self.kernel_list, {'world_map': self.world, 'agent_name': 1}, self.max_reward, self.size)
        self.gov_kern_agent_two = self.get_gov_reks(self.kernel_list, {'world_map': self.world, 'agent_name': 2}, self.max_reward, self.size)
        self.preward_flag_agent_one = True
        self.preward_flag_agent_two = True
        self.prev_obs_agent_one = deepcopy(self.world)
        self.prev_obs_agent_two = deepcopy(self.world)

    def get_gov_reks(self, kernel_list, args_dict, max_reward, size):
        kernel_signal = np.zeros((self.size, self.size))

        for (kernel_function, kernel_args) in kernel_list:
            
            func_args_dict = args_dict.copy()
            if kernel_args:
                func_args_dict.update(args_dict)

            if kernel_function == 'irregular_gradient_kernel':
                func_args_dict.pop('agent_name')
                kernel_signal = kernel_signal + normalize_rewards(irregular_gradient_kernel(**func_args_dict), max_reward)
            elif kernel_function == 'regular_gradient_kernel':
                func_args_dict.pop('agent_name')
                kernel_signal = kernel_signal + normalize_rewards(regular_gradient_kernel(**func_args_dict), max_reward)
            elif kernel_function == 'splitted_gradient_kernel':
                func_args_dict.pop('agent_name')
                kernel_signal = kernel_signal + normalize_rewards(splitted_gradient_kernel(**func_args_dict), max_reward)
            elif kernel_function == 'inverse_radial_kernel':
                kernel_signal = kernel_signal + normalize_rewards(inverse_radial_kernel(**func_args_dict), max_reward)
            elif kernel_function == 'squared_exponential_kernel':
                kernel_signal = kernel_signal + normalize_rewards(squared_exponential_kernel(**func_args_dict), max_reward)
            elif kernel_function == 'rational_quadratic_kernel':
                kernel_signal = kernel_signal + normalize_rewards(rational_quadratic_kernel(**func_args_dict), max_reward)
            elif kernel_function == 'periodic_kernel':
                kernel_signal = kernel_signal + normalize_rewards(periodic_kernel(**func_args_dict), max_reward)
            elif kernel_function == 'locally_periodic_kernel':
                kernel_signal = kernel_signal + normalize_rewards(locally_periodic_kernel(**func_args_dict), max_reward)
        
        return normalize_rewards(kernel_signal, max_reward)
    
    def governance_reward(self, observation, kernel, agent_name, prev_obs, greedy_fraction):
        agent_ind_x, agent_ind_y = np.where(observation == agent_name)
        prev_agent_ind_x, prev_agent_ind_y = np.where(prev_obs == agent_name)
        prev_obs = observation # saving the previous observation state
        # if agent is not present in the observation grid, return 0 reward
        if agent_ind_x[0] >= kernel.shape[0] or agent_ind_y[0] >= kernel.shape[1]: 
            return 0, prev_obs
        # if agent is still at present at the previous grid position, return 0 reward
        if agent_ind_x[0] == prev_agent_ind_x[0] and agent_ind_y[0] == prev_agent_ind_y[0]: 
            return 0, prev_obs

        kernel_reward_val = kernel[agent_ind_x[0]][agent_ind_y[0]] 

        if agent_name == 1:
            self.gov_kern_agent_one[agent_ind_x[0]][agent_ind_y[0]] = kernel[agent_ind_x[0]][agent_ind_y[0]] * greedy_fraction
        if agent_name == 2:
            self.gov_kern_agent_two[agent_ind_x[0]][agent_ind_y[0]] = kernel[agent_ind_x[0]][agent_ind_y[0]] * greedy_fraction

    def reset(self):
        self.prev_obs_agent_one = deepcopy(self.world)
        self.prev_obs_agent_two = deepcopy(self.world)
        self.preward_flag_agent_one = True
        self.preward_flag_agent_two = True
        self.gov_kern_agent_one = self.get_gov_reks(self.kernel_list, {'world_map': self.world, 'agent_name': 1}, self.max_reward, self.size)
        self.gov_kern_agent_two = self.get_gov_reks(self.kernel_list, {'world_map': self.world, 'agent_name': 2}, self.max_reward, self.size)
        return super().reset()

    def step(self, action):
        next_state, reward, done, info, agent = super().step(action)
        reward_new = 0
        if agent.name == 1:
            reward_new, self.prev_obs_agent_one = self.governance_reward(next_state, self.gov_kern_agent_one, agent.name, self.prev_obs_agent_one)
        elif agent.name == 2:
            reward_new, self.prev_obs_agent_two = self.governance_reward(next_state, self.gov_kern_agent_two, agent.name, self.prev_obs_agent_two)

        if agent.package == 3 and self.preward_flag_agent_one == True and agent.name == 1:
            reward_new = reward_new + round(self.max_reward / 4, 2)
            self.preward_flag_agent_one = False
        elif agent.package == 3 and self.preward_flag_agent_two == True and agent.name == 2:
            reward_new = reward_new + round(self.max_reward / 4, 2)
            self.preward_flag_agent_two = False

        reward_new  = round(reward_new, 2) + reward
        return next_state, reward_new, done, info


def main():
    gov_env_wrp = PlanarKernelGovernanceWrapper(
                        kernel_list = [ ('locally_periodic_kernel', None),
                                ('regular_gradient_kernel', None)
                                
                            ], size = 10, gas = 9,
                        )


if __name__ == '__main__':
    main()
