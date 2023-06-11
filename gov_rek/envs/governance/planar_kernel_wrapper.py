# import statements
from copy import deepcopy
from gov_rek.envs.openai.road_env import GridRoadEnv
from gov_rek.envs.governance.planar_kernels import *

class PlanarKernelGovernanceWrapper(GridRoadEnv):

    def __init__(self, kernel_list, size, gas, randomize_world = False, \
                 default_world = True, num_blockers = 0, her_goal = False):
        # kernel_list: list[tuple(str, dict)] specifies the reward kernels that are used in the environment
        # for example: kernel list[[(kernel types, function arg dict)]}], the function arg dict specifies
        super(PlanarKernelGovernanceWrapper, self).__init__(size, gas, randomize_world, \
                                                            default_world, num_blockers, her_goal)
        self.kernel_list = kernel_list
        self.gov_kern_agent_one = self.get_gov_reks(kernel_list, self.world) # need for dict preparator
        self.gov_kern_agent_two = self.get_gov_reks(kernel_list, self.world) # need for dict preparator

    def get_gov_reks(self, kernel_list, ):
        base_kernel = deepcopy(getattr(kernel_list[0][0]))
        if len(kernel_list) == 1:
            return base_kernel
        
        '''
        kernel_list = kernel_list[1:]
        for (kernel, kernel_args) in kernel_list:
            base_kernel = np.add(self.gradient_kernel(self.world_start), self.circular_kernel(self.world_start, 2))
        '''

def main():
    gov_env_wrp = PlanarKernelGovernanceWrapper(
                        kernel_list = [ ('locally_periodic_kernel', None),
                                ('regular_gradient_kernel', None),
                                ('splitted_gradient_kernel', {'pos_slope_flag': False, 'slope_gradient': 0.1})
                            ], size = 10, gas = 9,
                        )

    print(gov_env_wrp.gov_kern_agent_one)
    print(gov_env_wrp.gov_kern_agent_two)
    


if __name__ == '__main__':
    main()
