import math
import random
import numpy as np

random.seed(16)

def prob_num_generator(round_number):
    # return mutation variant flag for deciding
    # configuration mutate in which manner
    return math.floor(random.random()*round_number)


class PlanarGovernedTrainer():

    def __init__(self, num_hpo_rounds=3, total_budget=5e6, num_brackets=3, halving_eta=3):
        self.num_hpo_rounds = num_hpo_rounds
        self.total_budget = total_budget
        self.halving_eta = halving_eta
        self.num_brackets = num_brackets
        self.bracket_budget = total_budget / num_hpo_rounds / num_brackets
        self.agent_kernels = [('inverse_radial_kernel', {'slope_gradient': 0.05}),
                              ('squared_exponential_kernel', {'size': 5.0, 'length_scale': 5.0}),
                              ('rational_quadratic_kernel', {'size': 5.0, 'length_scale': 5.0, 'alpha': 5.0}),
                              ('periodic_kernel', {'size':5.0, 'length_scale': 5.0, 'period': 2*np.pi}),
                              ('locally_periodic_kernel', {'size': 5.0, 'length_scale': 5.0, 'period':2*np.pi, 'alpha': 5.0})]
        self.spatial_kernels = [('irregular_gradient_kernel', {'pos_slope_flag': True, 'left_pos_vals': True, 'slope_gradient': 0.01}),
                                ('regular_gradient_kernel', {'size': 5.0, 'length_scale': 5.0, 'period':2*np.pi, 'alpha': 5.0}),
                                ('splitted_gradient_kernel', {'size': 5.0, 'length_scale': 5.0, 'period':2*np.pi, 'alpha': 5.0})]
        self.budget_dict_list = self.get_budget_dict(self.bracket_budget, self.num_brackets, self.halving_eta,
                                                     self.total_budget, self.num_hpo_rounds)
        self.re_hpo_config_list = self.kernel_config_generator(self.budget_dict_list, self.agent_kernels, self.spatial_kernels)

    def get_budget_dict(self, bracket_budget, num_brackets, halving_eta, total_budget, num_hpo_rounds):

        budget_dict_list = []
        total_budgets = [int(total_budget * (i+1)/sum(range(num_hpo_rounds+1))) for i in range(num_hpo_rounds)]
        for round_budget in total_budgets:
            # hyperband algorithm's budget distributor algorithm implementation
            budget_dist_dict = {}
            for i in range(num_brackets-1, -1, -1):
                config_budget_tuples_dict = {}
                config_base_value = math.ceil( (round_budget * halving_eta**(i)) / (bracket_budget * (i+1)) )
                budget_base_value = bracket_budget / halving_eta**(i)
                for j in range(i+1):
                    if math.floor(config_base_value/(halving_eta**j)) > 0:
                        config_budget_tuples_dict[math.floor(config_base_value/(halving_eta**j))] = int(budget_base_value*(halving_eta**(j-1)))
                budget_dist_dict[i] = config_budget_tuples_dict

            # applying budget correction for every bracket
            for s_val, config_val_dict in budget_dist_dict.items():
                config_budget = 0
                init_conf_val = 0
                init_conf_flg = True
                delt_budget = 0
                for (conf_val, conf_budget) in config_val_dict.items():
                    if init_conf_flg:
                        init_conf_flg = False
                        init_conf_val = conf_val
                    config_budget =  config_budget + conf_val * conf_budget

                if bracket_budget > config_budget:
                    delt_budget = bracket_budget - config_budget

                for (conf_val, conf_budget) in config_val_dict.items():
                    if conf_val == init_conf_val:
                        conf_budget =  int(conf_budget + delt_budget/conf_val)
                        budget_dist_dict[s_val][init_conf_val] = conf_budget

            budget_dict_list.append(budget_dist_dict)

        return budget_dict_list

    def kernel_config_generator(self, budget_dict_list, agent_kernels, spatial_kernels):
        # generate base kernel configurations for the inputted multi-round hyperband budget dict
        # the final kernel configuration list has the below stated sample data structure
        # kernel_config_list: list[dict: {dict: {list[(tuple)]}}] specifies the trainer configuration blueprint
        # for example: round list[bracket dict: {config dict: kernel configs[(base kernel name, budget value, mutation flag)]}]

        kernel_config_list = []
        agent_kernel_list = [i[0] for i in agent_kernels]
        spatial_kernel_list = [i[0] for i in spatial_kernels]
        itr_idx = 0
        hp_round_conf_counter = []
        for budget_dict in budget_dict_list:
            budget_dist_dict = {}
            conf_counter = 0
            for (budget_bracket_idx, budget_bracket_dict) in budget_dict.items():
                config_budget_config_dict = {}
                for (conf_count, conf_timesteps) in budget_bracket_dict.items():
                    conf_counter = conf_count + conf_counter
                    conf_kernels = []
                    for i in range(conf_count):
                        if itr_idx == 0:
                            conf_kernels.append((random.choice(agent_kernel_list), conf_timesteps, 0))
                        elif itr_idx == 1:
                            conf_kernels.append((random.choice(spatial_kernel_list), conf_timesteps,
                                                 prob_num_generator(itr_idx + 1)))
                        else:
                            conf_kernels.append((random.choice(spatial_kernel_list + agent_kernel_list), conf_timesteps,
                                                 prob_num_generator(itr_idx + 1)))
                    config_budget_config_dict[conf_count] = conf_kernels
                budget_dist_dict[budget_bracket_idx] = config_budget_config_dict
            hp_round_conf_counter.append(conf_counter)
            itr_idx = itr_idx + 1
            kernel_config_list.append(budget_dist_dict)

        return kernel_config_list

def main():
    model_trainer = PlanarGovernedTrainer()
    # print(model_trainer.budget_dict_list)
    import pprint
    pprint.pprint(model_trainer.re_hpo_config_list)

if __name__ == '__main__':
    main()
