import math
import pprint
import random
import collections
import numpy as np

random.seed(16)

def mutation_factor_generator(round_number):
    # return mutation variant flag for deciding
    # configuration mutate in which manner
    return math.floor(random.random()*round_number)

def gov_rek_simulator(kernel_name=None, kernel_config=None, timesteps=None):
    return (random.randint(12, 24), random.uniform(3, 9))

class PlanarGovernedTrainer():

    def __init__(self, num_hpo_rounds=3, total_budget=2e6, num_brackets=3, halving_eta=3):
        self.num_hpo_rounds = num_hpo_rounds
        self.total_budget = total_budget
        self.halving_eta = halving_eta
        self.num_brackets = num_brackets
        self.bracket_budget = total_budget / num_hpo_rounds / num_brackets
        self.surface_kernels = [('ellipsoid_kernel', None),
                              ('hyperboloid_kernel', None),
                              ('elliptic_paraboloid_kernel', None),
                              ('hyperbolic_paraboloid_kernel', None),
                              ('torus_kernel', None),
                              ('genus_ray_kernel', None),
                              ('diagonal_gradient', None),]
        self.budget_dict_list = self.get_budget_dict(self.bracket_budget, self.num_brackets, self.halving_eta,
                                                     self.total_budget, self.num_hpo_rounds)
        self.re_hpo_config_list = self.kernel_config_generator(self.budget_dict_list, self.surface_kernels, self.surface_kernels)

    def get_budget_dict(self, bracket_budget, num_brackets, halving_eta, total_budget, num_hpo_rounds):
        budget_dict_list = []
        total_budgets = [int(total_budget * ((i+1) / sum(range(num_hpo_rounds+1))) ) for i in range(num_hpo_rounds)]
        total_budgets.sort(reverse=True)

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
                conf_count_max = max(list(budget_bracket_dict.keys()))
                for (conf_count, conf_timesteps) in budget_bracket_dict.items():
                    conf_kernels = []
                    for i in range(conf_count):
                        if itr_idx == 0:
                            if conf_count_max == conf_count:
                                conf_kernels.append((random.choice(agent_kernel_list), conf_timesteps, 0))
                            else:
                                conf_kernels.append(('', conf_timesteps, -1))
                        elif itr_idx == 1:
                            if conf_count_max == conf_count:
                                conf_kernels.append((random.choice(spatial_kernel_list), conf_timesteps,
                                                     mutation_factor_generator(itr_idx + 1)))
                            else:
                                conf_kernels.append(('', conf_timesteps, -1))
                        else:
                            if conf_count_max == conf_count:
                                conf_kernels.append((random.choice(spatial_kernel_list + agent_kernel_list), conf_timesteps,
                                                     mutation_factor_generator(itr_idx + 1)))
                            else:
                                conf_kernels.append(('', conf_timesteps, -1))
                    config_budget_config_dict[conf_count] = conf_kernels
                budget_dist_dict[budget_bracket_idx] = config_budget_config_dict
            hp_round_conf_counter.append(conf_counter)
            itr_idx = itr_idx + 1
            kernel_config_list.append(budget_dist_dict)

        return kernel_config_list
   
    def run_random_trainer(self, re_hpo_config_list, top_k_count=3):
        # implemented to test prototyping logic for the multi-round hpo executor
        top_k_global = []
        for budget_dict in re_hpo_config_list:
            top_k_hpo_round = []
            # for every hpo round check whether previous configurations are present to be added into the current round's configs
            for (budget_bracket_idx, budget_bracket_dict) in budget_dict.items():
                budget_bracket_dict = collections.OrderedDict(sorted(budget_bracket_dict.items(), reverse=True))
                # print(budget_bracket_idx, budget_bracket_dict)
                top_k_hpo_bracket = []
                budget_bracket_conf_counts = list(budget_bracket_dict.keys())
                # print(budget_bracket_conf_counts)
                brack_conf_counter_idx = 0
                for (conf_count, model_confs) in budget_bracket_dict.items():
                    top_k_success_halving = []
                    # print(conf_count, model_confs)
                    for (kernel_name, timesteps, mut_flgs) in model_confs:
                        # model training start, intermediate model save and clean up with appropriate rename logic needed
                        top_k_success_halving.append((kernel_name, gov_rek_simulator(kernel_name, timesteps, mut_flgs)))
                    if (brack_conf_counter_idx+1) < len(budget_bracket_conf_counts):
                        # sorting logic implementation before needed
                        # print(brack_conf_counter_idx, budget_bracket_conf_counts[brack_conf_counter_idx+1])
                        top_k_success_halving = top_k_success_halving[:budget_bracket_conf_counts[brack_conf_counter_idx+1]]
                    brack_conf_counter_idx = brack_conf_counter_idx + 1
                    # print(top_k_success_halving)
                    top_k_hpo_bracket.extend(top_k_success_halving)
                top_k_hpo_round.extend(top_k_hpo_bracket)
            top_k_global.extend(top_k_hpo_round)
        # previous round configurations will be merged with the new configurations
        return top_k_global


def main():
    model_trainer = PlanarGovernedTrainer()    
    pprint.pprint(model_trainer.re_hpo_config_list)
    model_trainer.run_random_trainer(model_trainer.re_hpo_config_list)


if __name__ == '__main__':
    main()
