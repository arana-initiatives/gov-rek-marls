import math


class GovernedMultiHyperbandTrainer():

    def __init__(self, num_hpo_rounds=2, total_budget=3e6, num_brackets=3, halving_eta=3):
        self.num_hpo_rounds = num_hpo_rounds
        self.total_budget = total_budget
        self.halving_eta = halving_eta
        self.num_brackets = num_brackets
        self.bracket_budget = total_budget / num_hpo_rounds / num_brackets
        self.budget_dist_dict = self.get_budget_dict(self.bracket_budget, self.num_brackets, self.halving_eta,
                                                     self.total_budget, self.num_hpo_rounds)

    def get_budget_dict(self, bracket_budget, num_brackets, halving_eta, total_budget, num_hpo_rounds):

        # hyperband algorithm's budget distributor algorithm implementation
        budget_dist_dict = {}
        for i in range(num_brackets-1, -1, -1):
            config_budget_tuples_dict = {}
            config_base_value = math.ceil( (total_budget * halving_eta**(i)) / (num_hpo_rounds * bracket_budget * (i+1)) )
            budget_base_value = bracket_budget / halving_eta**(i)
            for j in range(i+1):
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
                    print(conf_budget)
                    print('\n')

        return budget_dist_dict


def main():
    model_trainer = GovernedMultiHyperbandTrainer()
    print(model_trainer.budget_dist_dict)


if __name__ == '__main__':
    main()
