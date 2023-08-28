"""result visualizer script which summarizes the governance experiments."""
from gov_rek.models.common.constants import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import OrderedDict
# import statements from matplotlib package font style setup
import matplotlib.font_manager
matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
# setting up the plotter function with 'Times New Roman' font style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# setting up the plotter function with 'Times New Roman' font style
sns.set(font="Times New Roman")
sns.set_style({'font.family': 'Times New Roman'})


def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')

def result_processor(trainer_logs_dict):
    """returns an OrderedDict with reward and episode mean values and confidence intervals for the plotting function"""
    experiment_result_list = []
    for names, paths in trainer_logs_dict.items():
        reward_nested_list = []
        eps_len_nested_list = []
        
        for path in paths:
            exp_df = pd.read_csv(path, comment='#')
            reward_nested_list.append(list(exp_df['r']))
            eps_len_nested_list.append(list(exp_df['l']))

        experiment_result_list.append((names, (reward_nested_list, eps_len_nested_list)))
    
    min_len = -1 # for creating a minimum length cutt-off for thee plot
    for (exp_name, (rwd_lst, eps_len_lst)) in experiment_result_list:
        for rwds in rwd_lst:
            if min_len < 0 or len(rwds) < min_len:
                min_len = len(rwds)
    # preparing a cropped nested experiment list for plotting
    intermediate_result_list = []
    for (exp_name, (rwd_lst, eps_lst)) in experiment_result_list:
        for idx, rwds in enumerate(rwd_lst):
            rwd_lst[idx] = rwds[:min_len]
        for idx, epss in enumerate(eps_lst):
            eps_lst[idx] = epss[:min_len]
        intermediate_result_list.append((exp_name, (rwd_lst, eps_lst)))
    
    final_result_list = []
    for (exp_name, (rwd_lst, eps_lst)) in intermediate_result_list:
        exp_rewards_arr = np.array(rwd_lst).T
        exp_eps_len_arr = np.array(eps_lst).T

        exp_rewards_mean_arr, exp_rewards_std_arr = moving_average(np.mean(exp_rewards_arr, axis=1), 25), moving_average(np.std(exp_rewards_arr, axis=1) , 25)
        exp_rewards_ci_lower, exp_rewards_ci_upper = exp_rewards_mean_arr - (3 * exp_rewards_std_arr) / np.sqrt(len(exp_rewards_std_arr)), \
                                                     exp_rewards_mean_arr + (3 * exp_rewards_std_arr) / np.sqrt(len(exp_rewards_std_arr))

        exp_eps_len_mean_arr, exp_eps_len_std_arr = moving_average(np.mean(exp_eps_len_arr, axis=1), 25), moving_average(np.std(exp_eps_len_arr, axis=1), 25)
        exp_eps_len_ci_lower, exp_eps_len_ci_upper = exp_eps_len_mean_arr - (3 * exp_eps_len_std_arr) / np.sqrt(len(exp_eps_len_std_arr)), \
                                                     exp_eps_len_mean_arr + (3 * exp_eps_len_std_arr) / np.sqrt(len(exp_eps_len_std_arr))
        
        final_result_list.append((exp_name, (exp_rewards_mean_arr, exp_rewards_ci_lower, exp_rewards_ci_upper,
                                            exp_eps_len_mean_arr, exp_eps_len_ci_lower, exp_eps_len_ci_upper)))

    
    return OrderedDict(final_result_list)


def _plotter(experiment_results_dict, y_limits, title):
    sns.set_style("whitegrid", {'axes.grid' : True,
                            'axes.edgecolor':'black'
                  })
    fig = plt.figure()
    sns.set(font="Times New Roman", rc={'figure.figsize':(19,12.0)})
    sns.set_style({'font.family': 'Times New Roman'})
    plt.clf()
    ax = fig.gca()
    colors = ["forestgreen", "purple", "orange", "blue", "crimson"]
    color_patch = []
    for color, (experiment_name, data_tuple) in zip(colors, experiment_results_dict.items()):
        # sns.lineplot(data=data, color=color, linewidth=2.5)
        ax.plot(range(0, len(data_tuple[0])), data_tuple[0], color=color, alpha=.6)
        ax.fill_between(range(0, len(data_tuple[0])), data_tuple[1], data_tuple[2], color=color, alpha=.3)
        color_patch.append(mpatches.Patch(color=color, label=experiment_name))
    
    ax.set_ylim([0, len(data_tuple[0])])
    ax.set_ylim(y_limits)
    plt.xlabel('Timesteps Duration ($\\times$ 32 Times)', fontsize=15)
    plt.ylabel('Average Reward Returns', fontsize=15)
    lgd=plt.legend(
    frameon=True, fancybox=True, \
    # prop={'size':14}, handles=color_patch, loc="best")
    prop={'weight':'bold', 'size':14}, handles=color_patch, loc="best")
    plt.title(title, fontsize=20)
    ax = plt.gca()
    
    # uncomment for adding custom tick values
    # ax.set_xticks([10, 20, 30, 40, 50])
    # ax.set_xticklabels([0.5, 1, 1.5, 2.5, 3.0])

    plt.setp(ax.get_xticklabels(), fontsize=15)
    plt.setp(ax.get_yticklabels(), fontsize=15)
    sns.despine()
    plt.tight_layout()
    plt.show()


def experiment_plotter(experiment_path, experiment_list_name, y_limits, title):
    experiment_paths = [f for f in experiment_path.iterdir() if f.is_dir()]
    experiment_paths.sort()
    trainer_logs_paths = []
    for exp_name, exp_pth in zip(experiment_list_name, experiment_paths):
        trainer_pth = [f / MONITOR_STR for f in exp_pth.iterdir() if f.is_dir()]
        trainer_logs_paths.append((exp_name, trainer_pth))

    trainer_logs_dict = OrderedDict(trainer_logs_paths)
    processed_result_dict = result_processor(trainer_logs_dict)

    _plotter(processed_result_dict, y_limits, title)

if __name__ == "__main__":
    # plotter function for plotting average reward returns and average episode lengths during the learning stage
    experiment_result_path = GOV_REK_ROBUSTNESS_PATH # GOV_REK_VS_MORS_OBJ_PATH
    experiment_list_name = ["One Blocker Object", "Two Blocker Object", "Three Blocker Object", "Four Blocker Object", "Five Blocker Object"]
    y_limits = [1., 4.5]
    experiment_plotter(experiment_result_path, experiment_list_name, y_limits, "Average Reward Returns for 5X5 Environment")
