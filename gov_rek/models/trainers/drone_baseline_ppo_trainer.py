from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gov_rek.envs.governance.surface_kernel_wrapper import SurfaceKernelGovernanceWrapper
from gov_rek.models.common.callback_logger import SaveOnBestTrainingRewardCallback
from gov_rek.models.common.constants import *
from gov_rek.envs.common.env_plotter import render_drone_agent

def ppo_trainer(config_path, file_name, render_episodes=False):
    configs = OmegaConf.load(config_path)

    kernels = configs.kernel_list
    if not configs.gov_rek:
        kernels = configs.mors_objective

    for idx, kernel_name in enumerate(kernels):
        kernels[idx] = (kernel_name, None)

    env = SurfaceKernelGovernanceWrapper(
                        kernel_list = kernels,
                        size = configs.size,
                        gas = configs.gas,
                        her_goal = configs.her_goal,
                        num_blockers=configs.num_blockers,
                        default_world=configs.default_world,
                        delay=configs.delay,
                        randomize_world=configs.randomize_world,
                        greedy_fraction=configs.greedy_fraction,
                        )
    env = Monitor(env, configs.log_dir)
    ppo_callback = SaveOnBestTrainingRewardCallback(check_freq=configs.check_freq, log_dir=configs.log_dir)

    ppo_learner = PPO(configs.trainer_policy, env, verbose=configs.verbose)
    ppo_learner.learn(total_timesteps=configs.total_timesteps, callback=ppo_callback)

    if render_episodes:
        ppo_simple_obs_list = []
        result_test = []
        env = SurfaceKernelGovernanceWrapper(
                        kernel_list = kernels,
                        size = configs.size,
                        gas = configs.gas,
                        her_goal = configs.her_goal,
                        num_blockers=configs.num_blockers,
                        default_world=configs.default_world,
                        delay=configs.delay,
                        randomize_world=configs.randomize_world,
                        greedy_fraction=configs.greedy_fraction,
                        )
        obs = env.reset()
        for itr in range(0,5):
            for i in range(200):
                action, _states = ppo_learner.predict(obs)
                obs, reward, done, info = env.step(action)
                ppo_simple_obs_list.append(obs)
                if done:
                    result_test.append(info['state'])

        render_drone_agent(ppo_simple_obs_list, file_name)


if __name__ == '__main__':
    # TODO: update the below `config_path` to select the correct experimentation configurations
    config_path = DRONE_SCALABILITY_PER_EXP_CONFIG # SCALABILITY_PER_EXP_CONFIG, # BASELINE_PER_EXP_CONFIG, # ROBUSTNESS_PER_EXP_CONFIG
    ppo_trainer(config_path, file_name='test_drone_render_plot.mp4', render_episodes=True)
