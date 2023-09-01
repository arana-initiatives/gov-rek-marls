"""Currently the HER implementation continue to show package dependency issues for our correct implementation."""
from omegaconf import OmegaConf
from stable_baselines3 import HerReplayBuffer, DQN
from stable_baselines3.common.monitor import Monitor
from gov_rek.envs.governance.planar_kernel_wrapper import PlanarKernelGovernanceWrapper
from gov_rek.envs.openai.road_env import GridRoadEnv
from gov_rek.models.common.callback_logger import SaveOnBestTrainingRewardCallback
from gym.wrappers import TimeLimit

def her_trainer(config_path):
    configs = OmegaConf.load(config_path)

    kernels = configs.kernel_list
    for idx, kernel_name in enumerate(kernels):
        kernels[idx] = (kernel_name, None)
    env = PlanarKernelGovernanceWrapper(
            kernel_list = kernels,
            size = configs.size,
            gas = configs.gas,
            her_goal = configs.her_goal,
        )
    if not configs.gov_rek:
        env = GridRoadEnv(
                size = configs.size,
                gas = configs.gas,
                her_goal = configs.her_goal,
            )

    env = Monitor(TimeLimit(env, max_episode_steps=env.max_episode_steps), configs.log_dir)
    sac_callback = SaveOnBestTrainingRewardCallback(check_freq=configs.check_freq, log_dir=configs.log_dir)

    sac_learner = DQN(
            configs.trainer_policy,
            env,
            replay_buffer_class=HerReplayBuffer,
            # Parameters for HER
            replay_buffer_kwargs=dict(
            n_sampled_goal=configs.n_sampled_goal,
            goal_selection_strategy=configs.goal_selection_strategy,
            ),
            verbose=configs.verbose,
        )

    sac_learner.learn(total_timesteps=configs.total_timesteps, callback=sac_callback)


if __name__ == '__main__':
    # TODO: update the below `config_path` to select the correct experimentation configurations
    config_path = "gov_rek/models/configs/baseline_performance_experiment_her.yaml"
    her_trainer(config_path)
