from omegaconf import OmegaConf
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from gov_rek.envs.governance.planar_kernel_wrapper import PlanarKernelGovernanceWrapper
from gov_rek.models.common.callback_logger import SaveOnBestTrainingRewardCallback


def ppo_trainer(config_path):
    configs = OmegaConf.load(config_path)

    kernels = configs.kernel_list
    if not configs.gov_rek:
        kernels = configs.mors_objective

    for idx, kernel_name in enumerate(kernels):
        kernels[idx] = (kernel_name, None)

    env = PlanarKernelGovernanceWrapper(
                        kernel_list = kernels,
                        size = configs.size,
                        gas = configs.gas,
                        her_goal = configs.her_goal,
                        num_blockers=configs.num_blockers,
                        default_world=configs.default_world,
                        delay=configs.delay,
                        )
    env = Monitor(env, configs.log_dir)
    ppo_callback = SaveOnBestTrainingRewardCallback(check_freq=configs.check_freq, log_dir=configs.log_dir)

    ppo_learner = PPO(configs.trainer_policy, env, verbose=configs.verbose)
    ppo_learner.learn(total_timesteps=configs.total_timesteps, callback=ppo_callback)


if __name__ == '__main__':
    # TODO: update the below `config_path` to select the correct experimentation configurations
    config_path = "gov_rek/models/configs/robustness_performance_experiment_mors.yaml"
    ppo_trainer(config_path)
