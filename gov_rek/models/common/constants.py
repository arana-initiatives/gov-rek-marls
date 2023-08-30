from pathlib import Path

GOV_REK_VS_MORS_OBJ_PATH = Path("experiment_results/performance_comparison_task/govs_rek_vs_mors_obj")
GOV_REK_ROBUSTNESS_PATH = Path("experiment_results/robustness_measurement_task/gov_rek_vs_mors_blockers")
GOV_REK_SCALABILITY_PATH = Path("experiment_results/scalability_measurement_task")
GOV_REK_DRONE_SCALABILITY_PATH = Path("experiment_results/drone_scalability_measurement_task")
SCALABILITY_PER_EXP_CONFIG = Path("gov_rek/models/configs/scalability_performance_experiment.yaml")
DRONE_SCALABILITY_PER_EXP_CONFIG = Path("gov_rek/models/configs/drone_scalability_performance_experiment.yaml")
BASELINE_PER_EXP_CONFIG = Path("gov_rek/models/configs/baseline_performance_experiment_mors.yaml")
ROBUSTNESS_PER_EXP_CONFIG = Path("gov_rek/models/configs/robustness_performance_experiment.yaml")
MONITOR_STR = Path("monitor.csv")