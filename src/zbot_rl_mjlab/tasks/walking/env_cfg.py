"""周期踏步与行走任务共用的环境配置类型。"""

from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg

TASK_ID = "Mjlab-Zbot-6dof-Periodic-Stepping"
WALKING_TASK_ID = "Mjlab-Zbot-6dof-Walking"
DISTURBED_WALKING_TASK_ID = "Mjlab-Zbot-6dof-Disturbed-Walking"
MIN_STEP_FREQUENCY = 0.2
MAX_STEP_FREQUENCY = 1.0


@dataclass
class ZbotEnvCfg(ManagerBasedRlEnvCfg):
    step_frequency_min: float = MIN_STEP_FREQUENCY
    step_frequency_max: float = MAX_STEP_FREQUENCY
    test_frequency: float | None = None
    frequency_sampling: str = "random"
