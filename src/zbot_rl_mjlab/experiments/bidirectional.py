"""独立的蛇形／双足双向指令实验。"""

from ..tasks.transformation.commanded import (
    TASK_ID,
    BidirectionalEnvCfg,
    configure_bidirectional,
)
from .transformation import TeacherTrainingExperiment


class SnakeBipedBidirectional(TeacherTrainingExperiment):
    """共用镜像训练基础，直接装配双向行为与指令观测。"""

    task_id = TASK_ID
    experiment_name = "zbot_transform_snake_biped_bidirectional"
    env_cfg_type = BidirectionalEnvCfg

    def edit(self, cfg, play):
        configure_bidirectional(cfg, play)
