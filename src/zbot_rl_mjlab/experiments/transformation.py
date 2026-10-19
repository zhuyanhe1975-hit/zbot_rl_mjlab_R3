"""两个变形方向的实验配方与训练设置。"""

from ..rl import RobustSnakeToBipedRunner, SnakeToBipedHotRewardRunner
from ..tasks.transformation.actions import (
    JointTargetRateActionCfg,
    MirroredJointTargetRateActionCfg,
)
from ..tasks.transformation.env_cfg import (
    TASK_ID,
    BipedToSnakeEnvCfg,
    transformation_env_cfg,
)
from ..tasks.transformation.presets import (
    configure_arc,
    configure_beta,
    configure_mirrored_standing,
    configure_teacher,
)
from ..tasks.transformation.robust import ROBUST_TASK_ID, RobustSnakeToBipedEnvCfg
from ..tasks.transformation.snake_to_biped import (
    ARC_TASK_ID,
    BETA_TASK_ID,
    SnakeToBipedEnvCfg,
    snake_to_biped_env_cfg,
)
from ..tasks.transformation.snake_to_biped import (
    TASK_ID as REVERSE_TASK_ID,
)
from ..tasks.transformation.teacher_observations import TEACHER_TASK_ID
from .base import Experiment


class TransformationExperiment(Experiment):
    """共享变形训练默认值，由配置类型和工厂选择具体方向。"""

    supports_walking_addons = False
    env_cfg_type = BipedToSnakeEnvCfg
    action_cfg_type = JointTargetRateActionCfg
    compact_standing = False

    def create_env(self, play=False):
        return transformation_env_cfg(play=play, cfg_cls=self.env_cfg_type)

    def build_ppo(self):
        cfg = super().build_ppo()
        cfg.actor.distribution_cfg = {
            "class_name": "GaussianDistribution", "init_std": 0.5,
            "std_type": "scalar", "std_range": (0.1, 1.0),
        }
        cfg.algorithm.entropy_coef = 0.005
        return cfg


class BipedToSnake(TransformationExperiment):
    task_id = TASK_ID
    experiment_name = "zbot_transform_biped_to_snake"


class SnakeToBiped(TransformationExperiment):
    task_id = REVERSE_TASK_ID
    experiment_name = "zbot_transform_snake_to_biped"
    env_cfg_type = SnakeToBipedEnvCfg
    runner_cls = SnakeToBipedHotRewardRunner

    def create_env(self, play=False):
        return snake_to_biped_env_cfg(
            play=play, cfg_cls=self.env_cfg_type,
            action_cfg_cls=self.action_cfg_type, compact=self.compact_standing,
        )


class SnakeToBipedBeta(SnakeToBiped):
    """显式有界 Beta 候选方案；保留原 Gaussian 任务。"""

    task_id = BETA_TASK_ID
    experiment_name = "zbot_transform_snake_to_biped_beta"
    train_envs = 8192
    max_iterations = 4000

    def build_ppo(self):
        cfg = super().build_ppo()
        cfg.actor.distribution_cfg = {
            "class_name": "BetaDistribution", "action_range": (-1.0, 1.0),
        }
        return cfg

    def edit(self, cfg, play):
        configure_beta(cfg, play)


class SnakeToBipedArc(SnakeToBipedBeta):
    """从与站立目标同分支的低位弧形姿态探索。"""

    task_id = ARC_TASK_ID
    experiment_name = "zbot_transform_snake_to_biped_arc"

    def edit(self, cfg, play):
        configure_arc(cfg, play)


class MirroredStandingExperiment(SnakeToBipedBeta):
    """单向鲁棒起立与双向教师共用的镜像配方。"""

    env_cfg_type = RobustSnakeToBipedEnvCfg
    action_cfg_type = MirroredJointTargetRateActionCfg
    compact_standing = True
    runner_cls = RobustSnakeToBipedRunner
    max_iterations = 2000

    def edit(self, cfg, play):
        configure_mirrored_standing(cfg, play)

    def build_ppo(self):
        cfg = super().build_ppo()
        cfg.resume = False
        cfg.algorithm.learning_rate = 1e-4
        cfg.algorithm.entropy_coef = 0.001
        return cfg


class SnakeToBipedRobust(MirroredStandingExperiment):
    task_id = ROBUST_TASK_ID
    experiment_name = "zbot_transform_snake_to_biped_robust"


class TeacherTrainingExperiment(MirroredStandingExperiment):
    """教师共享物理单位缩放及训练参数，观测由具体任务安装。"""

    def build_ppo(self):
        cfg = super().build_ppo()
        cfg.actor.obs_normalization = False
        cfg.critic.obs_normalization = False
        cfg.algorithm.learning_rate = 3e-4
        cfg.algorithm.entropy_coef = 0.005
        return cfg


class SnakeToBipedTeacher(TeacherTrainingExperiment):
    """完整状态教师策略，供后续学生策略学习。"""

    task_id = TEACHER_TASK_ID
    experiment_name = "zbot_transform_snake_to_biped_teacher"

    def edit(self, cfg, play):
        configure_teacher(cfg, play)
