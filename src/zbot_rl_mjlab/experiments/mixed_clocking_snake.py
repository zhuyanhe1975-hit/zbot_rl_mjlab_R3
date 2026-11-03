"""Independent snake teacher for [270,270,180,90,90] mounting angles."""
from ..mixed_clocking_snake_teacher import MixedClockingSnakeTeacherRunner
from ..tasks.snake.mixed_clocking import mixed_clocking_spec
from .rz270_snake import SnakeRz270Teacher


class SnakeMixedClockingTeacher(SnakeRz270Teacher):
    task_id = 'Mjlab-Zbot-6dof-Snake-Mixed-270-270-180-90-90-Teacher'
    experiment_name = 'zbot_snake_mixed_270_270_180_90_90_teacher'
    runner_cls = MixedClockingSnakeTeacherRunner

    def edit(self, cfg, play):
        cfg.scene.entities['robot'].spec_fn = mixed_clocking_spec

    def _build_ppo(self):
        cfg = super()._build_ppo()
        cfg.class_name = 'zbot_rl_mjlab.mixed_clocking_snake_teacher:MixedClockingSnakeTeacherRunner'
        return cfg
