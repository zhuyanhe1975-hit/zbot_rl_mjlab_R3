"""Standalone command-conditioned snake locomotion; archived world-X stays intact."""

from ..site_snake_teacher import SiteSnakeTeacherRunner, site_snake_ppo_cfg
from ..tasks.snake.site_omni import site_snake_env_cfg
from .base import Experiment


class SnakeSiteOmniTeacher(Experiment):
    task_id = 'Mjlab-Zbot-6dof-Snake-Site-Omni-Teacher'
    experiment_name = 'zbot_snake_site_omni_teacher'
    train_envs = 8192
    max_iterations = 6000
    runner_cls = SiteSnakeTeacherRunner

    def create_env(self, play=False):
        return site_snake_env_cfg(play=play)

    def build_ppo(self):
        cfg = site_snake_ppo_cfg()
        cfg.experiment_name = self.experiment_name
        if self.max_iterations is not None:
            cfg.max_iterations = self.max_iterations
        return cfg
