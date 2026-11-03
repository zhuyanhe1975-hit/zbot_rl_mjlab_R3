"""Independent site-frame snake experiment for the 270-degree assembly."""

from dataclasses import asdict

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ..rz270_snake_teacher import Rz270SnakeTeacherRunner
from ..tasks.snake import rz270_heading as heading_task
from ..tasks.snake.rz270 import reset_rz270_snake, rz270_spec
from .site_snake import SiteSnakeExperiment


class SnakeRz270Teacher(SiteSnakeExperiment):
    task_id = 'Mjlab-Zbot-6dof-Snake-RZ270-Teacher'
    experiment_name = 'zbot_snake_rz270_teacher'
    runner_cls = Rz270SnakeTeacherRunner

    def edit(self, cfg, play):
        cfg.scene.entities['robot'].spec_fn = rz270_spec
        cfg.events['reset_transition'].func = reset_rz270_snake
        cfg.commands['transform'] = heading_task.HeadingSnakeCommandCfg(**asdict(cfg.commands['transform']))
        cfg.observations['actor'].terms['privileged'].func = heading_task.heading_observation
        for name in ('track_forward_velocity', 'track_lateral_velocity', 'track_angular_velocity'):
            del cfg.rewards[name]
        cfg.rewards['head_forward_speed'] = RewardTermCfg(func=heading_task.head_forward_speed, weight=4., params={'stop_speed_scale': .05})
        cfg.rewards['heading_alignment'] = RewardTermCfg(func=heading_task.heading_alignment, weight=1.)
        cfg.rewards['site_tilt'].func = heading_task.head_tilt_angle_penalty
        cfg.rewards['head_lateral_speed'] = RewardTermCfg(func=heading_task.head_lateral_speed_penalty, weight=-5.)
        cfg.metrics = {
            'head_forward_speed_m_s': MetricsTermCfg(func=heading_task.head_forward_velocity),
            'forward_command_m_s': MetricsTermCfg(func=heading_task.forward_speed_command),
            'forward_error_m_s': MetricsTermCfg(func=heading_task.forward_speed_error),
            'heading_error_rad': MetricsTermCfg(func=heading_task.heading_error_abs),
            'target_heading_rad': MetricsTermCfg(func=heading_task.target_heading),
            'head_tilt_rad': MetricsTermCfg(func=heading_task.site_tilt),
            'head_lateral_speed_m_s': MetricsTermCfg(func=heading_task.head_lateral_speed),
        }

    def _build_ppo(self):
        cfg = super()._build_ppo()
        cfg.class_name = 'zbot_rl_mjlab.rz270_snake_teacher:Rz270SnakeTeacherRunner'
        return cfg
