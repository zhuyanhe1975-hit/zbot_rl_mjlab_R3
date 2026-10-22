"""Standalone command-conditioned snake locomotion; archived world-X stays intact."""


import math
from dataclasses import asdict
from typing import cast

from mjlab.envs import mdp as common_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

import zbot_rl_mjlab.archived_site_snake as archive_task
import zbot_rl_mjlab.tasks.snake.base_omni as base_task
import zbot_rl_mjlab.tasks.snake.site_forward as forward_task
import zbot_rl_mjlab.tasks.snake.site_omni as site_task
from zbot_rl_mjlab.archived_site_snake import ArchivedSiteSnakeTeacherRunner
from zbot_rl_mjlab.experiments.omni_teacher import omni_teacher_ppo_cfg
from zbot_rl_mjlab.experiments.universal import universal_env_cfg
from zbot_rl_mjlab.omni_teacher import OmniTeacherRunnerCfg
from zbot_rl_mjlab.site_snake_teacher import SiteSnakeTeacherRunner
from zbot_rl_mjlab.tasks.snake import mdp as snake_mdp
from zbot_rl_mjlab.tasks.snake.site_omni import (
    SiteSnakeEnvCfg,
    command_exposure,
    joint_limit_penalty,
    site_height,
    site_tilt,
    site_tilt_limit_reached,
    site_tilt_penalty,
    velocity_error,
)
from zbot_rl_mjlab.tasks.universal.actions import SnakeCrawlJointTargetRateActionCfg
from zbot_rl_mjlab.tasks.universal.omni import reset_omni
from zbot_rl_mjlab.tasks.universal.privileged import privileged_state

from ..adaptive_site_snake import AdaptiveSiteSnakeTeacherRunner
from ..base_snake_teacher import BaseSnakeAdaptiveTeacherRunner
from ..site_forward_teacher import (
    ForwardSnakeTeacherRunner,
    RelaxedForwardSnakeTeacherRunner,
)
from .base import Experiment


def archived_site_snake_ppo_cfg():
    cfg = omni_teacher_ppo_cfg()
    cfg.class_name = 'zbot_rl_mjlab.archived_site_snake:ArchivedSiteSnakeTeacherRunner'
    # Successful snake iter1499:256/128 ELU, normalized Beta,1e-3/.002.
    cfg.algorithm.learning_rate = 1e-3
    cfg.algorithm.entropy_coef = .002
    cfg.console_mode = 'full'
    cfg.validation_interval_iterations = 0
    cfg.robust_validation_interval = 0
    return cfg


def archived_site_snake_env_cfg(play=False) -> SiteSnakeEnvCfg:
    cfg = cast(SiteSnakeEnvCfg, universal_env_cfg(play=play, cfg_cls=SiteSnakeEnvCfg))
    cfg.omni_site_frame = True
    cfg.archive_site_checkpoint = True
    cfg.motion_site = 'foot_0_sole'
    cfg.workspace_radius = 15.
    cfg.episode_length_s = 30.
    cfg.auto_reset = not play
    action = SnakeCrawlJointTargetRateActionCfg(**asdict(cfg.actions['joint_pos']))
    action.target_limit = math.pi / 3
    action.snake_target_limit = math.pi / 3
    action.max_target_speed = 2 * math.pi
    action.biped_residual_scale = None
    action.transition_end_joint_radius = None
    cfg.actions['joint_pos'] = action
    cfg.commands = {'transform': archive_task.ArchivedSiteSnakeCommandCfg(
        training=not play, mode='snake', task_observation_encoding='scalar',
        debug_vis=play,
        play_random_velocity=play,
    )}
    cfg.events = {
        'reset_scene': EventTermCfg(func=common_mdp.reset_scene_to_default, mode='reset'),
        'reset_transition': EventTermCfg(func=reset_omni, mode='reset'),
    }
    cfg.observations = {'actor': ObservationGroupCfg(
        {'privileged': ObservationTermCfg(func=privileged_state)},
        concatenate_terms=True, enable_corruption=False, nan_policy='disabled',
    )}
    cfg.scene.sensors = (*cfg.scene.sensors, ContactSensorCfg(
        name='snake_self_contact',
        primary=ContactMatch(mode='subtree', pattern='foot_0', entity='robot'),
        secondary=ContactMatch(mode='subtree', pattern='foot_0', entity='robot'),
        fields=('found', 'force', 'dist'), reduce='maxforce', history_length=cfg.decimation,
    ))
    cfg.sim.contact_sensor_maxmatch = 128
    cfg.rewards = {
        'track_forward_velocity': RewardTermCfg(
            func=archive_task.forward_tracking, weight=0., params={'std': .1}),
        'track_lateral_velocity': RewardTermCfg(
            func=archive_task.lateral_tracking, weight=0., params={'std': .1}),
        'track_angular_velocity': RewardTermCfg(
            func=archive_task.angular_tracking, weight=1., params={'std': .25}),
        'wz_velocity': RewardTermCfg(func=site_task.site_yaw_velocity_penalty, weight=0.),
        'site_tilt': RewardTermCfg(func=site_tilt_penalty, weight=-1.0),
        'site_tilt_angular_velocity': RewardTermCfg(
            func=site_task.site_tilt_angular_velocity_penalty, weight=0.),
        'joint_pos_limits': RewardTermCfg(
            func=joint_limit_penalty, weight=-1., params={'soft_limit_factor': .9}),
        'joint_target_limits': RewardTermCfg(
            func=joint_limit_penalty, weight=-1., params={'soft_limit_factor': .9, 'use_target': True}),
    }
    cfg.terminations = {} if play else {
        'self_collision': TerminationTermCfg(func=snake_mdp.self_collision),
        'site_tilt': TerminationTermCfg(func=site_tilt_limit_reached),
        'time_out': TerminationTermCfg(func=common_mdp.time_out, time_out=True),
    }
    cfg.metrics = {
        **{f'velocity_error_{name}': MetricsTermCfg(func=velocity_error, params={'axis': axis})
           for axis, name in enumerate(('vx', 'vy', 'wz'))},
        **{f'command_{name}_{direction}_fraction': MetricsTermCfg(
            func=command_exposure, params={'axis': axis, 'sign': sign})
           for axis, name in enumerate(('vx', 'vy', 'wz'))
           for sign, direction in ((1, 'positive'), (-1, 'negative'))},
        'site_height': MetricsTermCfg(func=site_height),
        'site_tilt_rad': MetricsTermCfg(func=site_tilt),
    }
    cfg.viewer.lookat = (0., 0., .05)
    return cfg


def site_snake_ppo_cfg() -> OmniTeacherRunnerCfg:
    cfg = omni_teacher_ppo_cfg()
    cfg.class_name = 'zbot_rl_mjlab.site_snake_teacher:SiteSnakeTeacherRunner'
    # Match the local G1 velocity task's Gaussian exploration parameters.
    cfg.actor.distribution_cfg = {
        'class_name': 'GaussianDistribution', 'init_std': 1., 'std_type': 'scalar',
    }
    cfg.clip_actions = 1.
    cfg.algorithm.learning_rate = 1e-3
    cfg.algorithm.entropy_coef = .01
    cfg.console_mode = 'full'
    cfg.validation_interval_iterations = 0
    cfg.robust_validation_interval = 0
    return cfg


def base_snake_env_cfg(play=False):
    cfg = site_snake_env_cfg(play)
    cfg.scene.entities['robot'].spec_fn = base_task.base_motion_spec
    cfg.motion_site = 'base_motion'
    if play:
        cfg.command_visual_height = .1
        cfg.command_visual_actual_velocity = True
    return cfg


def site_forward_env_cfg(play=False, relaxed=False):
    cfg = site_snake_env_cfg(play)
    cfg.commands['transform'] = forward_task.ForwardSnakeCommandCfg(**(
        asdict(cfg.commands['transform']) | {'play_velocity': (.1, 0., 0.)}
    ))
    # Freeze the paired experiment even when the main task's weights are tuned.
    cfg.rewards['track_forward_velocity'].weight = 4.
    cfg.rewards['track_lateral_velocity'].weight = 4.
    cfg.rewards['track_angular_velocity'].weight = 1.
    if relaxed:
        cfg.rewards['track_lateral_velocity'].weight = 0.
        cfg.rewards['track_angular_velocity'].weight = 0.
    cfg.metrics.update({
        **{f'head_velocity_{name}': MetricsTermCfg(func=forward_task.velocity_component, params={'axis': axis})
           for axis, name in enumerate(('vx', 'vy', 'wz'))},
        **{f'tracking_component_{name}': MetricsTermCfg(func=forward_task.tracking_component, params={'axis': axis})
           for axis, name in enumerate(('vx', 'vy', 'wz'))},
        'target_hard_limit_fraction': MetricsTermCfg(func=forward_task.target_hard_limit_fraction),
        'mean_abs_joint_velocity': MetricsTermCfg(func=forward_task.mean_abs_joint_velocity),
    })
    return cfg


def site_snake_env_cfg(play=False) -> site_task.SiteSnakeEnvCfg:
    cfg = cast(site_task.SiteSnakeEnvCfg, universal_env_cfg(play=play, cfg_cls=site_task.SiteSnakeEnvCfg))
    cfg.omni_site_frame = True
    cfg.motion_site = 'foot_0_sole'
    cfg.workspace_radius = 15.
    cfg.episode_length_s = 15.
    cfg.auto_reset = True
    action = SnakeCrawlJointTargetRateActionCfg(**asdict(cfg.actions['joint_pos']))
    action.target_limit = math.pi * 2 / 3
    action.snake_target_limit = math.pi * 2 / 3
    action.max_target_speed = 2 * math.pi
    action.biped_residual_scale = None
    action.transition_end_joint_radius = None
    cfg.actions['joint_pos'] = action
    cfg.commands = {'transform': site_task.SiteSnakeCommandCfg(
        training=not play, mode='snake', task_observation_encoding='scalar',
        debug_vis=play,
        play_random_velocity=play,
    )}
    cfg.events = {
        'reset_scene': EventTermCfg(func=common_mdp.reset_scene_to_default, mode='reset'),
        'reset_transition': EventTermCfg(func=reset_omni, mode='reset'),
    }
    cfg.observations = {'actor': ObservationGroupCfg(
        {'privileged': ObservationTermCfg(func=site_task.site_snake_state)},
        concatenate_terms=True, enable_corruption=False, nan_policy='disabled',
    )}
    cfg.scene.sensors = (*cfg.scene.sensors, ContactSensorCfg(
        name='snake_self_contact',
        primary=ContactMatch(mode='subtree', pattern='foot_0', entity='robot'),
        secondary=ContactMatch(mode='subtree', pattern='foot_0', entity='robot'),
        fields=('found', 'force', 'dist'), reduce='maxforce', history_length=cfg.decimation,
    ))
    cfg.sim.contact_sensor_maxmatch = 128
    cfg.rewards = {
        'alive': RewardTermCfg(func=common_mdp.is_alive, weight=0.2),
        'track_forward_velocity': RewardTermCfg(
            func=site_task.forward_tracking, weight=1.0),
        'track_lateral_velocity': RewardTermCfg(
            func=site_task.lateral_tracking, weight=1.0),
        'track_angular_velocity': RewardTermCfg(
            func=site_task.angular_tracking, weight=0.0),
        'action_rate': RewardTermCfg(func=common_mdp.action_rate_l2, weight=-.002),
        'joint_pose_deviation': RewardTermCfg(func=site_task.joint_pose_deviation_cost, weight=-.02),
        'site_tilt': RewardTermCfg(func=site_task.site_tilt_penalty, weight=-1.0),
        'joint_pos_limits': RewardTermCfg(
            func=site_task.joint_limit_penalty, weight=-1., params={'soft_limit_factor': .9}),
        'joint_target_limits': RewardTermCfg(
            func=site_task.joint_limit_penalty, weight=-1., params={'soft_limit_factor': .9, 'use_target': True}),
        'failure_reset': RewardTermCfg(func=site_task.failure_reset_cost, weight=-5.),
    }
    cfg.terminations = {
        'site_tilt': TerminationTermCfg(func=site_task.site_tilt_limit_reached),
    }
    if not play:
        cfg.terminations['self_collision'] = TerminationTermCfg(func=snake_mdp.self_collision)
        cfg.terminations['time_out'] = TerminationTermCfg(func=common_mdp.time_out, time_out=True)
    cfg.metrics = {
        **{f'velocity_error_{name}': MetricsTermCfg(func=site_task.velocity_error, params={'axis': axis})
           for axis, name in enumerate(('vx', 'vy', 'wz'))},
        **{f'command_{name}_{direction}_fraction': MetricsTermCfg(
            func=site_task.command_exposure, params={'axis': axis, 'sign': sign})
           for axis, name in enumerate(('vx', 'vy', 'wz'))
           for sign, direction in ((1, 'positive'), (-1, 'negative'))},
        'site_height': MetricsTermCfg(func=site_task.site_height),
        'site_tilt_rad': MetricsTermCfg(func=site_task.site_tilt),
    }
    cfg.viewer.lookat = (0., 0., .05)
    return cfg


class SiteSnakeExperiment(Experiment):
    train_envs = 8192
    max_iterations = 6000
    runner_cls = SiteSnakeTeacherRunner

    def create_env(self, play=False):
        return site_snake_env_cfg(play=play)

    def _build_ppo(self):
        cfg = site_snake_ppo_cfg()
        cfg.experiment_name = self.experiment_name
        if self.max_iterations is not None:
            cfg.max_iterations = self.max_iterations
        return cfg


class SnakeSiteOmniTeacher(SiteSnakeExperiment):
    task_id = 'Mjlab-Zbot-6dof-Snake-Site-Omni-Teacher'
    experiment_name = 'zbot_snake_site_omni_teacher'
    runner_cls = ArchivedSiteSnakeTeacherRunner

    def create_env(self, play=False):
        return archived_site_snake_env_cfg(play)

    def _build_ppo(self):
        cfg = archived_site_snake_ppo_cfg()
        cfg.experiment_name = self.experiment_name
        if self.max_iterations is not None:
            cfg.max_iterations = self.max_iterations
        return cfg


class SnakeSiteForwardTeacher(SiteSnakeExperiment):
    task_id = 'Mjlab-Zbot-6dof-Snake-Site-Forward-Teacher'
    experiment_name = 'zbot_snake_site_forward_teacher'
    runner_cls = ForwardSnakeTeacherRunner

    def create_env(self, play=False):


        return site_forward_env_cfg(play=play)

    def _build_ppo(self):
        cfg = super()._build_ppo()
        cfg.class_name = 'zbot_rl_mjlab.site_forward_teacher:ForwardSnakeTeacherRunner'
        return cfg


class SnakeSiteForwardRelaxedTeacher(SnakeSiteForwardTeacher):
    task_id = 'Mjlab-Zbot-6dof-Snake-Site-Forward-Relaxed-Teacher'
    experiment_name = 'zbot_snake_site_forward_relaxed_teacher'
    runner_cls = RelaxedForwardSnakeTeacherRunner

    def create_env(self, play=False):


        return site_forward_env_cfg(play=play, relaxed=True)

    def _build_ppo(self):
        cfg = super()._build_ppo()
        cfg.class_name = 'zbot_rl_mjlab.site_forward_teacher:RelaxedForwardSnakeTeacherRunner'
        return cfg



class SnakeSiteAdaptiveTeacher(SiteSnakeExperiment):
    task_id = 'Mjlab-Zbot-6dof-Snake-Site-Adaptive-Teacher'
    experiment_name = 'zbot_snake_site_adaptive_teacher'
    runner_cls = AdaptiveSiteSnakeTeacherRunner

    def _build_ppo(self):
        cfg = super()._build_ppo()
        cfg.class_name = 'zbot_rl_mjlab.adaptive_site_snake:AdaptiveSiteSnakeTeacherRunner'
        return cfg


class SnakeBaseAdaptiveTeacher(SnakeSiteAdaptiveTeacher):
    """Experiment20: keep experiment19's logic with the base as motion reference."""

    task_id = 'Mjlab-Zbot-6dof-Snake-Base-Adaptive-Teacher'
    experiment_name = 'zbot_snake_base_adaptive_teacher'
    runner_cls = BaseSnakeAdaptiveTeacherRunner

    def create_env(self, play=False):


        return base_snake_env_cfg(play)

    def _build_ppo(self):
        cfg = super()._build_ppo()
        cfg.class_name = 'zbot_rl_mjlab.base_snake_teacher:BaseSnakeAdaptiveTeacherRunner'
        return cfg
