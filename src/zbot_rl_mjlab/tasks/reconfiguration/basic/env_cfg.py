"""Experiment 1: straight, prone 180-degree chain learns passive head-tail capture."""

import math
from dataclasses import dataclass

from mjlab.envs import mdp as common_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from ....robot import JOINT_NAMES
from ...transformation.actions import JointTargetRateAction, JointTargetRateActionCfg
from ...transformation.env_cfg import TransformationEnvCfg, transformation_scene_cfg
from . import mdp
from .model import head_tail_spec


@dataclass(kw_only=True)
class DockingActionCfg(JointTargetRateActionCfg):
    def build(self, env):
        return DockingAction(self, env)


class DockingAction(JointTargetRateAction):
    def process_actions(self, actions):
        mdp.state(self._env).begin_control_step()
        super().process_actions(actions)

    def apply_actions(self):
        super().apply_actions()
        mdp.state(self._env).substep()


@dataclass
class ReconfigurationEnvCfg(TransformationEnvCfg):
    head_tail_docking: bool = True
    # Explicit, uncalibrated prototype assumptions, not measured magnet specs.
    magnetic_model_calibrated: bool = False
    magnetic_range: float = .04
    magnetic_force: float = 8.
    magnetic_linear_damping: float = .4
    magnetic_angular_stiffness: float = .15
    magnetic_angular_damping: float = .02
    capture_distance: float = .004
    capture_angle: float = math.radians(8)
    capture_speed: float = .10
    capture_angular_speed: float = 1.
    capture_hold_s: float = .03
    release_distance: float = .02
    recapture_cooldown_s: float = .15
    break_force: float = 80.
    break_torque: float = 4.
    allowed_dock_quarters: tuple[int, ...] = (0, 1, 2, 3)
    success_hold_s: float = .5
    workspace_radius: float = 3.


def head_tail_env_cfg(play=False):
    cfg = transformation_scene_cfg(play, cfg_cls=ReconfigurationEnvCfg, action_cfg_cls=DockingActionCfg)
    robot = cfg.scene.entities['robot']
    robot.spec_fn = head_tail_spec
    robot.init_state.pos = (0., 0., .0555)
    robot.init_state.rot = (2**-.5, 0., -(2**-.5), 0.)
    robot.init_state.joint_pos = dict.fromkeys(JOINT_NAMES, 0.)
    cfg.scene.num_envs = 1 if play else 2048
    cfg.episode_length_s = 25.
    cfg.sim.mujoco.timestep = .005
    cfg.sim.mujoco.iterations = 30
    cfg.sim.njmax = 1000
    action = cfg.actions['joint_pos']
    action.target_limit = math.pi
    action.max_target_speed = math.pi
    cfg.observations = {name: ObservationGroupCfg(
        {'reconfiguration': ObservationTermCfg(func=mdp.observations)},
        concatenate_terms=True, enable_corruption=False,
    ) for name in ('actor', 'critic')}
    cfg.commands = {}
    cfg.events = {
        'reset_scene': EventTermCfg(func=common_mdp.reset_scene_to_default, mode='reset'),
        'reset_docking': EventTermCfg(func=mdp.reset_docking, mode='reset'),
    }
    cfg.terminations = {
        'unsafe': TerminationTermCfg(func=mdp.unsafe),
        'time_out': TerminationTermCfg(func=common_mdp.time_out, time_out=True),
    }
    if not play:
        cfg.terminations['success'] = TerminationTermCfg(func=mdp.success)
    cfg.rewards = {
        'end_distance': RewardTermCfg(func=mdp.distance_reward, weight=2.),
        'end_alignment': RewardTermCfg(func=mdp.alignment_reward, weight=.5),
        'end_distance_velocity': RewardTermCfg(func=mdp.distance_velocity_reward, weight=1.),
        'end_alignment_velocity': RewardTermCfg(func=mdp.alignment_velocity_reward, weight=.5),
        'approach_velocity': RewardTermCfg(func=mdp.approach_velocity_cost, weight=-.1),
        'connected': RewardTermCfg(func=mdp.connected_reward, weight=2.),
        'success': RewardTermCfg(func=mdp.success_reward, weight=5.),
        'action_rate': RewardTermCfg(func=common_mdp.action_rate_l2, weight=-.002),
        'joint_velocity': RewardTermCfg(func=mdp.joint_velocity_cost, weight=-.001),
        'limit_proximity': RewardTermCfg(func=mdp.limit_proximity_cost, weight=-.5, params={'margin': .3}),
    }
    cfg.metrics = {name: MetricsTermCfg(func=func) for name, func in {
        'end_distance_m': mdp.end_distance,
        'end_angle_rad': mdp.end_angle,
        'end_normal_score': mdp.alignment_reward,
        'end_closing_speed_m_s': mdp.distance_closing_speed,
        'end_alignment_rate_rad_s': mdp.alignment_closing_rate,
        'relative_speed_m_s': mdp.relative_speed,
        'connected_fraction': mdp.connected_reward,
        'success_fraction': mdp.success,
        'connection_count': mdp.connection_count,
        'break_count': mdp.break_count,
        'weld_force_n': mdp.weld_force,
        'weld_torque_nm': mdp.weld_torque,
    }.items()}
    cfg.viewer.lookat = (-.25, 0., .08)
    cfg.viewer.distance = 1.2
    return cfg
