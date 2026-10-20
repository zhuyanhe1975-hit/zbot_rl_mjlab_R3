"""统一物理场景下的双足、蛇形与双向变形任务。"""
import math
from dataclasses import dataclass
from typing import Literal, cast

from mjlab.envs import mdp as common_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from ...robot import DEFAULT_JOINT_POS
from ..transformation.actions import JointTargetRateActionCfg
from ..transformation.env_cfg import TransformationEnvCfg, transformation_scene_cfg
from . import mdp
from .commands import UniversalCommandCfg

TASK_ID = "Mjlab-Zbot-6dof-Universal-Locomotion"


@dataclass
class UniversalEnvCfg(TransformationEnvCfg):
    omni_site_frame: bool = False
    omni_frame_backend: Literal['eager', 'cuda_graph'] = 'eager'
    motion_site: str = 'foot_0_sole'
    snake_teacher_enabled: bool = False
    straighten_before_stand: bool = False
    standing_joint_pos: tuple[float, ...] = DEFAULT_JOINT_POS
    standing_base_height: float = .264
    workspace_radius: float = 15.


def universal_env_cfg(play=False, *, cfg_cls: type[UniversalEnvCfg] = UniversalEnvCfg) -> UniversalEnvCfg:
    cfg = cast(UniversalEnvCfg, transformation_scene_cfg(play=play, cfg_cls=cfg_cls))
    cfg.scene.num_envs = 1 if play else 8192
    cfg.episode_length_s = 45.
    action = cast(JointTargetRateActionCfg, cfg.actions["joint_pos"])
    action.max_target_speed = 2 * math.pi
    action.target_limit = math.pi
    cfg.commands = {"transform": UniversalCommandCfg(resampling_time_range=(6., 12.), training=not play, mode="stand" if play else "source")}
    cfg.observations = {
        "actor": ObservationGroupCfg({"universal_state": ObservationTermCfg(func=mdp.actor_state)}, concatenate_terms=True, enable_corruption=False),
        "critic": ObservationGroupCfg({"universal_critic_state": ObservationTermCfg(func=mdp.critic_state)}, concatenate_terms=True, enable_corruption=False),
    }
    cfg.scene.sensors = (
        ContactSensorCfg(name="base_ground_contact", primary=ContactMatch(mode="body", pattern="base", entity="robot"), secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), reduce="netforce"),
        ContactSensorCfg(name="feet_ground_contact", primary=ContactMatch(mode="body", pattern=("foot_0", "foot_1"), entity="robot"), secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), reduce="netforce"),
        ContactSensorCfg(name="transform_ground_contact", primary=ContactMatch(mode="body", pattern=".*", entity="robot"), secondary=ContactMatch(mode="body", pattern="terrain"), fields=("force",), reduce="netforce", history_length=cfg.decimation),
    )
    cfg.events["reset_transition"] = EventTermCfg(func=mdp.reset_universal, mode="reset")
    cfg.terminations = {
        "unsafe": TerminationTermCfg(func=mdp.unsafe),
        "demonstration_time_out": TerminationTermCfg(func=mdp.demonstration_time_out, time_out=True),
        "time_out": TerminationTermCfg(func=mdp.play_time_out if play else common_mdp.time_out, time_out=True),
    }
    cfg.rewards = {
        "velocity_tracking": RewardTermCfg(func=mdp.velocity_tracking, weight=4.),
        "mode_posture": RewardTermCfg(func=mdp.posture_reward, weight=2.),
        "transition_progress": RewardTermCfg(func=mdp.transition_progress_reward, weight=2.),
        "standstill": RewardTermCfg(func=mdp.standstill_reward, weight=1.),
        "energy": RewardTermCfg(func=mdp.energy_cost, weight=-.002),
        "action_rate": RewardTermCfg(func=mdp.action_rate_cost, weight=-.01),
        "contact_impact": RewardTermCfg(func=mdp.impact_cost, weight=-.2),
        "failure": RewardTermCfg(func=mdp.failure_cost, weight=-5.),
    }
    cfg.metrics = {
        **{
            f"phase_{name}_fraction": MetricsTermCfg(func=mdp.phase_fraction, params={"phase": phase})
            for phase, name in enumerate(("biped", "descending", "snake", "ascending"))
        },
        "biped_motion_normalized_velocity_error": MetricsTermCfg(func=mdp.mode_motion_error, params={"phase": 0}),
        "snake_motion_normalized_velocity_error": MetricsTermCfg(func=mdp.mode_motion_error, params={"phase": 2}),
    }
    return cfg
