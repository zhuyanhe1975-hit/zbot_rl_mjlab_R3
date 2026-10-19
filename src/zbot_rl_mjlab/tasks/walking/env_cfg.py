"""周期踏步与行走任务共用的基础环境。"""

import math
import os
from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from ...robot import robot_cfg
from . import mdp

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


def env_cfg(play=False):
    fixed = os.environ.get("ZBOT_TEST_FREQUENCY") if play else None
    fixed = float(fixed) if fixed is not None else None
    if fixed is not None and (not math.isfinite(fixed) or fixed <= 0):
        raise ValueError("Test frequency must be finite and positive")
    feet = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="body", pattern=("foot_0", "foot_1"), entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        track_air_time=True,
        history_length=4,
    )
    height = TerrainHeightSensorCfg(
        name="foot_height_scan",
        # 射线从足底位点发出，而非足部刚体：foot_1 是旧版的 "b6" 刚体，
        # 其原点在足底上方约 54 mm；此前会导致离地高度读数偏移，
        # 并让射线碰到邻近连杆。
        frame=(
            ObjRef(type="site", name="foot_0_sole", entity="robot"),
            ObjRef(type="site", name="foot_1_sole", entity="robot"),
        ),
        ray_alignment="yaw",
        pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=6),
        max_distance=1.0,
        exclude_parent_body=True,
        include_geom_groups=(0,),
        debug_vis=True,
    )
    actor = {
        "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel),
        "base_ang_vel": ObservationTermCfg(func=mdp.base_ang_vel),
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
        "yaw_error": ObservationTermCfg(func=mdp.yaw_error_observation),
        "joint_pos": ObservationTermCfg(func=envs_mdp.joint_pos_rel),
        "joint_vel": ObservationTermCfg(func=envs_mdp.joint_vel_rel),
        "actions": ObservationTermCfg(func=envs_mdp.last_action),
        "phase": ObservationTermCfg(func=mdp.phase_observation),
        "frequency": ObservationTermCfg(func=mdp.frequency_observation),
    }
    rewards = {
        "frequency_tracking": RewardTermCfg(
            func=mdp.alternating_foot_phase,
            weight=3.0,
            params={"sensor_name": feet.name},
        ),
        "swing_clearance": RewardTermCfg(
            func=mdp.swing_clearance,
            weight=0.5,
            params={"sensor_name": feet.name, "height_sensor_name": height.name},
        ),
        "forward_velocity": RewardTermCfg(
            func=mdp.forward_velocity_reward, weight=5.0, params={"scale": 0.1}
        ),
        "lateral_velocity": RewardTermCfg(
            func=mdp.lateral_velocity_penalty, weight=-0.2
        ),
        "upright": RewardTermCfg(func=mdp.upright, weight=0.2),
        "yaw_drift": RewardTermCfg(func=mdp.yaw_drift_penalty, weight=-0.2),
        "support_foot_slip": RewardTermCfg(
            func=mdp.support_foot_slip,
            weight=-5.0,
            params={
                "sensor_name": feet.name,
                "site_names": ("foot_0_sole", "foot_1_sole"),
            },
        ),
        "soft_landing": RewardTermCfg(
            func=mdp.soft_landing, weight=-2e-2, params={"sensor_name": feet.name}
        ),
        "foot_nonparallel_ground": RewardTermCfg(
            func=mdp.foot_nonparallel_ground_penalty,
            weight=-5.0,
            params={"sensor_name": feet.name, "std": 0.1},
        ),
        "action_rate": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
        "joint_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-0.1),
    }
    rewards["joint_acceleration"] = RewardTermCfg(func=mdp.joint_acc_l2, weight=-0.0005)

    class VizCfg(UniformVelocityCommandCfg):
        def build(self, env):
            return mdp.ForwardVelocityVisualizer(self, env)

    commands = (
        {
            "forward_velocity_viz": VizCfg(
                entity_name="robot",
                resampling_time_range=(1.0, 1.0),
                ranges=UniformVelocityCommandCfg.Ranges(
                    lin_vel_x=(0.0, 0.0),
                    lin_vel_y=(0.0, 0.0),
                    ang_vel_z=(0.0, 0.0),
                    heading=None,
                ),
                debug_vis=True,
            )
        }
        if play
        else {}
    )
    return ZbotEnvCfg(
        seed=42,
        test_frequency=fixed,
        step_frequency_min=MIN_STEP_FREQUENCY,
        step_frequency_max=MAX_STEP_FREQUENCY,
        frequency_sampling="sequential" if play else "random",
        decimation=4,
        episode_length_s=20.0,
        scene=SceneCfg(
            num_envs=1 if play else 4096,
            env_spacing=4.0,
            terrain=TerrainEntityCfg(terrain_type="plane"),
            entities={"robot": robot_cfg()},
            sensors=(feet, height),
        ),
        observations={
            "actor": ObservationGroupCfg(
                actor, concatenate_terms=True, enable_corruption=not play
            ),
            "critic": ObservationGroupCfg(
                actor, concatenate_terms=True, enable_corruption=False
            ),
        },
        actions={
            "joint_pos": JointPositionActionCfg(
                entity_name="robot",
                actuator_names=(".*",),
                scale=0.25,
                use_default_offset=True,
            )
        },
        commands=commands,
        rewards=rewards,
        terminations={
            "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
            "fallen": TerminationTermCfg(func=mdp.fallen),
        },
        events={
            "reset_scene_to_default": EventTermCfg(
                func=envs_mdp.reset_scene_to_default, mode="reset"
            )
        },
        sim=SimulationCfg(
            mujoco=MujocoCfg(
                timestep=0.005,
                iterations=10,
                ls_iterations=20,
                ccd_iterations=50,
                gravity=(0, 0, -9.81),
            ),
            njmax=300,
            nconmax=None,
            contact_sensor_maxmatch=64,
        ),
        viewer=ViewerConfig(
            entity_name="robot",
            body_name="base",
            distance=1.2,
            elevation=-20.0,
            azimuth=135.0,
            width=640,
            height=480,
            origin_type=ViewerConfig.OriginType.WORLD,
            lookat=(0.0, 0.0, 0.2),
        ),
    )

