"""周期踏步、平地行走、速度与能效跟踪及全向行走配方。"""


import math
import os

from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.command_manager import CommandTermCfg
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

from ..robot import robot_cfg
from ..tasks.walking import mdp
from ..tasks.walking.env_cfg import (
    MAX_STEP_FREQUENCY,
    MIN_STEP_FREQUENCY,
    TASK_ID,
    WALKING_TASK_ID,
    ZbotEnvCfg,
)
from .base import RewardEdit, WalkingExperiment

VELOCITY_TRACKING_TASK_ID = "Mjlab-Zbot-6dof-Velocity-Frequency-Tracking-Walking"

VELOCITY_ENERGY_TRACKING_TASK_ID = "Mjlab-Zbot-6dof-Velocity-Energy-Tracking-Walking"

OMNI_VELOCITY_ENERGY_TASK_ID = "Mjlab-Zbot-6dof-Omni-Velocity-Energy-Walking"

# 指令范围：前向 m/s、左向 m/s、偏航角速度 rad/s（逆时针为正）。
LIN_VEL_X = (-0.3, 0.5)
LIN_VEL_Y = (-0.2, 0.2)
ANG_VEL_Z = (-0.5, 0.5)


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

    commands: dict[str, CommandTermCfg] = (
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


class PeriodicStepping(WalkingExperiment):
    task_id = TASK_ID
    experiment_name = "zbot_periodic_stepping"

    def reward_edit(self):
        return RewardEdit(
            remove=("forward_velocity",),
            terms={
                "planar_velocity": RewardTermCfg(
                    func=mdp.planar_velocity_penalty, weight=-5.0
                )
            },
        )


class FlatWalking(WalkingExperiment):
    task_id = WALKING_TASK_ID
    experiment_name = "zbot_walking"
    train_envs = 8192
    step_frequency = (0.25, 1.0)

    def reward_edit(self):
        return RewardEdit(
            weights={"frequency_tracking": 2.0},
            terms={
                "forward_velocity": RewardTermCfg(
                    func=mdp.forward_velocity_reward, weight=2.0
                )
            },
        )


class VelocityTracking(FlatWalking):
    task_id = VELOCITY_TRACKING_TASK_ID
    experiment_name = "zbot_velocity_tracking"
    step_frequency = (0.25, 1.0)

    def reward_edit(self):
        return RewardEdit(
            remove=("forward_velocity",),
            terms={
                "track_forward_speed": RewardTermCfg(
                    func=mdp.track_forward_speed, weight=2.0, params={"std": 0.1}
                )
            },
        )

    def edit(self, cfg, play):
        cfg.test_frequency = None
        cfg.frequency_sampling = "random"
        cfg.commands["target_speed"] = mdp.velocity_command_cfg(sequential=play)
        for group in cfg.observations.values():
            group.terms = dict(group.terms)
            group.terms["target_speed"] = ObservationTermCfg(func=mdp.target_speed)


class VelocityEnergyTracking(VelocityTracking):
    task_id = VELOCITY_ENERGY_TRACKING_TASK_ID
    experiment_name = "zbot_velocity_energy_tracking"

    def reward_edit(self):
        return RewardEdit(
            remove=("frequency_tracking",),
            weights={"joint_acceleration": -0.0001, "lateral_velocity": -0.5},
            terms={
                "joint_energy": RewardTermCfg(
                    func=mdp.joint_energy_penalty,
                    weight=-0.5,
                    params={"energy_scale": 10.0},
                )
            },
        )


class OmniVelocityEnergy(VelocityEnergyTracking):
    task_id = OMNI_VELOCITY_ENERGY_TASK_ID
    experiment_name = "zbot_omni_velocity_energy"

    def reward_edit(self):
        # 横向速度和偏航漂移原为惩罚，此处改为指令目标。
        return RewardEdit(
            remove=("lateral_velocity", "yaw_drift"),
            terms={
                "track_lateral_speed": RewardTermCfg(
                    func=mdp.track_lateral_speed, weight=1.0, params={"std": 0.1}
                ),
                "track_yaw_rate": RewardTermCfg(
                    func=mdp.track_yaw_rate, weight=1.0, params={"std": 0.25}
                ),
            },
        )

    def edit(self, cfg, play):
        cfg.commands["target_speed"] = mdp.velocity_command_cfg(
            lin_vel_x=LIN_VEL_X,
            lin_vel_y=LIN_VEL_Y,
            ang_vel_z=ANG_VEL_Z,
            cfg_cls=mdp.VelocityCommandVizCfg,
            debug_vis=play,
            full_scale=tuple(
                max(map(abs, r)) for r in (LIN_VEL_X, LIN_VEL_Y, ANG_VEL_Z)
            ),
        )
        # 三维指令箭头替代回放中仅表示前进速度的箭头。
        cfg.commands.pop("forward_velocity_viz", None)
        for group in cfg.observations.values():
            group.terms = dict(group.terms)
            # 转弯时，相对初始方向的航向漂移不再有意义。
            group.terms.pop("yaw_error")
            group.terms.pop("target_speed")
            group.terms["velocity_command"] = ObservationTermCfg(func=mdp.velocity_command)
