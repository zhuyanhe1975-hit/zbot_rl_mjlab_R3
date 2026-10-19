"""周期踏步、平地行走、速度与能效跟踪及全向行走配方。"""

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from ..tasks.walking import mdp
from ..tasks.walking.env_cfg import TASK_ID, WALKING_TASK_ID
from .base import RewardEdit, WalkingExperiment

VELOCITY_TRACKING_TASK_ID = "Mjlab-Zbot-6dof-Velocity-Tracking-Walking"

VELOCITY_ENERGY_TRACKING_TASK_ID = "Mjlab-Zbot-6dof-Velocity-Energy-Tracking-Walking"

OMNI_VELOCITY_ENERGY_TASK_ID = "Mjlab-Zbot-6dof-Omni-Velocity-Energy-Walking"

# 指令范围：前向 m/s、左向 m/s、偏航角速度 rad/s（逆时针为正）。
LIN_VEL_X = (-0.3, 0.5)
LIN_VEL_Y = (-0.2, 0.2)
ANG_VEL_Z = (-0.5, 0.5)


class PeriodicStepping(WalkingExperiment):
    task_id = TASK_ID
    experiment_name = "zbot_periodic_stepping"


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
