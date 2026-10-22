"""行走实验共用的配置。"""


from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.sensor import BuiltinSensorCfg, ObjRef

from ..tasks.common import add_base_disturbance  # noqa: F401  # 共用扰动配置入口
from ..tasks.walking import mdp


def ppo_cfg():
    actor = RslRlModelCfg(
        hidden_dims=(256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    )
    critic = RslRlModelCfg(
        hidden_dims=(256, 128), activation="elu", obs_normalization=True
    )
    return RslRlOnPolicyRunnerCfg(
        actor=actor,
        critic=critic,
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="zbot_periodic_stepping",
        num_steps_per_env=24,
        max_iterations=1000,
        save_interval=50,
        logger="tensorboard",
    )


def add_walking_regularizers(cfg):
    """添加共用的摆动足、接触、能耗及加速度项。"""
    foot_accelerometers = tuple(
        BuiltinSensorCfg(
            name=f"foot_{index}_lin_acc",
            sensor_type="framelinacc",
            obj=ObjRef(type="body", name=f"foot_{index}", entity="robot"),
        )
        for index in range(2)
    )
    cfg.scene.sensors = (*cfg.scene.sensors, *foot_accelerometers)
    cfg.rewards.update(
        {
            "swing_foot_acceleration": RewardTermCfg(
                func=mdp.swing_foot_acceleration_penalty,
                weight=-0.1,
                params={
                    "acceleration_sensors": tuple(
                        sensor.prefixed_name for sensor in foot_accelerometers
                    ),
                    "acceleration_scale": 50.0,
                },
            ),
            "swing_foot_excess_height": RewardTermCfg(
                func=mdp.swing_foot_excess_height_penalty,
                weight=-20.0,
                params={"max_height": 0.05, "height_scale": 0.05},
            ),
            "joint_energy": RewardTermCfg(
                func=mdp.joint_energy_penalty,
                weight=-0.2,
                params={"energy_scale": 10.0},
            ),
            "feet_self_collision": RewardTermCfg(
                func=mdp.foot_self_collision_penalty,
                weight=-0.5,
                params={"sensor_name": "feet_self_contact", "scale": 10.0},
            ),
            "foot_nonparallel_ground": RewardTermCfg(
                func=mdp.foot_nonparallel_ground_penalty,
                weight=-10.0,
                params={"sensor_name": "feet_ground_contact", "std": 0.05},
            ),
            "joint_acceleration": RewardTermCfg(
                func=mdp.joint_acceleration_penalty, weight=-0.001
            ),
        }
    )
    return cfg


def add_domain_randomization(
    cfg,
    friction_range=(0.3, 1.2),
    encoder_bias_range=(-0.015, 0.015),
    base_com_ranges=None,
    geom_names=("a1_geom", "b6_geom"),
):
    """启动时随机化足部摩擦、编码器零偏与基座质心。"""
    if base_com_ranges is None:
        base_com_ranges = {0: (-0.025, 0.025), 1: (-0.025, 0.025), 2: (-0.03, 0.03)}
    events = {
            "randomize_foot_friction": EventTermCfg(
                mode="startup",
                func=dr.geom_friction,
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", geom_names=geom_names
                    ),
                    "operation": "abs",
                    "ranges": friction_range,
                    "shared_random": True,
                },
            ),
            "randomize_encoder_bias": EventTermCfg(
                mode="startup",
                func=dr.encoder_bias,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "bias_range": encoder_bias_range,
                },
            ),
            "randomize_base_com": EventTermCfg(
                mode="startup",
                func=dr.body_com_offset,
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
                    "operation": "add",
                    "ranges": base_com_ranges,
                },
            ),
    }
    if encoder_bias_range is None:
        del events["randomize_encoder_bias"]
    cfg.events.update(events)
    return cfg
