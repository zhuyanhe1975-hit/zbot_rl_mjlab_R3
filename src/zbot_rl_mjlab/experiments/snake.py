"""单独培养蛇形全向运动教师，不加载尚未学会蛇行的综合学生。"""


import math
from typing import Any, cast

import torch
from mjlab.envs import mdp as common_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.sensor import BuiltinSensorCfg, ContactMatch, ContactSensorCfg, ObjRef
from rsl_rl.modules import BetaDistribution

import zbot_rl_mjlab.tasks.snake.env_cfg as snake_task
from zbot_rl_mjlab.tasks.snake import mdp
from zbot_rl_mjlab.tasks.snake.commands import SnakeVelocityCommandCfg
from zbot_rl_mjlab.tasks.snake.env_cfg import TASK_ID, SnakeEnvCfg
from zbot_rl_mjlab.tasks.transformation.actions import JointTargetRateActionCfg
from zbot_rl_mjlab.tasks.transformation.env_cfg import transformation_scene_cfg

from ..tasks.snake.commands import SnakeVelocityCommand
from .base import Experiment


def snake_env_cfg(play=False) -> snake_task.SnakeEnvCfg:
    # 复用全身摩擦、平地、伺服和求解器，替换任务行为，不继承变形奖励。
    cfg = cast(snake_task.SnakeEnvCfg, transformation_scene_cfg(play=play, cfg_cls=snake_task.SnakeEnvCfg, action_cfg_cls=JointTargetRateActionCfg))
    cfg.scene.num_envs = 1 if play else 4096
    cfg.episode_length_s = 30.
    robot = cfg.scene.entities["robot"]
    robot.init_state.pos = (0., 0., .0555)
    # XML立柱姿态绕foot_0局部Y轴-90度：局部Z朝世界-X，局部X朝世界+Z。
    robot.init_state.rot = (math.sqrt(.5), 0., -math.sqrt(.5), 0.)
    robot.init_state.joint_pos = {".*": 0.}
    action = cast(JointTargetRateActionCfg, cfg.actions["joint_pos"])
    # 只限制本任务的控制目标，不改XML或运行时模型的机械关节范围。
    action.target_limit = snake_task.SNAKE_JOINT_LIMIT
    action.max_target_speed = snake_task.SNAKE_MAX_TARGET_SPEED
    cfg.commands = {"motion": SnakeVelocityCommandCfg(
        training=not play, resampling_time_range=(5., 9.), debug_vis=True,
        translation_probability=.5, turn_probability=.2, stop_probability=.1,
        sample_velocity_commands=False, translation_min_speed=0.,
    )}
    cfg.scene.sensors = (
        ContactSensorCfg(
            name="snake_ground_contact",
            primary=ContactMatch(mode="geom", pattern=".*_geom", entity="robot"),
            secondary=ContactMatch(mode="body", pattern="terrain"),
            fields=("found", "force"), reduce="maxforce", history_length=cfg.decimation,
        ),
        ContactSensorCfg(
            name="snake_self_contact",
            primary=ContactMatch(mode="subtree", pattern="foot_0", entity="robot"),
            secondary=ContactMatch(mode="subtree", pattern="foot_0", entity="robot"),
            fields=("found", "force", "dist"), reduce="maxforce", history_length=cfg.decimation,
        ),
        BuiltinSensorCfg(name="head_lin_acc", sensor_type="framelinacc", obj=ObjRef(type="xbody", name="foot_0", entity="robot")),
        BuiltinSensorCfg(name="head_ang_acc", sensor_type="frameangacc", obj=ObjRef(type="xbody", name="foot_0", entity="robot")),
    )
    cfg.events = {
        "reset_scene": EventTermCfg(func=common_mdp.reset_scene_to_default, mode="reset"),
        "reset_snake": EventTermCfg(func=mdp.reset_snake, mode="reset"),
    }
    cfg.observations = {
        "actor": ObservationGroupCfg({"snake": ObservationTermCfg(func=mdp.actor_state)}, concatenate_terms=True, enable_corruption=False),
        "critic": ObservationGroupCfg({"snake": ObservationTermCfg(func=mdp.critic_state)}, concatenate_terms=True, enable_corruption=False),
    }
    cfg.rewards = {
        "alive": RewardTermCfg(func=common_mdp.is_alive, weight=.02),
        "head_forward_speed": RewardTermCfg(func=mdp.head_forward_speed, weight=15.),
        "action_rate": RewardTermCfg(func=common_mdp.action_rate_l2, weight=-.0),
        "head_lateral_speed_cost": RewardTermCfg(func=mdp.head_lateral_speed_cost, weight=-5.0),
        "head_vertical_speed_cost": RewardTermCfg(func=mdp.head_vertical_speed_cost, weight=-0.2),
        "joint_pose_deviation": RewardTermCfg(func=mdp.joint_pose_deviation_cost, weight=-.02),
        "joint_fold_cost": RewardTermCfg(func=mdp.joint_fold_cost, weight=-1.,
                                        params={"soft_angle": 2 * math.pi / 3, "full_angle": math.pi}),
        "target_saturation_fraction": RewardTermCfg(func=mdp.target_saturation_fraction, weight=-.1),
        "body_yaw_integral_cost": RewardTermCfg(func=mdp.body_yaw_integral_cost, weight=-1.0),
        "body_yaw_cost": RewardTermCfg(func=mdp.body_yaw_cost, weight=-.1),
        "body_yaw_reset_cost": RewardTermCfg(func=mdp.body_yaw_reset_cost, weight=-5.),
        "head_yaw_integral_cost": RewardTermCfg(func=mdp.head_yaw_integral_cost, weight=-.5),
        "head_yaw_cost": RewardTermCfg(func=mdp.head_yaw_cost, weight=-.5),
        "head_yaw_reset_cost": RewardTermCfg(func=mdp.head_yaw_reset_cost, weight=-5.),
        "head_tilt_reset_cost": RewardTermCfg(func=mdp.head_tilt_reset_cost, weight=-5.0),
        "head_angular_velocity_cost": RewardTermCfg(func=mdp.head_angular_velocity_cost, weight=-.2),
        "self_collision_reset_cost": RewardTermCfg(func=mdp.self_collision_reset_cost, weight=-5.0),
    }
    cfg.terminations = {
        "unsafe": TerminationTermCfg(func=mdp.unsafe),
        "head_tilt": TerminationTermCfg(func=mdp.head_tilt_limit_reached),
        "head_yaw": TerminationTermCfg(func=mdp.head_yaw_limit_reached),
        "body_yaw": TerminationTermCfg(func=mdp.body_yaw_limit_reached),
        "self_collision": TerminationTermCfg(func=mdp.self_collision),
        "time_out": TerminationTermCfg(func=mdp.play_time_out if play else common_mdp.time_out, time_out=True),
    }
    cfg.metrics = {
        **{f"world_velocity_{name}": MetricsTermCfg(
            func=mdp.world_velocity_component, params={"axis": axis})
           for axis, name in enumerate(("x", "y", "z"))},
        "target_saturation_fraction": MetricsTermCfg(func=mdp.target_saturation_fraction),
        "action_clip_fraction": MetricsTermCfg(func=mdp.action_clip_fraction),
        "joint_pose_rms_rad": MetricsTermCfg(func=mdp.joint_pose_rms),
        "joint_velocity_rms_rad_s": MetricsTermCfg(func=mdp.joint_velocity_rms),
        "motion_penalty_scale": MetricsTermCfg(func=mdp.motion_penalty_scale),
        "head_forward_velocity_m_s": MetricsTermCfg(func=mdp.head_forward_velocity),
        "head_angular_speed_rad_s": MetricsTermCfg(func=mdp.head_angular_speed),
        "body_yaw_integral_abs_rad_s": MetricsTermCfg(func=mdp.body_yaw_integral_abs),
        "body_yaw_deviation_rad": MetricsTermCfg(func=mdp.body_yaw_abs_deviation),
        "body_yaw_peak_rad": MetricsTermCfg(func=mdp.body_yaw_abs_deviation, reduce="max"),
        "head_yaw_integral_abs_rad_s": MetricsTermCfg(func=mdp.head_yaw_integral_abs),
        "head_yaw_deviation_rad": MetricsTermCfg(func=mdp.head_yaw_abs_deviation),
        "head_yaw_peak_rad": MetricsTermCfg(func=mdp.head_yaw_abs_deviation, reduce="max"),
        "head_tilt_rad": MetricsTermCfg(func=mdp.head_tilt),
        "head_tilt_peak_rad": MetricsTermCfg(func=mdp.head_tilt, reduce="max"),
        "ground_force_peak_bodyweights": MetricsTermCfg(func=mdp.normalized_ground_peak, reduce="max"),
    }
    cfg.sim.contact_sensor_maxmatch = 128
    cfg.viewer.lookat = (0., 0., .05)
    return cfg


class CompactSnakeRunner(MjlabOnPolicyRunner):
    """原生训练概览与全部奖励，详细诊断仅写入日志。"""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        full_log = self.logger.log
        env_cfg = getattr(getattr(env, "unwrapped", env), "cfg", None)

        def log_with_group_means(*args, **kwargs):
            console_lines = []
            if self.logger.writer is not None:
                conditional = {}
                for group in ("translation", "yaw", "stop", "mixed"):
                    prefix = f"Episode_Metrics/group_{group}_"
                    for metric in ("linear_reward", "yaw_reward", "target_speed",
                                   "actual_speed", "linear_error", "yaw_error"):
                        # 匹配同一批回合的掩码分子与曝光分母；不平均各批条件均值。
                        pairs = [(torch.as_tensor(info[prefix + metric], device=self.logger.device).reshape(-1),
                                  torch.as_tensor(info[prefix + "fraction"], device=self.logger.device).reshape(-1))
                                 for info in self.logger.ep_extras
                                 if prefix + metric in info and prefix + "fraction" in info]
                        if not pairs:
                            continue
                        fraction = torch.cat([denominator for _, denominator in pairs]).float().mean()
                        if fraction.item() <= 0:
                            continue  # 无曝光组保留fraction=0，不伪造条件奖励或误差。
                        numerator = torch.cat([sample for sample, _ in pairs]).float().mean()
                        conditional[prefix + "conditional_" + metric] = numerator / fraction
                if conditional:
                    self.logger.ep_extras.append(conditional)

                def mean_value(key):
                    values = [torch.as_tensor(info[key], device=self.logger.device).reshape(-1)
                              for info in self.logger.ep_extras if key in info]
                    return torch.cat(values).float().mean().item() if values else None

                def display(key, scale=1., digits=4):
                    value = mean_value(key)
                    return f"{value * scale:.{digits}f}" if value is not None else "--"

                reward_keys = dict.fromkeys(key for info in self.logger.ep_extras
                                            for key in info if key.startswith("Episode_Reward/"))
                pad = kwargs.get("pad", 40)
                console_lines.extend(f"{key + ':':>{pad}} {display(key)}" for key in reward_keys)
                if self.logger.ep_extras:
                    if getattr(env_cfg, "observation_reference", "command") == "world_x":
                        console_lines.append(
                            f"世界速度 m/s：X {display('Episode_Metrics/world_velocity_x')} "
                            f"Y {display('Episode_Metrics/world_velocity_y')} "
                            f"Z {display('Episode_Metrics/world_velocity_z')}"
                        )
                    else:
                        velocities = " ".join(
                            f"{label}{display(f'Episode_Metrics/{name}_velocity')}"
                            for name, label in (("forward", "前"), ("backward", "后"),
                                                ("left", "左"), ("right", "右"),
                                                ("turn_left", "左转"), ("turn_right", "右转"))
                        )
                        prefix = "Episode_Metrics/group_translation_conditional_"
                        console_lines.extend((
                            f"速度：{velocities} （平移m/s，转向rad/s）",
                            (f"纯平移 m/s：目标{display(prefix + 'target_speed')} "
                             f"实际{display(prefix + 'actual_speed')} 误差{display(prefix + 'linear_error')} "
                             f"课程上限{display('Episode_Metrics/translation_curriculum_cap')}"),
                        ))
                    console_lines.extend((
                        (f"诊断：头倾{display('Episode_Metrics/head_tilt_rad', 180 / math.pi, 1)}° "
                        f"航向偏差{display('Episode_Metrics/head_yaw_deviation_rad', 180 / math.pi, 1)}° "
                        f"体轴偏差{display('Episode_Metrics/body_yaw_deviation_rad', 180 / math.pi, 1)}° "
                        f"头部积分{display('Episode_Metrics/head_yaw_integral_abs_rad_s', 180 / math.pi, 1)}°·s "
                        f"体轴积分{display('Episode_Metrics/body_yaw_integral_abs_rad_s', 180 / math.pi, 1)}°·s "
                        f"头角速度{display('Episode_Metrics/head_angular_speed_rad_s')}rad/s "
                        f"运动惩罚倍率{display('Episode_Metrics/motion_penalty_scale')} "
                        f"关节速度{display('Episode_Metrics/joint_velocity_rms_rad_s')}rad/s "
                        f"目标顶限{display('Episode_Metrics/target_saturation_fraction', 100, 1)}%"),
                        (f"复位计数：头倾{display('Episode_Termination/head_tilt', digits=2)} "
                        f"航向{display('Episode_Termination/head_yaw', digits=2)} "
                        f"体轴{display('Episode_Termination/body_yaw', digits=2)} "
                        f"自碰{display('Episode_Termination/self_collision', digits=2)} "
                        f"安全{display('Episode_Termination/unsafe', digits=2)}"),
                    ))
            # 所有标量照常写入TensorBoard；仅关闭原生终端的完整extras清单。
            kwargs["print_minimal"] = True
            result = full_log(*args, **kwargs)
            if console_lines:
                print("\n".join(console_lines), flush=True)
            return result

        cast(Any, self.logger).log = log_with_group_means


class SnakeTeacherRunner(CompactSnakeRunner):
    """原生Beta教师训练，仅加载包含头部速度观测的新版检查点。"""

    @staticmethod
    def _describe_checkpoint(checkpoint):
        recipe = (checkpoint.get("infos") or {}).get("snake_recipe")
        if not isinstance(recipe, dict) or recipe.get("version") != 22:
            raise ValueError("旧蛇形检查点动作或观测不兼容，新蛇形教师任务请从零训练（需要第22版检查点）。")
        if (recipe.get("motion_frame") not in ("body_gravity", "legacy_head")
                or recipe.get("head_gyro_frame") not in ("head", "world")
                or not isinstance(recipe.get("motion_body"), str) or not recipe["motion_body"]
                or not isinstance(recipe.get("motion_site"), str) or not recipe["motion_site"]
                or recipe.get("objective") != "snake_world_x_capped_speed"
                or recipe.get("observation_reference") != "world_x"
                or type(recipe.get("forward_speed_cap")) not in (float, int)
                or not math.isfinite(recipe["forward_speed_cap"]) or recipe["forward_speed_cap"] <= 0):
            raise ValueError("蛇形检查点任务或观测坐标元数据无效。")
        speed_gain = recipe.get("motion_penalty_speed_gain")
        min_scale = recipe.get("motion_penalty_min_scale")
        if (recipe.get("motion_penalty_scaling") != "clipped_positive_world_x_ramp"
                or type(speed_gain) not in (float, int)
                or not math.isfinite(speed_gain) or speed_gain < 0
                or type(min_scale) not in (float, int)
                or not math.isfinite(min_scale) or not 0 <= min_scale <= 1):
            raise ValueError("蛇形检查点运动惩罚速度缩放元数据无效。")
        yaw_limit = recipe.get("head_yaw_limit")
        if (recipe.get("yaw_reference") != "episode_initial_heading"
                or recipe.get("head_orientation_observation") != "quaternion_wxyz_and_relative_yaw_sincos"
                or type(yaw_limit) not in (float, int)
                or not math.isfinite(yaw_limit) or not 0 < yaw_limit <= math.pi
                or recipe.get("actor_observation_dim") != 51
                or recipe.get("critic_observation_dim") != 66):
            raise ValueError("蛇形检查点头部姿态或初始航向元数据无效。")
        body_yaw_limit = recipe.get("body_yaw_limit")
        if (recipe.get("body_yaw_reference") != "episode_initial_heading"
                or not isinstance(recipe.get("tail_site"), str) or not recipe["tail_site"]
                or type(body_yaw_limit) not in (float, int)
                or not math.isfinite(body_yaw_limit) or not 0 < body_yaw_limit <= math.pi):
            raise ValueError("蛇形检查点体轴初始航向元数据无效。")
        integral_time_scale = recipe.get("body_yaw_integral_time_scale")
        if (recipe.get("body_yaw_integral") != "signed_episode_dt_tanh_squared"
                or type(integral_time_scale) not in (float, int)
                or not math.isfinite(integral_time_scale) or integral_time_scale <= 0):
            raise ValueError("蛇形检查点体轴航向积分元数据无效。")
        head_integral_time_scale = recipe.get("head_yaw_integral_time_scale")
        if (recipe.get("head_yaw_integral") != "signed_episode_dt_tanh_squared"
                or type(head_integral_time_scale) not in (float, int)
                or not math.isfinite(head_integral_time_scale) or head_integral_time_scale <= 0):
            raise ValueError("蛇形检查点头部航向积分元数据无效。")
        for network, expected in (("actor", 51), ("critic", 66)):
            inputs = [(int(key.split('.')[1]), value)
                      for key, value in checkpoint.get(f"{network}_state_dict", {}).items()
                      if key.startswith("mlp.") and key.endswith(".weight")
                      and key.split('.')[1].isdigit()]
            first_weight = min(inputs, key=lambda item: item[0])[1] if inputs else None
            if first_weight is None or first_weight.ndim != 2 or first_weight.shape[1] != expected:
                raise ValueError("蛇形检查点头部姿态观测维度不兼容。")
        target_limit = recipe.get("joint_target_limit")
        max_target_speed = recipe.get("max_target_speed")
        if (recipe.get("action_control") != "integrated_joint_velocity"
                or type(max_target_speed) not in (float, int)
                or not math.isfinite(max_target_speed) or max_target_speed <= 0
                or type(target_limit) not in (float, int)
                or not math.isfinite(target_limit) or target_limit <= 0):
            raise ValueError("蛇形检查点关节速度积分动作元数据无效。")
        action_range = recipe.get("action_range")
        actor = checkpoint.get("actor_state_dict", {})
        if (recipe.get("distribution") != "BetaDistribution"
                or not isinstance(action_range, (tuple, list))
                or tuple(action_range) != (-1., 1.)
                or "distribution.std_param" in actor or "distribution.log_std_param" in actor):
            raise ValueError("Beta检查点动作分布或值域不匹配。")
        # 直接检查输出头，避免先加载部分权重后才遇到形状错误。
        heads = [(int(key.split('.')[1]), value) for key, value in actor.items()
                 if key.startswith('mlp.') and key.endswith('.weight')
                 and key.split('.')[1].isdigit()]
        if not heads:
            raise ValueError("检查点缺少动作输出头。")
        head_index, head_weight = max(heads, key=lambda item: item[0])
        head_bias = actor.get(f'mlp.{head_index}.bias')
        if (head_weight.ndim != 2 or head_weight.shape[0] != 12
                or head_bias is None or head_bias.shape != (12,)):
            raise ValueError("检查点动作输出头与声明的分布不匹配。")
        return recipe

    @staticmethod
    def prepare_playback_cfg(train_cfg, path):
        """构造推理网络前验证新版教师检查点。"""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        SnakeTeacherRunner._describe_checkpoint(checkpoint)
        train_cfg["actor"]["distribution_cfg"] = {
            "class_name": "BetaDistribution", "action_range": (-1., 1.),
        }

    def _recipe(self):
        distribution = self.alg.get_policy().distribution
        if not isinstance(distribution, BetaDistribution):
            raise TypeError("蛇形教师检查点必须使用Beta策略。")
        if tuple(distribution.action_range) != (-1., 1.):
            raise ValueError("蛇形Beta动作值域必须为[-1, 1]。")
        cfg = cast(SnakeEnvCfg, self.env.unwrapped.cfg)
        return {"version": 22, "motion_frame": cfg.motion_frame, "motion_body": cfg.head_body,
                "motion_site": cfg.motion_site, "tail_site": cfg.tail_site,
                "body_yaw_limit": cfg.body_yaw_limit,
                "body_yaw_reference": "episode_initial_heading",
                "body_yaw_integral": "signed_episode_dt_tanh_squared",
                "body_yaw_integral_time_scale": cfg.body_yaw_integral_time_scale,
                "objective": "snake_world_x_capped_speed",
                "forward_speed_cap": cfg.forward_speed_cap,
                "motion_penalty_speed_gain": cfg.motion_penalty_speed_gain,
                "motion_penalty_min_scale": cfg.motion_penalty_min_scale,
                "motion_penalty_scaling": "clipped_positive_world_x_ramp",
                "yaw_reference": "episode_initial_heading",
                "head_yaw_limit": cfg.head_yaw_limit,
                "head_yaw_integral": "signed_episode_dt_tanh_squared",
                "head_yaw_integral_time_scale": cfg.head_yaw_integral_time_scale,
                "head_orientation_observation": "quaternion_wxyz_and_relative_yaw_sincos",
                "actor_observation_dim": 51, "critic_observation_dim": 66,
                "observation_reference": cfg.observation_reference,
                "head_gyro_frame": cfg.head_gyro_frame,
                "action_control": "integrated_joint_velocity",
                "max_target_speed": cast(Any, cfg.actions["joint_pos"]).max_target_speed,
                "joint_target_limit": cast(Any, cfg.actions["joint_pos"]).target_limit,
                "distribution": "BetaDistribution", "action_range": tuple(distribution.action_range)}

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        distribution = self.alg.get_policy().distribution
        if not isinstance(distribution, BetaDistribution) or tuple(distribution.action_range) != (-1., 1.):
            raise ValueError("新蛇形训练只使用原生Beta策略，请从零训练。")
        if cast(SnakeEnvCfg, self.env.unwrapped.cfg).observation_reference != "world_x":
            raise ValueError("蛇形教师训练需要world_x观测参考，请从零训练。")
        return super().learn(num_learning_iterations, init_at_random_ep_len)

    def save(self, path: str, infos=None) -> None:
        saved_infos = {**(infos or {}), "snake_recipe": self._recipe()}
        if hasattr(self, "_resume_joint_limit_change"):
            saved_infos["snake_resume_joint_limit_change"] = self._resume_joint_limit_change
        manager = getattr(self.env.unwrapped, "command_manager", None)
        term = manager.get_term("motion") if manager is not None else None
        if isinstance(term, SnakeVelocityCommand) and term.cfg.translation_curriculum:
            saved_infos["snake_curriculum"] = term.curriculum_state()
        super().save(path, saved_infos)

    def load(self, path: str, load_cfg=None, strict=True, map_location=None):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        recipe = self._describe_checkpoint(checkpoint)
        actor_only = load_cfg is not None and load_cfg.get("actor", False) and not any(
            load_cfg.get(key, False) for key in ("critic", "optimizer", "iteration", "rnd")
        )
        distribution = self.alg.get_policy().distribution
        if not isinstance(distribution, BetaDistribution):
            raise TypeError("蛇形教师检查点仅支持原生Beta分布。")
        if not actor_only:
            current_recipe = self._recipe()
            differences = sorted(key for key in recipe.keys() | current_recipe.keys()
                                 if recipe.get(key) != current_recipe.get(key))
            allowed_limit_change = (differences == ["joint_target_limit"]
                                    and getattr(self.env.unwrapped.cfg, "allow_resume_joint_limit_change", False))
            if differences and not allowed_limit_change:
                details = "; ".join(f"{key}: 检查点={recipe.get(key)!r}, 当前={current_recipe.get(key)!r}"
                                    for key in differences)
                hint = "请使用训练时的配置续训；若有意改变任务或观测语义，请从零训练。"
                if differences == ["joint_target_limit"]:
                    hint = ("保持原关节目标范围续训可添加 "
                            f"--env.actions.joint-pos.target-limit {recipe['joint_target_limit']}；"
                            "有意改变范围继续训练可添加 --env.allow-resume-joint-limit-change True。")
                raise ValueError(f"检查点任务或观测语义不兼容：{details}。{hint}")
        cfg = cast(SnakeEnvCfg, self.env.unwrapped.cfg)
        if actor_only and recipe["motion_body"] != cfg.head_body:
            raise ValueError("检查点运动坐标参考模块与回放配置不一致。")
        if actor_only and recipe["motion_site"] != cfg.motion_site:
            raise ValueError("检查点运动坐标参考site与回放配置不一致。")
        if actor_only and recipe["head_yaw_limit"] != cfg.head_yaw_limit:
            raise ValueError("检查点初始航向复位阈值与回放配置不一致。")
        if actor_only and (recipe["tail_site"] != cfg.tail_site
                           or recipe["body_yaw_limit"] != cfg.body_yaw_limit):
            raise ValueError("检查点体轴参考site或初始航向复位阈值与回放配置不一致。")
        if actor_only and recipe["body_yaw_integral_time_scale"] != cfg.body_yaw_integral_time_scale:
            raise ValueError("检查点体轴航向积分时间尺度与回放配置不一致。")
        if actor_only and recipe["head_yaw_integral_time_scale"] != cfg.head_yaw_integral_time_scale:
            raise ValueError("检查点头部航向积分时间尺度与回放配置不一致。")
        if actor_only and (recipe["motion_penalty_speed_gain"] != cfg.motion_penalty_speed_gain
                           or recipe["motion_penalty_min_scale"] != cfg.motion_penalty_min_scale):
            raise ValueError("检查点运动惩罚速度缩放与回放配置不一致。")
        curriculum = (checkpoint.get("infos") or {}).get("snake_curriculum")
        manager = getattr(self.env.unwrapped, "command_manager", None)
        term = manager.get_term("motion") if manager is not None else None
        if not actor_only and curriculum is not None:
            if not isinstance(term, SnakeVelocityCommand):
                raise ValueError("当前环境缺少蛇形课程指令项")
            term.validate_curriculum_state(curriculum)
        previous_counter = self.env.unwrapped.common_step_counter
        infos = super().load(path, load_cfg, strict, map_location)
        if not actor_only and allowed_limit_change:
            self._resume_joint_limit_change = {
                "source_checkpoint": str(path),
                "source_iteration": checkpoint.get("iter"),
                "old_limit": recipe["joint_target_limit"],
                "new_limit": current_recipe["joint_target_limit"],
            }
            print(f"蛇形迁移续训：关节目标范围 ±{math.degrees(recipe['joint_target_limit']):g}° → "
                  f"±{math.degrees(current_recipe['joint_target_limit']):g}°；恢复权重、优化器及迭代计数。")
        if not actor_only and curriculum is not None and isinstance(term, SnakeVelocityCommand):
            term.load_curriculum_state(curriculum)
        if actor_only:
            env = self.env.unwrapped
            cfg = cast(SnakeEnvCfg, env.cfg)
            env.common_step_counter = previous_counter
            cfg.motion_frame = "legacy_head" if recipe["motion_frame"] == "legacy_head" else "body_gravity"
            cfg.head_gyro_frame = "world" if recipe["head_gyro_frame"] == "world" else "head"
            cfg.observation_reference = "world_x" if recipe["observation_reference"] == "world_x" else "command"
            env.observation_manager.compute(update_history=True)
        return infos


class SnakeOmniTeacher(Experiment):
    task_id = TASK_ID
    experiment_name = "zbot_snake_omni_teacher"
    train_envs = 4096
    max_iterations = 3000
    runner_cls = SnakeTeacherRunner

    def create_env(self, play=False):
        return snake_env_cfg(play=play)

    def _build_ppo(self):
        cfg = super()._build_ppo()
        cfg.actor.distribution_cfg = {
            "class_name": "BetaDistribution", "action_range": (-1., 1.),
        }
        cfg.algorithm.entropy_coef = .002
        return cfg
