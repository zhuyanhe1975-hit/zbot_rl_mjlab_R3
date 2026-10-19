"""单独培养蛇形全向运动教师，不加载尚未学会蛇行的综合学生。"""

import math
from typing import cast

import torch
from mjlab.rl import MjlabOnPolicyRunner
from rsl_rl.modules import BetaDistribution, GaussianDistribution

from ..tasks.snake.commands import SnakeVelocityCommand
from ..tasks.snake.env_cfg import TASK_ID, SnakeEnvCfg, snake_env_cfg
from .base import Experiment


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
                        f"关节速度{display('Episode_Metrics/joint_velocity_rms_rad_s')}rad/s "
                        f"目标顶限{display('Episode_Metrics/target_saturation_fraction', 100, 1)}%"),
                        (f"复位计数：头倾{display('Episode_Termination/head_tilt', digits=2)} "
                        f"自碰{display('Episode_Termination/self_collision', digits=2)} "
                        f"安全{display('Episode_Termination/unsafe', digits=2)}"),
                    ))
            # 所有标量照常写入TensorBoard；仅关闭原生终端的完整extras清单。
            kwargs["print_minimal"] = True
            result = full_log(*args, **kwargs)
            if console_lines:
                print("\n".join(console_lines), flush=True)
            return result

        self.logger.log = log_with_group_means


class SnakeTeacherRunner(CompactSnakeRunner):
    """原生Beta训练；检查点区分动作分布和观测语义，不迁移高斯输出头。"""

    @staticmethod
    def _describe_checkpoint(checkpoint):
        recipe = (checkpoint.get("infos") or {}).get("snake_recipe")
        actor = checkpoint.get("actor_state_dict", {})
        scalar_std = "distribution.std_param" in actor
        log_std = "distribution.log_std_param" in actor
        gaussian = scalar_std or log_std
        if scalar_std and log_std:
            raise ValueError("检查点同时包含两种高斯标准差，无法识别动作分布。")
        frames = {"motion_frame": "legacy_head", "head_gyro_frame": "world"}
        if recipe is None:
            if not gaussian:
                raise ValueError("未标记动作分布的旧检查点不能作为Beta模型加载。")
            version = 0
        else:
            if not isinstance(recipe, dict):
                raise ValueError("蛇形检查点元数据无效。")
            version = recipe.get("version")
            if (recipe.get("motion_frame") not in ("body_gravity", "legacy_head")
                    or recipe.get("head_gyro_frame") not in ("head", "world")):
                raise ValueError("蛇形检查点观测坐标元数据无效。")
            frames = {key: recipe[key] for key in frames}
            if version == 2:
                if not gaussian:
                    raise ValueError("第2版检查点应为高斯策略，动作分布不匹配。")
            elif version in (3, 4, 5, 6, 7):
                action_range = recipe.get("action_range")
                if (gaussian or recipe.get("distribution") != "BetaDistribution"
                        or not isinstance(action_range, (tuple, list))
                        or tuple(action_range) != (-1., 1.)):
                    raise ValueError("Beta检查点动作分布或值域不匹配。")
                if version in (4, 5, 6, 7) and (
                    not isinstance(recipe.get("motion_body"), str) or not recipe["motion_body"]
                ):
                    raise ValueError("第4版蛇形检查点缺少运动坐标参考模块。")
                if version in (5, 6, 7) and (
                    not isinstance(recipe.get("motion_site"), str) or not recipe["motion_site"]
                ):
                    raise ValueError("第5版蛇形检查点缺少运动坐标参考site。")
                if version in (6, 7) and (
                    recipe.get("objective") != ("snake_forward_signed_speed" if version == 7 else "world_x_signed_speed")
                    or recipe.get("observation_reference") != "world_x"
                ):
                    raise ValueError("蛇形检查点的推进任务或观测参考无效。")
            else:
                raise ValueError("不支持的蛇形检查点版本。")
        # 直接检查输出头，避免先加载部分权重后才遇到形状错误。
        heads = [(int(key.split('.')[1]), value) for key, value in actor.items()
                 if key.startswith('mlp.') and key.endswith('.weight')
                 and key.split('.')[1].isdigit()]
        expected_rows = 6 if gaussian else 12
        if not heads:
            raise ValueError("检查点缺少动作输出头。")
        head_index, head_weight = max(heads, key=lambda item: item[0])
        head_bias = actor.get(f'mlp.{head_index}.bias')
        if (head_weight.ndim != 2 or head_weight.shape[0] != expected_rows
                or head_bias is None or head_bias.shape != (expected_rows,)):
            raise ValueError("检查点动作输出头与声明的分布不匹配。")
        frames["observation_reference"] = "world_x" if version in (6, 7) else "command"
        return version, gaussian, log_std, frames

    @staticmethod
    def prepare_playback_cfg(train_cfg, path):
        """构造推理网络前选择真实分布；只为旧模型回放，不用于迁移训练。"""
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        version, gaussian, log_std, _ = SnakeTeacherRunner._describe_checkpoint(checkpoint)
        train_cfg["actor"]["distribution_cfg"] = (
            {"class_name": "GaussianDistribution", "init_std": .25,
             "std_type": "log" if log_std else "scalar",
             "std_range": (.08, .45) if version == 2 else (1e-6, 1e6)}
            if gaussian else {"class_name": "BetaDistribution", "action_range": (-1., 1.)}
        )

    def _recipe(self):
        distribution = self.alg.get_policy().distribution
        if not isinstance(distribution, BetaDistribution):
            raise TypeError("旧高斯策略仅供回放，不能保存为Beta训练检查点。")
        if tuple(distribution.action_range) != (-1., 1.):
            raise ValueError("蛇形Beta动作值域必须为[-1, 1]。")
        cfg = cast(SnakeEnvCfg, self.env.unwrapped.cfg)
        return {"version": 7, "motion_frame": cfg.motion_frame, "motion_body": cfg.head_body,
                "motion_site": cfg.motion_site,
                "objective": "snake_forward_signed_speed", "observation_reference": cfg.observation_reference,
                "head_gyro_frame": cfg.head_gyro_frame,
                "distribution": "BetaDistribution", "action_range": tuple(distribution.action_range)}

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        distribution = self.alg.get_policy().distribution
        if not isinstance(distribution, BetaDistribution) or tuple(distribution.action_range) != (-1., 1.):
            raise ValueError("新蛇形训练只使用原生Beta策略，请从零训练。")
        if cast(SnakeEnvCfg, self.env.unwrapped.cfg).observation_reference != "world_x":
            raise ValueError("旧速度指令观测模型仅支持回放，蛇形前向任务请从零训练。")
        return super().learn(num_learning_iterations, init_at_random_ep_len)

    def save(self, path: str, infos=None) -> None:
        saved_infos = {**(infos or {}), "snake_recipe": self._recipe()}
        manager = getattr(self.env.unwrapped, "command_manager", None)
        term = manager.get_term("motion") if manager is not None else None
        if isinstance(term, SnakeVelocityCommand) and term.cfg.translation_curriculum:
            saved_infos["snake_curriculum"] = term.curriculum_state()
        super().save(path, saved_infos)

    def load(self, path: str, load_cfg=None, strict=True, map_location=None):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        version, gaussian, _, frames = self._describe_checkpoint(checkpoint)
        actor_only = load_cfg is not None and load_cfg.get("actor", False) and not any(
            load_cfg.get(key, False) for key in ("critic", "optimizer", "iteration", "rnd")
        )
        distribution = self.alg.get_policy().distribution
        if not isinstance(distribution, (BetaDistribution, GaussianDistribution)):
            raise TypeError("蛇形检查点仅支持原生Beta或旧高斯分布。")
        if gaussian != isinstance(distribution, GaussianDistribution):
            raise ValueError("高斯和Beta输出头不兼容：新训练请从零开始；旧回放请先prepare_playback_cfg或使用play.py。")
        if not actor_only and (gaussian or version != 7
                              or (checkpoint.get("infos") or {}).get("snake_recipe") != self._recipe()):
            raise ValueError("检查点任务或观测语义不兼容，蛇形前向任务请从零训练。")
        if (actor_only and version in (4, 5, 6, 7)
                and checkpoint["infos"]["snake_recipe"]["motion_body"] != cast(SnakeEnvCfg, self.env.unwrapped.cfg).head_body):
            raise ValueError("检查点运动坐标参考模块与回放配置不一致。")
        if (actor_only and version in (5, 6, 7)
                and checkpoint["infos"]["snake_recipe"]["motion_site"] != cast(SnakeEnvCfg, self.env.unwrapped.cfg).motion_site):
            raise ValueError("检查点运动坐标参考site与回放配置不一致。")
        curriculum = (checkpoint.get("infos") or {}).get("snake_curriculum")
        manager = getattr(self.env.unwrapped, "command_manager", None)
        term = manager.get_term("motion") if manager is not None else None
        if not actor_only and curriculum is not None:
            if not isinstance(term, SnakeVelocityCommand):
                raise ValueError("当前环境缺少蛇形课程指令项")
            term.validate_curriculum_state(curriculum)
        previous_counter = self.env.unwrapped.common_step_counter
        infos = super().load(path, load_cfg, strict, map_location)
        if not actor_only and curriculum is not None and isinstance(term, SnakeVelocityCommand):
            term.load_curriculum_state(curriculum)
        if actor_only:
            env = self.env.unwrapped
            cfg = cast(SnakeEnvCfg, env.cfg)
            env.common_step_counter = previous_counter
            cfg.motion_frame = "legacy_head" if frames["motion_frame"] == "legacy_head" else "body_gravity"
            cfg.head_gyro_frame = "world" if frames["head_gyro_frame"] == "world" else "head"
            cfg.observation_reference = "world_x" if frames["observation_reference"] == "world_x" else "command"
            env.observation_manager.compute(update_history=True)
            if gaussian:
                print("旧高斯蛇形模型回放：保留原分布和观测坐标；新Beta策略需要从零训练。")
            elif version == 3:
                print("旧Beta检查点未记录运动参考模块；当前按foot_0解释速度指令，与原base训练定义可能不同。")
        return infos


class SnakeOmniTeacher(Experiment):
    task_id = TASK_ID
    experiment_name = "zbot_snake_omni_teacher"
    train_envs = 4096
    max_iterations = 3000
    runner_cls = SnakeTeacherRunner

    def create_env(self, play=False):
        return snake_env_cfg(play=play)

    def build_ppo(self):
        cfg = super().build_ppo()
        cfg.actor.distribution_cfg = {
            "class_name": "BetaDistribution", "action_range": (-1., 1.),
        }
        cfg.algorithm.entropy_coef = .002
        return cfg
