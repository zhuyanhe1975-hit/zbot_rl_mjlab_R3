"""用于回放 Zbot 检查点的 MJLab 风格命令行入口。"""

import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import mjlab
import tyro
from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.viewer import ViserPlayViewer

import zbot_rl_mjlab  # noqa: F401
from zbot_rl_mjlab.cli import CliError, apply_cli, latest_checkpoint
from zbot_rl_mjlab.tasks.snake.env_cfg import SnakeEnvCfg
from zbot_rl_mjlab.tasks.transformation.commanded import TransformCommandCfg
from zbot_rl_mjlab.tasks.transformation.env_cfg import TransformationEnvCfg
from zbot_rl_mjlab.tasks.universal.env_cfg import UniversalEnvCfg
from zbot_rl_mjlab.tasks.walking.env_cfg import ZbotEnvCfg
from zbot_rl_mjlab.viewer import FollowCameraViewer


@dataclass
class PlayConfig:
    env: ManagerBasedRlEnvCfg
    checkpoint_file: str | None = None
    frequency: float | None = None
    target_speed: float | None = None
    log_root: str = "logs/rsl_rl"
    device: str = "cuda:0"
    viewer: str = "native"
    stochastic: bool = False


class SampledPlayPolicy:
    """观察训练式采样探索，保留原生策略的reset接口。"""

    def __init__(self, policy):
        self.policy = policy

    def __call__(self, obs):
        return self.policy(obs, stochastic_output=True)

    def reset(self, *args, **kwargs):
        return self.policy.reset(*args, **kwargs)


def main():
    try:
        sys.argv[1:] = apply_cli()
    except CliError as error:
        sys.exit(f"error: {error}")
    task, remaining = tyro.cli(
        tyro.extras.literal_type_from_choices(
            [t for t in list_tasks() if t.startswith("Mjlab-Zbot")]
        ),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    cfg = tyro.cli(
        PlayConfig,
        args=remaining,
        default=PlayConfig(env=load_env_cfg(task, play=True)),
        config=mjlab.TYRO_FLAGS,
    )
    cfg.env.scene.env_spacing = 1.0
    cfg.env.viewer.enable_shadows = True
    cfg.env.viewer.enable_reflections = True
    if cfg.stochastic and "motion" not in cfg.env.commands:
        raise ValueError("随机采样回放当前仅支持独立蛇形教师任务。")
    if cfg.frequency is not None:
        if not isinstance(cfg.env, ZbotEnvCfg):
            raise ValueError("frequency requires a periodic walking task")
        if cfg.frequency <= 0:
            raise ValueError("frequency must be positive")
        cfg.env.test_frequency = cfg.frequency
    if cfg.target_speed is not None:
        if cfg.target_speed < 0:
            raise ValueError("target-speed must be non-negative")
        command = cfg.env.commands.get("target_speed")
        if not isinstance(command, UniformVelocityCommandCfg):
            raise ValueError("target-speed requires the velocity-tracking task")
        command.ranges.lin_vel_x = (cfg.target_speed, cfg.target_speed)
    periodic = cfg.env if isinstance(cfg.env, ZbotEnvCfg) else None
    if periodic is not None and not (
        0 < periodic.step_frequency_min <= periodic.step_frequency_max
    ):
        raise ValueError("Require 0 < step-frequency-min <= step-frequency-max")
    agent = load_rl_cfg(task)
    checkpoint = (
        Path(cfg.checkpoint_file)
        if cfg.checkpoint_file
        else latest_checkpoint(agent.experiment_name, cfg.log_root)
    )
    print(f"Checkpoint: {checkpoint}")
    if periodic is not None:
        print(
            f"Frequency: {periodic.step_frequency_min}–{periodic.step_frequency_max} Hz"
        )
    if cfg.frequency is not None:
        print(f"Fixed frequency: {cfg.frequency} Hz")
    if cfg.target_speed is not None:
        print(f"Fixed target speed: {cfg.target_speed} m/s")
    # 即使采用固定频率测试，也要保持频率编码与训练时一致。
    if (
        periodic is not None
        and periodic.step_frequency_min == periodic.step_frequency_max
    ):
        periodic.test_frequency = periodic.step_frequency_min
        training_env = load_env_cfg(task)
        if not isinstance(training_env, ZbotEnvCfg):
            raise TypeError("periodic task requires ZbotEnvCfg for training and play")
        periodic.step_frequency_min = training_env.step_frequency_min
        periodic.step_frequency_max = training_env.step_frequency_max
    if cfg.viewer not in ("native", "viser"):
        raise ValueError("viewer must be native or viser")
    env = ManagerBasedRlEnv(cfg.env, device=cfg.device)
    try:
        # MuJoCo 默认阴影裁剪距离仅为 1 m。扩大该距离，避免机器人离开初始原点后
        # 阴影消失。
        model = getattr(env.sim, "_mj_model", None)
        if model is not None:
            model.vis.map.shadowclip = 10.0
            model.vis.map.shadowscale = 0.9
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
        train_cfg = asdict(agent)
        from zbot_rl_mjlab.experiments.snake import SnakeTeacherRunner

        if issubclass(runner_cls, SnakeTeacherRunner):
            SnakeTeacherRunner.prepare_playback_cfg(train_cfg, str(checkpoint))
        runner = runner_cls(wrapped, train_cfg, device=cfg.device)
        runner.load(
            str(checkpoint),
            load_cfg={"actor": True},
            strict=True,
            map_location=cfg.device,
        )
        policy = runner.get_inference_policy(device=cfg.device)
        play_policy = SampledPlayPolicy(policy) if cfg.stochastic else policy
        roll_offsets = getattr(env, "_initial_snake_roll_offset", None)
        if roll_offsets is not None:
            angles = [round(angle * 180 / math.pi, 2) for angle in roll_offsets[:8].tolist()]
            print(f"Initial snake roll offsets (degrees, first {len(angles)} envs): {angles}")
        if "motion" in cfg.env.commands and getattr(env.cfg, "observation_reference", "command") == "world_x":
            print("蛇形前向推进：奖励前向质心速度，惩罚左右/上下速度绝对值与关节构型偏差。")
            print("本任务不使用目标速度指令；蓝色箭头表示蛇形前向，绿色箭头表示实际质心速度。")
            print("Enter 复位，空格暂停，R 切换可视化，\\ 跟随相机。")
        elif "motion" in cfg.env.commands:
            print("回放模式：随机采样探索（不代表已学会能力）。" if cfg.stochastic else "回放模式：确定性均值，不展示训练探索噪声。")
            print("蛇形全向运动：W/S 前后，A/D 侧移，Q/E 左右转，X 平滑停车。")
            if getattr(env.cfg, "motion_frame", "legacy_head") == "body_gravity":
                print(f"运动site={cast(SnakeEnvCfg, env.cfg).motion_site}：左向=重力×site Z轴，前向=左向×向上；绕向上逆时针为正。")
            print("每次平移调整 0.025 m/s，转向调整 0.1 rad/s；启动时静止。")
            print("运动site原点的蓝点为指令基点；W/S/A/D 后出现蓝色平移箭头，Q/E 后出现橙色转向箭头。R 切换可视化显示。")
            print("控制作用于全部播放环境；按键不重置姿态。Enter 复位，空格暂停，\\ 跟随相机。")
            if cfg.viewer == "viser":
                print("蛇形键盘控制仅支持原生查看器，请使用 --viewer native。")
        elif isinstance(cfg.env, UniversalEnvCfg):
            print("综合运动：W/S 前后，A/D 侧移，Q/E 左右转，X 停走；按键逐次调整速度。")
            print("构型：] 双足、[ 蛇形；切换时停车，到位后恢复目标速度。Enter 复位，空格暂停。")
            print("控制作用于全部播放环境；手动模式保留物理状态，数值异常仍会复位。")
        elif "transform" in cfg.env.commands:
            transform_env = cast(TransformationEnvCfg, env.cfg)
            transform = env.command_manager.get_command("transform")
            if transform is None:
                raise ValueError("Transform command is configured but unavailable")
            values = transform[:, 0]
            stand_count = int((values > 0).sum().item())
            snake_count = int((values < 0).sum().item())
            print(f"Initial directions: snake -> stand {stand_count}, biped -> snake {snake_count}")
            print(f"Commands (first 16 envs; +1 stand, -1 snake): {values[:16].tolist()}")
            command_cfg = cast(TransformCommandCfg, cfg.env.commands["transform"])
            if command_cfg.auto_switch_on_success and command_cfg.mode == "source":
                print(f"Automatic stand/snake alternation after {transform_env.success_hold_s:g} s stable near the goal (joint tolerance {command_cfg.auto_switch_joint_tolerance:g} rad).")
            print("Transform controls: ] = stand (+1), [ = snake (-1). Keyboard control locks the command and disables automatic resets; no pose reset.")
        viewer = FollowCameraViewer if cfg.viewer == "native" else ViserPlayViewer
        viewer(wrapped, play_policy).run()
    finally:
        env.close()


if __name__ == "__main__":
    main()
