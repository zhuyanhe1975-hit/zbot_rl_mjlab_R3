"""为任务添加 ``experiments/addons.py`` 可选扩展的命令行开关。

    python train.py Mjlab-Zbot-6dof-Deploy-Walking --disturbance --domain-randomization

``apply_cli(argv)`` 从 ``argv`` 中移除这些开关，并将指定任务替换为叠加所选扩展的
同一实验。其余参数（``--env.scene.num-envs 4096``、``--agent.max-iterations 500`` 等）
原样交给 mjlab 的解析器。``train.py`` 与 ``play.py`` 均使用此函数，确保训练与回放
使用相同开关时构建相同环境。

开关说明（每项以默认参数添加一个扩展）：

    --disturbance             基座力／力矩冲击
    --domain-randomization    足部摩擦、编码器偏置、基座质心
    --mass-randomization      连杆质量和惯量
    --gain-randomization      PD 增益
    --observation-noise       策略网络观测噪声和偏置
    --action-delay [N]        位置目标延迟 0..N 步到达（默认 2；仅训练时生效）
    --blind-actor             不向策略网络提供 base_lin_vel 和 yaw_error
    --deployment              同时启用以上全部扩展

通过 ``--option=value`` 调参（必须使用等号）：

    --disturbance-fraction=0.5   --disturbance-force=8   --disturbance-torque=0.8
    --friction-range=0.4,1.0     --gain-scale=0.8,1.2    --mass-alpha=0.05
    --noise-scale=2.0 （按比例缩放全部噪声标准差和偏置）

改变策略网络输入的开关（``--blind-actor``、``--deployment``）会使未使用这些开关
训练的检查点无法使用；任务名不会改变，因此使用开关训练的模型必须以相同开关回放。
"""

import sys
from dataclasses import replace
from pathlib import Path

from mjlab.tasks import registry as mjlab_registry

from .experiments import addons as ad
from .experiments.base import variant

SWITCHES = {
    "--disturbance",
    "--domain-randomization",
    "--mass-randomization",
    "--gain-randomization",
    "--observation-noise",
    "--action-delay",
    "--blind-actor",
    "--deployment",
}
OPTIONS = {
    "--disturbance-fraction",
    "--disturbance-force",
    "--disturbance-torque",
    "--friction-range",
    "--gain-scale",
    "--mass-alpha",
    "--noise-scale",
}


class CliError(ValueError):
    pass


def _pair(text, name):
    try:
        low, high = (float(x) for x in text.split(","))
    except ValueError:
        raise CliError(f"{name} needs two numbers like 0.4,1.0, got {text!r}") from None
    if low >= high:
        raise CliError(f"{name}: the first number must be smaller, got {text!r}")
    return low, high


def _number(text, name, minimum=0.0):
    try:
        value = float(text)
    except ValueError:
        raise CliError(f"{name} needs a number, got {text!r}") from None
    if value < minimum:
        raise CliError(f"{name} must be at least {minimum}, got {value}")
    return value


def split_argv(argv):
    """将扩展开关与交给 mjlab 的参数分离。

    返回 ``(rest, switches, options)``；``switches`` 将开关映射到其值
    （``True`` 或 ``--action-delay`` 对应的整数），``options`` 将 ``--name`` 映射到文本值。
    """
    rest, switches, options = [], {}, {}
    i = 0
    while i < len(argv):
        arg = argv[i]
        name, equals, value = arg.partition("=")
        if arg in SWITCHES:
            if arg == "--action-delay" and i + 1 < len(argv) and argv[i + 1].isdigit():
                switches[arg] = int(argv[i + 1])
                i += 1
            else:
                switches[arg] = True
        elif name in OPTIONS:
            if not equals:
                raise CliError(f"{name} needs a value: write {name}=VALUE")
            options[name] = value
        elif arg.startswith(
            ("--disturbance", "--domain", "--mass-", "--gain-", "--noise")
        ):
            known = sorted(SWITCHES | OPTIONS)
            raise CliError(f"unknown option {arg!r}; the add-on options are {known}")
        else:
            rest.append(arg)
        i += 1
    return rest, switches, options


def build_addons(switches, options, defaults=None):
    """Use card defaults, then override only explicitly supplied CLI fields."""
    defaults = defaults or {}
    selected = set(switches)
    if "--deployment" in selected:
        selected |= SWITCHES - {"--deployment"}
    scale = _number(options.get("--noise-scale", "1"), "--noise-scale")
    result = []
    if "--disturbance" in selected:
        default = defaults.get(ad.BaseDisturbance, ad.BaseDisturbance())
        values = {}
        if "--disturbance-fraction" in options:
            fraction = _number(options["--disturbance-fraction"], "--disturbance-fraction")
            if fraction > 1.0:
                raise CliError(f"--disturbance-fraction must be at most 1, got {fraction}")
            values["disturbed_fraction"] = fraction
        for option, field in (("--disturbance-force", "force_range"), ("--disturbance-torque", "torque_range")):
            if option in options:
                magnitude = _number(options[option], option)
                values[field] = (-magnitude, magnitude)
        result.append(replace(default, **values))
    if "--domain-randomization" in selected:
        default = defaults.get(ad.DomainRandomization, ad.DomainRandomization())
        result.append(replace(default, friction_range=_pair(options["--friction-range"], "--friction-range"))
                      if "--friction-range" in options else default)
    if "--mass-randomization" in selected:
        default = defaults.get(ad.MassRandomization, ad.MassRandomization())
        if "--mass-alpha" in options:
            alpha = _number(options["--mass-alpha"], "--mass-alpha")
            default = replace(default, alpha_range=(-alpha, alpha))
        result.append(default)
    if "--gain-randomization" in selected:
        default = defaults.get(ad.GainRandomization, ad.GainRandomization())
        result.append(replace(default, scale=_pair(options["--gain-scale"], "--gain-scale"))
                      if "--gain-scale" in options else default)
    if "--observation-noise" in selected:
        default = defaults.get(ad.ObservationNoise, ad.ObservationNoise())
        result.append(replace(default, std={k: v * scale for k, v in default.std.items()},
                              bias={k: v * scale for k, v in default.bias.items()}))
    if "--action-delay" in selected:
        default = defaults.get(ad.ActionDelay, ad.ActionDelay())
        lag = switches.get("--action-delay", True)
        result.append(replace(default, max_lag=default.max_lag if lag is True else lag))
    if "--blind-actor" in selected:
        result.append(defaults.get(ad.BlindActor, ad.BlindActor()))
    return result


def extended_task_ids():
    """Zbot 实验任务 ID，仅对这些任务应用开关。"""
    return {cls.task_id: cls for cls in _experiments()}


def _experiments():
    from . import EXPERIMENTS

    return EXPERIMENTS


def apply_cli(argv=None):
    """应用 ``argv`` 中的扩展开关，并返回去除这些开关后的 ``argv``。

    首个指向已注册 Zbot 任务的参数决定重建哪个任务。
    未指定开关时不作改动，任务保持注册时的配置。
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    rest, switches, options = split_argv(argv)
    if not switches and not options:
        return rest
    if options and not switches:
        raise CliError(f"{sorted(options)} only tune a switch; add the switch too")
    tasks = extended_task_ids()
    chosen = next((a for a in rest if a in tasks), None)
    if chosen is None:
        raise CliError(
            "add-on switches need a Zbot task name as an argument, e.g. "
            "Mjlab-Zbot-6dof-Deploy-Walking"
        )
    base = tasks[chosen]
    if not getattr(base, "supports_walking_addons", True):
        raise CliError(f"{chosen} does not support walking add-on switches")
    defaults = base().collected_addons()
    card = base().configuration_card()
    if card is not None:
        from .experiments.cards import configure_addons

        defaults = configure_addons(defaults, card)
    new = variant(base, chosen, base.experiment_name,
                  *build_addons(switches, options, {type(addon): addon for addon in defaults}))
    # 沿用同一任务 ID，将已注册任务替换为扩展后的任务。
    mjlab_registry._REGISTRY.pop(chosen, None)
    new().register()
    print(
        f"[INFO] {chosen}: add-ons {[type(a).__name__ for a in new().collected_addons()]}"
    )
    return rest


def latest_checkpoint(experiment_name, log_root: str | Path = "logs/rsl_rl"):
    """log_root/experiment_name 下编号最新的 model_*.pt。"""
    root = Path(log_root) / experiment_name
    candidates = [p for p in root.rglob("model_*.pt") if p.stem[6:].isdigit()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoints under {root}")
    return max(candidates, key=lambda p: p.stat().st_mtime_ns)
