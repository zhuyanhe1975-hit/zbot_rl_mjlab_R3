import pytest
from mjlab.tasks import registry
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import zbot_rl_mjlab  # noqa: F401
from zbot_rl_mjlab import cli
from zbot_rl_mjlab.experiments import addons as ad
from zbot_rl_mjlab.tasks.walking.actions import DelayedJointPositionActionCfg

TASK = "Mjlab-Zbot-6dof-Velocity-Energy-Tracking-Walking"
NATIVE = ["--env.scene.num-envs", "4096", "--agent.max-iterations", "500"]


@pytest.fixture(autouse=True)
def restore_registry():
    """apply_cli 会替换已注册任务；每次测试后都应恢复注册表。"""
    saved = dict(registry._REGISTRY)
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)


def events(task=TASK, play=False):
    return list(load_env_cfg(task, play=play).events)


def test_no_switches_leaves_everything_untouched():
    before = events()
    assert cli.apply_cli([TASK, *NATIVE]) == [TASK, *NATIVE]
    assert events() == before == ["reset_scene_to_default"]


def test_switches_are_removed_and_mjlab_arguments_are_passed_through_in_order():
    argv = [TASK, "--disturbance", *NATIVE, "--domain-randomization"]
    assert cli.apply_cli(argv) == [TASK, *NATIVE]


def test_disturbance_and_domain_randomization_switches():
    cli.apply_cli([TASK, "--disturbance", "--domain-randomization"])
    assert events() == [
        "reset_scene_to_default",
        "base_disturbance",
        "randomize_foot_friction",
        "randomize_encoder_bias",
        "randomize_base_com",
    ]
    assert events(play=True) == events()  # 回放使用相同的物理设置


def test_tuning_options():
    cli.apply_cli(
        [
            TASK,
            "--disturbance",
            "--disturbance-fraction=1.0",
            "--disturbance-force=8",
            "--domain-randomization",
            "--friction-range=0.5,0.9",
        ]
    )
    cfg = load_env_cfg(TASK)
    push = cfg.events["base_disturbance"].params
    assert push["disturbed_fraction"] == 1.0
    assert push["force_range"] == (-8.0, 8.0)
    assert cfg.events["randomize_foot_friction"].params["ranges"] == (0.5, 0.9)


def test_action_delay_with_and_without_a_value():
    cli.apply_cli([TASK, "--action-delay", "4"])
    assert load_env_cfg(TASK).actions["joint_pos"].max_lag == 4
    cli.apply_cli([TASK, "--action-delay"])
    assert load_env_cfg(TASK).actions["joint_pos"].max_lag == ad.MAX_ACTION_LAG
    assert load_env_cfg(TASK, play=True).actions["joint_pos"].max_lag == 0


def test_deployment_switch_equals_the_deployment_preset():
    cli.apply_cli([TASK, "--deployment"])
    cfg = load_env_cfg(TASK)
    assert not set(ad.PRIVILEGED_TERMS) & set(cfg.observations["actor"].terms)
    assert isinstance(cfg.actions["joint_pos"], DelayedJointPositionActionCfg)
    assert {"randomize_mass_inertia", "randomize_pd_gains", "base_disturbance"} <= set(
        cfg.events
    )


def test_noise_scale_multiplies_every_noise_level():
    cli.apply_cli([TASK, "--observation-noise", "--noise-scale=2"])
    actor = load_env_cfg(TASK).observations["actor"].terms
    assert actor["joint_vel"].noise.std == 2 * ad.NOISE_STD["joint_vel"]


def test_an_experiment_with_its_own_addons_gets_them_replaced_not_duplicated():
    task = "Mjlab-Zbot-6dof-Deploy-Walking"
    default = load_env_cfg(task).events["base_disturbance"].params["disturbed_fraction"]
    cli.apply_cli([task, "--disturbance", "--disturbance-fraction=1.0"])
    cfg = load_env_cfg(task)
    assert default == 0.25
    assert cfg.events["base_disturbance"].params["disturbed_fraction"] == 1.0
    # 部署任务的其他配置均应保留。
    assert not set(ad.PRIVILEGED_TERMS) & set(cfg.observations["actor"].terms)
    assert "randomize_mass_inertia" in cfg.events


def test_the_experiment_name_and_other_settings_do_not_change():
    before = load_rl_cfg(TASK)
    cli.apply_cli([TASK, "--deployment"])
    after = load_rl_cfg(TASK)
    assert after.experiment_name == before.experiment_name
    assert after.max_iterations == before.max_iterations


@pytest.mark.parametrize(
    "argv, message",
    [
        ([TASK, "--disturbance-fraction=0.5"], "only tune a switch"),
        (["--disturbance"], "need a Zbot task name"),
        ([TASK, "--disturbance", "--disturbance-fraction"], "needs a value"),
        ([TASK, "--disturbance", "--disturbance-fraction=2"], "at most 1"),
        ([TASK, "--domain-randomization", "--friction-range=0.9,0.4"], "smaller"),
        ([TASK, "--domain-randomization", "--friction-range=abc"], "two numbers"),
        ([TASK, "--disturbance", "--disturbance-forces=3"], "unknown option"),
    ],
)
def test_bad_input_is_rejected_with_a_clear_message(argv, message):
    with pytest.raises(cli.CliError, match=message):
        cli.apply_cli(argv)


def test_the_registered_task_is_not_changed_by_a_failed_command():
    before = events()
    with pytest.raises(cli.CliError):
        cli.apply_cli([TASK, "--disturbance", "--disturbance-fraction=2"])
    assert events() == before
