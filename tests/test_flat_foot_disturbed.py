from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

import zbot_rl_mjlab  # noqa: F401
from zbot_rl_mjlab.experiments import flat_foot_velocity_energy as flat
from zbot_rl_mjlab.experiments.variants import FLAT_FOOT_DISTURBED_TASK_ID

PLAIN = flat.FLAT_FOOT_VELOCITY_ENERGY_TASK_ID
DISTURBED = FLAT_FOOT_DISTURBED_TASK_ID


def test_disturbed_task_adds_events_and_keeps_the_flat_foot_rewards_and_curriculum():
    for play in (False, True):
        plain = load_env_cfg(PLAIN, play=play)
        cfg = load_env_cfg(DISTURBED, play=play)
        added = set(cfg.events) - set(plain.events)
        assert {
            "base_disturbance",
            "randomize_foot_friction",
            "randomize_encoder_bias",
            "randomize_base_com",
        } <= added
        assert set(cfg.rewards) == set(plain.rewards)
        assert set(cfg.curriculum) == set(plain.curriculum)
        assert list(cfg.observations["actor"].terms) == list(
            plain.observations["actor"].terms
        )


def test_disturbed_task_has_its_own_experiment_name():
    assert (
        load_rl_cfg(DISTURBED).experiment_name
        == "zbot_flat_foot_velocity_energy_disturbed"
    )
    assert load_rl_cfg(PLAIN).experiment_name != load_rl_cfg(DISTURBED).experiment_name
