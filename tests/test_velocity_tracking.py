from types import SimpleNamespace

import torch
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from zbot_rl_mjlab.experiments.walking import FlatWalking, VelocityTracking
from zbot_rl_mjlab.tasks.walking import mdp as tracking
from zbot_rl_mjlab.tasks.walking.env_cfg import ZbotEnvCfg


def test_tracking_inherits_flat_without_disturbances():
    for play in (False, True):
        flat = FlatWalking().build_env(play=play)
        cfg = VelocityTracking().build_env(play=play)
        assert isinstance(cfg, ZbotEnvCfg)
        assert cfg.events == flat.events
        assert cfg.scene.sensors == flat.scene.sensors
        assert (cfg.step_frequency_min, cfg.step_frequency_max) == (0.25, 1.0)
        assert cfg.test_frequency is None
        assert cfg.frequency_sampling == "random"
        command = cfg.commands["target_speed"]
        assert isinstance(command, UniformVelocityCommandCfg)
        assert command.ranges.lin_vel_x == (0.0, 0.5)
        assert "forward_velocity" not in cfg.rewards
        # The joint-tracking archive keeps its own tuned rewards while the
        # Walking parent uses its older best recipe.
        archive_weights = {
            "swing_clearance": 2.0,
            "soft_landing": -0.001,
            "joint_acceleration": -0.001,
        }
        for name, weight in archive_weights.items():
            assert cfg.rewards[name].weight == weight
        for name, term in flat.rewards.items():
            if name not in {"forward_velocity", "lateral_velocity", *archive_weights}:
                assert cfg.rewards[name] == term
        assert (
            cfg.rewards["lateral_velocity"].weight
            == flat.rewards["lateral_velocity"].weight
        )
        for group in ("actor", "critic"):
            assert list(cfg.observations[group].terms) == [
                *flat.observations[group].terms,
                "target_speed",
            ]


def test_tracking_penalizes_under_and_overspeed(monkeypatch):
    env = SimpleNamespace(
        command_manager=SimpleNamespace(
            get_command=lambda name: torch.tensor([[0.15, 0.0, 0.0]]).repeat(4, 1)
        )
    )
    speeds = torch.tensor([[0.15], [0.10], [0.20], [-0.15]])
    monkeypatch.setattr(tracking, "base_lin_vel", lambda env: speeds)
    score = tracking.track_forward_speed(env)
    assert score[0] == 1.0
    torch.testing.assert_close(score[1], score[2])
    assert score[3] < score[1] < score[0]
