from types import SimpleNamespace

import torch

from zbot_rl_mjlab.experiments.walking import FlatWalking, VelocityTracking
from zbot_rl_mjlab.tasks.walking import mdp as tracking


def test_tracking_inherits_flat_without_disturbances():
    for play in (False, True):
        flat = FlatWalking().build_env(play=play)
        cfg = VelocityTracking().build_env(play=play)
        assert cfg.events == flat.events
        assert cfg.scene.sensors == flat.scene.sensors
        assert (cfg.step_frequency_min, cfg.step_frequency_max) == (0.25, 1.0)
        assert cfg.test_frequency is None
        assert cfg.frequency_sampling == "random"
        assert cfg.commands["target_speed"].ranges.lin_vel_x == (0.0, 0.5)
        assert "forward_velocity" not in cfg.rewards
        for name, term in flat.rewards.items():
            if name not in ("forward_velocity", "lateral_velocity"):
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
