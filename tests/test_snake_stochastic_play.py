"""随机回放使用原生采样；默认仍是确定性能力回放。"""

from unittest.mock import Mock

import mjlab
import torch
import tyro
from test_snake_runner_recipe import make_runner

from play import PlayConfig, SampledPlayPolicy
from zbot_rl_mjlab.tasks.snake.env_cfg import snake_env_cfg


def test_cli_default_is_deterministic_and_sampling_is_explicit():
    default = PlayConfig(env=snake_env_cfg(play=True))
    assert not default.stochastic
    cfg = tyro.cli(PlayConfig, args=['--stochastic', 'True'], default=default, config=mjlab.TYRO_FLAGS)
    assert cfg.stochastic


def test_native_beta_sampling_varies_while_mean_and_reset_contract_stay_stable():
    runner, _ = make_runner()
    policy = runner.get_inference_policy(device='cpu')
    observations = runner.env.get_observations()
    with torch.inference_mode():
        deterministic = policy(observations)
        torch.testing.assert_close(policy(observations), deterministic, rtol=0, atol=0)
        sampled_policy = SampledPlayPolicy(policy)
        samples = torch.stack([sampled_policy(observations) for _ in range(32)])
    assert samples.min() >= -1 and samples.max() <= 1
    assert samples.std(dim=0).mean() > .1
    policy.reset = Mock()
    sampled_policy.reset()
    policy.reset.assert_called_once_with()
