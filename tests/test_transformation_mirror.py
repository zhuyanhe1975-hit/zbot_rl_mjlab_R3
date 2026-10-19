"""镜像约束探索与真实预训练 Beta 输出头的迁移。"""

from types import SimpleNamespace

import pytest
import torch
from mjlab.envs.mdp.actions import JointPositionAction
from mjlab.tasks.registry import load_rl_cfg
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from zbot_rl_mjlab.robot import DEFAULT_JOINT_POS
from zbot_rl_mjlab.tasks.transformation.actions import (
    MirroredJointTargetRateAction,
    MirroredJointTargetRateActionCfg,
    expand_mirror,
    migrate_beta_mirror_head,
    mirrored_last_action,
    project_mirror,
)
from zbot_rl_mjlab.tasks.transformation.robust import (
    PRETRAINED_CHECKPOINT,
    ROBUST_TASK_ID,
)


@pytest.fixture
def make_mirror_action(monkeypatch):
    def initialize(self, cfg, env):
        self._env = env
        self._target_names = [f"joint{i}" for i in range(1, 7)]
        self._target_ids = list(range(6))
        self._raw_actions = torch.zeros(2, 6)
        self._processed_actions = torch.zeros(2, 6)
        self._entity = SimpleNamespace(data=env.data)

    monkeypatch.setattr(JointPositionAction, "__init__", initialize)

    def make(bounds=None):
        data = SimpleNamespace(default_joint_pos=torch.zeros(2, 6), joint_pos=torch.full((2, 6), 0.25), joint_pos_limits=torch.tensor([-6.28, 6.28]).expand(2, 6, 2).clone())
        env = SimpleNamespace(num_envs=2, device="cpu", step_dt=0.02, data=data)
        cfg = MirroredJointTargetRateActionCfg(entity_name="robot", actuator_names=(".*",), use_default_offset=False, joint_target_bounds=bounds or {})
        return MirroredJointTargetRateAction(cfg, env), env

    return make


def test_standing_goal_and_three_action_expansion_have_exact_mirror_relation():
    goal = torch.tensor(DEFAULT_JOINT_POS)
    torch.testing.assert_close(expand_mirror(goal[:3]), goal)
    value = torch.randn(100, 6)
    symmetric = expand_mirror(project_mirror(value))
    assert (symmetric + symmetric.flip(-1)).eq(0).all()


def test_three_stochastic_coordinates_remain_mirrored_after_joint_limit_clipping(make_mirror_action):
    action, env = make_mirror_action({"joint1": (-1, 2), "joint6": (-0.4, 0.8), "joint2": (-2, 2), "joint5": (-1, 3), "joint3": (-3, 3), "joint4": (-2, 1)})
    assert action.action_dim == 3 and action.raw_action.shape == (2, 3)
    for _ in range(200):
        previous = action.commanded_target.clone()
        action.process_actions(torch.ones(2, 3))
        assert (action.commanded_target - previous).abs().max() <= 0.020001
        assert (action.commanded_target + action.commanded_target.flip(-1)).eq(0).all()
    torch.testing.assert_close(action.commanded_target[0], torch.tensor([0.4, 1, 2, -2, -1, -0.4]))
    for _ in range(300):
        action.process_actions(-torch.ones(2, 3))
    torch.testing.assert_close(action.commanded_target[0], expand_mirror(torch.tensor([-0.8, -2, -1])))
    assert env.data.joint_pos.eq(0.25).all()
    obs_env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: action))
    assert mirrored_last_action(obs_env).shape == (2, 6)
    with pytest.raises(ValueError, match="three actions"):
        action.process_actions(torch.zeros(2, 6))


def test_selected_reset_projects_actual_angles_without_moving_robot(make_mirror_action):
    action, env = make_mirror_action()
    action.process_actions(torch.ones(2, 3))
    before = action.commanded_target.clone()
    env.data.joint_pos[1] = torch.tensor([0.2, 0.3, 0.4, 0.6, 0.7, 0.8])
    actual = env.data.joint_pos.clone()
    action.reset(torch.tensor([1]))
    torch.testing.assert_close(action.commanded_target[0], before[0])
    torch.testing.assert_close(action.commanded_target[1], expand_mirror(project_mirror(actual[1])))
    torch.testing.assert_close(env.data.joint_pos, actual)


def test_incompatible_mirrored_joint_limits_are_rejected(make_mirror_action):
    with pytest.raises(ValueError, match="mirrored target interval"):
        make_mirror_action({"joint1": (1, 2), "joint6": (1, 2)})


def test_real_pretrained_head_migrates_to_three_betas_and_exports_three_actions():
    source = torch.load(PRETRAINED_CHECKPOINT, map_location="cpu", weights_only=False)["actor_state_dict"]
    migrated, changed = migrate_beta_mirror_head(source)
    assert changed and source["mlp.4.weight"].shape[0] == 12
    for name in source.keys() - {"mlp.4.weight", "mlp.4.bias"}:
        torch.testing.assert_close(migrated[name], source[name])
    config = load_rl_cfg(ROBUST_TASK_ID).actor
    obs = TensorDict({"actor": torch.randn(32, 30)}, batch_size=[32])
    actor = MLPModel(obs=obs, obs_groups={"actor": ["actor"]}, obs_set="actor", output_dim=3, hidden_dims=config.hidden_dims, activation=config.activation, obs_normalization=config.obs_normalization, distribution_cfg=config.distribution_cfg)
    actor.load_state_dict(migrated, strict=True)
    actor.eval()
    samples = actor(obs, stochastic_output=True)
    assert samples.shape == (32, 3) and samples.abs().le(1).all()
    assert torch.isfinite(actor.distribution.log_prob(samples)).all()
    torch.testing.assert_close(actor.as_onnx(verbose=False)(obs["actor"]), actor(obs))
    again, changed_again = migrate_beta_mirror_head(migrated)
    assert not changed_again and again is migrated
    torch.testing.assert_close(migrated["mlp.4.bias"][:3], (source["mlp.4.bias"][:3] + source["mlp.4.bias"][9:12].flip(0)) * 0.5)
