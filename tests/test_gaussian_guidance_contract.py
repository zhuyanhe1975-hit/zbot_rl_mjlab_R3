"""Gaussian labels retain intrinsic mirror support without legacy rate clipping."""
from types import SimpleNamespace

import torch
from test_guided_ppo import algorithm, observations

from zbot_rl_mjlab.gaussian_action_models import GaussianActionRoutedTaskNormalizedMLP
from zbot_rl_mjlab.guided_teachers import GuidedTeacherEnsemble
from zbot_rl_mjlab.tasks.transformation.actions import expand_mirror


def test_gaussian_guidance_accepts_unbounded_labels_for_every_task():
    obs = observations((0, 1, 2, 3) * 4)
    alg = algorithm(obs, steps=1, actor_cls=GaussianActionRoutedTaskNormalizedMLP,
                    guidance_tasks=(0, 1, 2, 3))
    labels = torch.full((len(obs), 6), 2.5)
    for task in (1, 3):
        ids = obs['actor'][:, 39] == task
        labels[ids] = expand_mirror(labels[ids, :3])
    alg._cache_guidance(obs, labels, torch.ones(len(obs)))
    assert alg._guidance_counts == [4, 4, 4, 4]
    alg._clear_guidance()
    labels[obs['actor'][:, 39] == 3] = 2.5
    alg._cache_guidance(obs, labels, torch.ones(len(obs)))
    assert alg._guidance_counts == [4, 4, 4, 0]


def test_transform_expert_labels_are_native_three_rates_not_six_alignment_rates():
    ensemble = GuidedTeacherEnsemble.__new__(GuidedTeacherEnsemble)
    action = SimpleNamespace(rate_cfg=SimpleNamespace(max_target_speed=2*torch.pi),
                             commanded_target=torch.tensor([[.2, .3, .4, .1, -.5, -.3]]))
    ensemble.env = SimpleNamespace(cfg=SimpleNamespace(mirrored_transform_actions=True),
                                  action_manager=SimpleNamespace(get_term=lambda _: action))
    ensemble.configs = {'transform': SimpleNamespace(actions={'joint_pos': SimpleNamespace(max_target_speed=2*torch.pi)})}
    source = torch.tensor([[.1, -.2, .3]])
    assert torch.equal(ensemble._convert('transform', source), expand_mirror(source))
