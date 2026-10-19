"""保护刷新接入PPO后仍保存原成功标签，并可完整恢复固定基础前缀。"""
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from zbot_rl_mjlab.distillation import MultiTeacherPPO, universal_ppo_cfg


def algorithm():
    cfg=asdict(universal_ppo_cfg());cfg.update(multi_gpu=None,num_steps_per_env=2)
    for name in ('actor','critic'):
        cfg[name]['hidden_dims']=(16,)
        for option in ('cnn_cfg','rnn_type','rnn_hidden_dim','rnn_num_layers'):
            cfg[name].pop(option,None)
    cfg['critic'].pop('distribution_cfg')
    cfg['algorithm'].update(anchor_max_samples=16,distillation_updates=1,distillation_batch_size=8)
    obs=TensorDict({'actor':torch.zeros(8,185),'critic':torch.zeros(8,53)},batch_size=[8])
    return cast(MultiTeacherPPO,MultiTeacherPPO.construct_algorithm(obs,cast(VecEnv,SimpleNamespace(num_envs=8,num_actions=6)),cfg,'cpu'))


def packet(n,value,prep=False):
    obs=TensorDict({'actor':torch.zeros(n,185),'critic':torch.zeros(n,53)},batch_size=[n])
    obs['critic'][:,38]=0 if prep else 1
    obs['critic'][:,39]=1
    obs['critic'][:,43]=.08
    return (obs,torch.full((n,6),value),torch.ones(n),torch.arange(n)%3,torch.arange(n)%5==0)


def seed(alg):
    alg._anchor=packet(15,.23)
    alg._anchor_observation_source='validation_trajectory'
    alg.ppo_start_iteration=2
    alg.distillation_iteration=10


def test_refresh_preserves_base_labels_and_full_resume_restores_prefix():
    alg=algorithm();seed(alg)
    alg.refresh_anchor(packet(30,-.4,prep=True))
    assert alg._anchor_foundation is not None
    assert alg._anchor is not None and len(alg._anchor[1])==16
    assert alg._anchor_foundation[1].eq(.23).all()
    base=deepcopy(alg._anchor_foundation)
    alg.refresh_anchor(packet(30,.7,prep=True))
    for before,after in zip(base,alg._anchor_foundation):torch.testing.assert_close(before,after)
    state=alg.save();assert state['multi_teacher']['anchor_foundation_count']==8
    assert 'anchor_foundation' not in state['multi_teacher']
    restored=algorithm();restored.load(state)
    assert restored._anchor_refresh_count==2
    assert restored._anchor_foundation is not None
    for before,after in zip(base,restored._anchor_foundation):torch.testing.assert_close(before,after)
    restored.load(state,{'actor':True})
    assert restored._anchor is None and restored._anchor_foundation is None and restored._anchor_refresh_count==0


def test_refresh_rejects_nonfinite_rows_and_does_not_create_unvalidated_anchor():
    alg=algorithm();incoming=packet(30,-.4,prep=True)
    alg.refresh_anchor(incoming);assert alg._anchor is None
    seed(alg)
    incoming[1][0,0]=float('nan')
    alg.refresh_anchor(incoming)
    assert alg._anchor is not None and torch.isfinite(alg._anchor[1]).all()


def test_invalid_prefix_rejected_before_weights_are_modified():
    alg=algorithm();seed(alg);saved=alg.save()
    saved['multi_teacher']['anchor_foundation_count']=999
    original=deepcopy(alg.actor.state_dict())
    with pytest.raises(ValueError,match='前缀'):alg.load(saved)
    for key,value in original.items():torch.testing.assert_close(alg.actor.state_dict()[key],value)
