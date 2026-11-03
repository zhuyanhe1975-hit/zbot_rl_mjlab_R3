"""Baseline restoration: compare saved 13:11 physics and exercise original controls."""
import enum
import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from zbot_rl_mjlab.experiments.cards import load_card
from zbot_rl_mjlab.tasks.reconfiguration.basic import validate_checkpoint

TASK='Mjlab-Zbot-6dof-Reconfigure-Head-Tail-Docking-Basic'
SAVED=json.loads((Path(__file__).parent/'data/reconfiguration_basic_initial.json').read_text())


def plain(value):
    if isinstance(value,enum.Enum):return value.value
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value


def test_physics_actuators_initial_state_and_rewards_match_initial_saved_run():
    cfg=load_env_cfg(TASK)
    assert plain(asdict(cfg.sim))==SAVED['sim']
    assert cfg.decimation==SAVED['decimation']==4
    expected_magnet = SAVED['magnet'] | {'episode_length_s': 10.}
    assert {k:plain(getattr(cfg,k)) for k in expected_magnet}==expected_magnet
    assert plain(asdict(cfg.actions['joint_pos']))==SAVED['actions']
    robot=cfg.scene.entities['robot']
    assert plain(asdict(robot.articulation))==SAVED['actuators']
    assert plain(asdict(robot.init_state))==SAVED['robot_initial_state']
    assert plain([asdict(g) for g in cfg.scene.terrain.geoms])==SAVED['terrain_geoms']
    assert not cfg.scene.sensors and not cfg.commands
    # Post-restore reward edits are authoritative; the saved physics stays fixed.
    assert set(cfg.rewards) == (set(SAVED['rewards']) - {'body_height'}) | {
        'limit_proximity', 'end_distance_velocity', 'end_alignment_velocity',
    }
    card = load_card('reconfiguration_head_tail_docking_basic')
    assert {k:v.weight for k,v in cfg.rewards.items()} == {
        k:v['weight'] for k,v in card['rewards'].items()
    }
    agent=load_rl_cfg(TASK)
    assert agent.clip_actions==1. and agent.algorithm.gamma==.99
    assert agent.actor.distribution_cfg['init_std']==1. and agent.save_interval==50
    assert agent.num_steps_per_env==24 and agent.max_iterations==500
    assert not agent.resume


def test_abandoned_observation_checkpoints_are_rejected(tmp_path):
    for width in (25,81,59):
        p=tmp_path/f'model_{width}.pt'
        torch.save({'actor_state_dict':{'mlp.0.weight':torch.zeros(256,width)}},p)
        if width==59:validate_checkpoint(p)
        else:
            with pytest.raises(ValueError,match='59维观测'):validate_checkpoint(p)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA original Basic pipeline')
@torch.inference_mode(False)
@torch.enable_grad()
def test_original_speed_integration_policy_update_and_saved_59d_model(tmp_path):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
    cfg=load_env_cfg(TASK);cfg.scene.num_envs=4
    core=ManagerBasedRlEnv(cfg,device='cuda:0')
    env=RslRlVecEnvWrapper(core,clip_actions=1.)
    agent=load_rl_cfg(TASK);agent.upload_model=False;agent.num_steps_per_env=4
    agent.algorithm.num_learning_epochs=1;agent.algorithm.num_mini_batches=1
    try:
        with torch.inference_mode():
            obs,_=env.reset();term=core.action_manager.get_term('joint_pos')
            assert obs['actor'].shape==obs['critic'].shape==(4,59)
            requested=torch.ones(4,6,device='cuda:0')
            term.process_actions(requested)
            torch.testing.assert_close(term.commanded_target,requested*math.pi*.02)
            term.process_actions(requested)
            torch.testing.assert_close(term.commanded_target,requested*math.pi*.04)
            term.process_actions(-requested)
            torch.testing.assert_close(term.commanded_target,requested*math.pi*.02)
            env.reset()
            # Velocity rewards read the physical origins and relative angular velocity.
            from zbot_rl_mjlab.tasks.reconfiguration.basic import mdp
            env.step(torch.zeros_like(requested))
            env.step(requested*.2)
            rates=torch.stack((mdp.distance_closing_speed(core),mdp.alignment_closing_rate(core)),dim=-1)
            assert rates.shape==(4,2) and torch.isfinite(rates).all()
            assert not hasattr(core,'_docking_reward_rates')
            env.reset()
        runner=MjlabOnPolicyRunner(env,asdict(agent),None,'cuda:0')
        runner.learn(1)
        path=tmp_path/'model.pt';runner.save(str(path));validate_checkpoint(path)
        runner.load(str(path),load_cfg={'actor':True})
        with torch.inference_mode():
            action=runner.get_inference_policy()(env.get_observations())
        assert action.shape==(4,6) and torch.isfinite(action).all()
    finally:env.close()
