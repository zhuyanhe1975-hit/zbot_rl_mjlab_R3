"""Immediate Gaussian-mean optimization without PPO, a critic or demonstrations."""
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from zbot_rl_mjlab.static_gaussian import (GaussianMeanCfg,GaussianTorquePolicy,
                                         StaticGaussianRunner,StaticGaussianRunnerCfg,score_gradient,MODE)


def test_gaussian_score_gradient_recovers_known_linear_reward_direction():
    std=torch.full((6,),.02)
    noise=torch.cat((torch.eye(6),-torch.eye(6)))*(.02*6**.5)
    true=torch.tensor([1.,-2.,3.,-.5,0.,4.])
    score=3.+noise@true
    assert torch.allclose(score_gradient(noise,score,std),true,atol=1e-5)
    assert torch.allclose(score_gradient(noise,score+100.,std),true,atol=2e-3)
    assert (score_gradient(noise,torch.zeros(12),std)==0).all()
    with pytest.raises(ValueError):score_gradient(noise,torch.full((12,),float('nan')),std)


def fake_runner():
    env=SimpleNamespace(unwrapped=SimpleNamespace(cfg=SimpleNamespace(enable_weld_failure=False,
                       auto_reset=True),num_envs=4))
    cfg={'algorithm':asdict(GaussianMeanCfg()),'seed':42,'save_interval':10}
    return StaticGaussianRunner(env,cfg,device='cpu')


def test_checkpoint_restores_gaussian_mean_rng_and_rejects_ppo(tmp_path):
    r=fake_runner()
    assert not hasattr(r,'alg')  # no PPO/critic/storage is constructed
    r.policy.mean.fill_(.1);r.policy.std.fill_(.005)
    r.completed_updates=7;r.current_learning_iteration=6
    r.best_mean=r.policy.mean.clone();r.best_report={'score':.01}
    path=tmp_path/'model.pt';r.save(path)
    expected=torch.randn((4,6),generator=r.generator)
    restored=fake_runner();restored.load(path)
    assert restored.completed_updates==7
    assert torch.equal(restored.policy.mean,r.policy.mean)
    assert torch.equal(restored.policy.std,r.policy.std)
    assert torch.equal(torch.randn((4,6),generator=restored.generator),expected)
    assert restored.best_report==r.best_report
    obs={'actor':torch.zeros(3,31)}
    assert torch.equal(restored.get_inference_policy()(obs),torch.full((3,6),.1))
    old=tmp_path/'old.pt';torch.save({'infos':{'disconnect_control_mode':'old_ppo'}},old)
    with pytest.raises(ValueError,match='旧PPO'):restored.load(old)


def test_safety_failure_shrinks_exploration_without_changing_rewards():
    r=fake_runner();calls=[]
    def evaluate(actions,steps):
        calls.append(actions.clone())
        fail=len(calls)==1
        return actions[:,0].clone(),{'failure_rate':float(fail)}
    r.evaluate=evaluate
    gradient,report=r.sample_gradient()
    assert len(calls)==2 and report['failure_rate']==0
    assert torch.all(r.policy.std==.01)
    assert torch.isfinite(gradient).all()


def test_gpu_training_save_resume_and_independent_constant_policy(tmp_path,monkeypatch):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    import zbot_rl_mjlab
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
    from rsl_rl.algorithms import PPO
    def forbidden(*args,**kwargs):raise AssertionError('PPO must not be constructed')
    monkeypatch.setattr(PPO,'__init__',forbidden)
    task='Mjlab-Zbot-6dof-Reconfigure-Disconnect-Torque'
    cfg=load_env_cfg(task);cfg.scene.num_envs=32;cfg.enable_weld_failure=False;cfg.jacobian_feedforward=False;cfg.separation_velocity_pd=False;cfg.separation_distance_jacobian=False
    from mjlab.actuator import IdealPdActuatorCfg
    for i,e in enumerate(cfg.scene.entities.values()):
        e.articulation.actuators=(IdealPdActuatorCfg(target_names_expr=(f'joint{i+1}',),stiffness=0.,damping=1.5,effort_limit=20.,armature=.001),)
    core=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        env=RslRlVecEnvWrapper(core,clip_actions=1.)
        agent=asdict(StaticGaussianRunnerCfg());agent['algorithm']['validation_steps']=20
        agent['algorithm']['sample_steps']=10
        agent['algorithm']['initial_checkpoint']=''
        runner=StaticGaussianRunner(env,agent,str(tmp_path),device='cuda:0')
        _,initial=runner.evaluate(runner.policy.mean,20)
        runner.learn(4)
        assert runner.completed_updates==4
        assert runner.best_report['failure_rate']==0.
        assert runner.best_report['score']>initial['score']
        assert all(k.startswith('penalty_') for k in runner.best_report['reward_terms'] if 'penalty' in k)
        restored=StaticGaussianRunner(env,agent,device='cuda:0')
        restored.load(tmp_path/'model_3.pt')
        assert torch.equal(restored.policy.mean,runner.policy.mean)
        assert restored.completed_updates==4
        obs,_=env.reset();policy=restored.get_inference_policy()
        with torch.inference_mode():
            action=policy(obs)
            assert action.shape==(32,6)
            assert torch.equal(action,restored.policy.mean.expand(32,-1))
            obs,_,done,_=env.step(action)
            assert not done.any()
        assert 'target_loading' in runner.best_report['reward_terms']
        assert 'penalty_failure' in runner.best_report['reward_terms']
    finally:core.close()


def test_phase_policy_freezes_loading_and_switches_after_target_weld_opens():
    policy=GaussianTorquePolicy(torch.full((6,),.01),.002,separation_phase=True,target_connector=5)
    policy.loading_mean.fill_(.2)
    obs={'actor':torch.zeros(3,31)};obs['actor'][0,29]=1.
    deterministic=policy(obs)
    assert (deterministic[0]==.2).all()
    assert (deterministic[1:]==.01).all()
    sampled=policy(obs,stochastic_output=True)
    assert (sampled[0]==.2).all()  # loading is never re-explored
    assert not torch.equal(sampled[1:],deterministic[1:])


def test_separation_progress_rewards_do_not_pay_for_waiting_or_closed_cycles():
    from zbot_rl_mjlab.tasks.reconfiguration.disconnect.state import ConnectorState,separation_reward,separation_regression
    s=ConnectorState.__new__(ConnectorState)
    s.max_target_gap=torch.zeros(1)
    s.cfg=SimpleNamespace(target_connector=5,separation_distance=.05,minimum_other_margin=.2,
                          post_separation_wait_s=2.)
    s.env=SimpleNamespace(physics_dt=.005)
    s.other_ids=list(range(5));s.wait_count=torch.zeros(1,dtype=torch.long)
    s.waiting=torch.zeros(1,dtype=torch.bool);s.gap_progress=torch.zeros(1)
    s.separation_progress_delta=torch.zeros(1);s.safety_margin_violation=torch.zeros(1,dtype=torch.bool)
    s.just_succeeded=torch.zeros(1,dtype=torch.bool);s.succeeded=torch.zeros(1,dtype=torch.bool)
    s.wrong_break=torch.zeros(1,dtype=torch.bool)
    gap=torch.zeros(1);s.geometry=lambda:{'distance':gap};s.other_margin=lambda:torch.ones(1)
    active=torch.ones(1,6,dtype=torch.bool);active[:,5]=False
    env=SimpleNamespace(_disconnect_state=s,step_dt=.02)
    net=0.
    for distance in (.01,.03,.02,.04,0.):
        gap.fill_(distance);s.separation_progress_delta.zero_();s.update_separation(active)
        net+=float((5*separation_reward(env)-5*separation_regression(env))*.02)
    assert abs(net)<1e-6
    gap.fill_(.06);s.separation_progress_delta.zero_();s.update_separation(active)
    assert s.waiting.all()
    assert torch.allclose(separation_reward(env)*.02,torch.ones(1))
    gap.zero_();s.separation_progress_delta.zero_();s.update_separation(active)
    assert separation_reward(env)==0 and separation_regression(env)==0


def test_static_checkpoint_only_initializes_frozen_loading_in_new_phase(tmp_path):
    source=fake_runner();source.policy.mean.fill_(.2)
    source.save(tmp_path/'static.pt')
    def dynamic():
        env=SimpleNamespace(unwrapped=SimpleNamespace(cfg=SimpleNamespace(enable_weld_failure=True,
            auto_reset=True,target_connector=5),num_envs=4))
        settings=asdict(GaussianMeanCfg());settings['initial_checkpoint']=str(tmp_path/'static.pt')
        return StaticGaussianRunner(env,{'algorithm':settings,'seed':42,'save_interval':10},device='cpu')
    runner=dynamic();runner.initialize_mean()
    assert (runner.policy.loading_mean==.2).all()
    assert (runner.policy.mean==0.).all()
    runner.policy.mean.fill_(.01);runner.save(tmp_path/'phase.pt')
    restored=dynamic();restored.load(tmp_path/'phase.pt')
    assert torch.equal(restored.policy.loading_mean,runner.policy.loading_mean)
    assert torch.equal(restored.policy.mean,runner.policy.mean)
    with pytest.raises(ValueError):runner.load(tmp_path/'static.pt')
    with pytest.raises(ValueError):fake_runner().load(tmp_path/'phase.pt')
