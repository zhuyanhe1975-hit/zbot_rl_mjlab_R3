"""Checked joint planning and pure supervised teaching, without PPO."""
import math
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
from tensordict import TensorDict

from zbot_rl_mjlab.planning_guidance import JointPlanTeacher,PlanImitation
from zbot_rl_mjlab.tasks.reconfiguration import plan_guided
from zbot_rl_mjlab.tasks.reconfiguration.planning import ClosurePlanner,load_plan

TASK='Mjlab-Zbot-6dof-Reconfigure-Head-Tail-Docking-Plan-Guided'
BASIC=TASK.replace('-Plan-Guided','-Basic')


def test_independent_identity_and_baseline_configuration():
    cfg,base=load_env_cfg(TASK),load_env_cfg(BASIC)
    agent,base_agent=load_rl_cfg(TASK),load_rl_cfg(BASIC)
    assert cfg.scene.num_envs==16 and cfg.episode_length_s==10.
    assert asdict(cfg.sim)==asdict(base.sim)
    assert asdict(cfg.actions['joint_pos'])==asdict(base.actions['joint_pos'])
    assert cfg.observations==base.observations
    assert not hasattr(base,'pose_hold_s') and 'pose_success' not in base.rewards
    assert 'success' in base.terminations and 'pose_success' in cfg.terminations
    assert 'pose_success' not in load_env_cfg(TASK,play=True).terminations
    assert not cfg.rewards and 'pose_success_fraction' in cfg.metrics
    assert agent.actor.distribution_cfg is None
    assert agent.experiment_name!=base_agent.experiment_name
    assert load_runner_cls(TASK) is not None and load_runner_cls(BASIC) is None
    cfg.actions['joint_pos'].target_limit=1.
    assert load_env_cfg(BASIC).actions['joint_pos'].target_limit==2.


def test_stored_ik_solutions_and_entire_edges_are_valid():
    document=load_plan();planner=ClosurePlanner()
    assert len(document['solutions'])>=2
    for solution in document['solutions']:
        points=np.asarray(solution['waypoints'])
        np.testing.assert_allclose(points[0],0.)
        assert np.abs(points).max()<2.
        assert np.linalg.norm(planner.error(points[-1],solution['quarter'],.001))<1e-4
        assert all(planner.edge_valid(a,b) for a,b in zip(points[:-1],points[1:],strict=True))
    assert np.dot(document['solutions'][0]['waypoints'][-1],document['solutions'][1]['waypoints'][-1])<0


def test_sampling_fallback_goes_around_an_obstacle():
    planner=object.__new__(ClosurePlanner)
    planner.bounds=np.array([[-1.,1.],[-1.,1.]])
    planner.valid=lambda q: np.linalg.norm(q)>.3 and (np.abs(q)<=1.).all()
    start,goal=np.array([-.8,0.]),np.array([.8,0.])
    assert not planner.edge_valid(start,goal)
    path=planner.connect(start,goal,seed=4,max_iterations=1000)
    np.testing.assert_allclose(path[0],start);np.testing.assert_allclose(path[-1],goal)
    assert all(planner.edge_valid(a,b) for a,b in zip(path[:-1],path[1:],strict=True))


def test_teacher_uses_one_branch_observed_target_and_recovers_overshoot():
    document=load_plan();teacher=JointPlanTeacher(document,0,4.,.02,math.pi,'cpu')
    observation=torch.zeros(2,59)
    obs=TensorDict({'actor':observation},batch_size=[2])
    action=teacher(obs)
    goal=teacher.points[-1]
    assert (action[0]*goal).gt(0).all() and action.abs().max()<1.
    observation[:,12:18]=goal
    torch.testing.assert_close(teacher(obs),torch.zeros(2,6),atol=1e-5,rtol=0)
    observation[0,12]=goal[0]-.04
    assert teacher(obs)[0,0]>0
    other=JointPlanTeacher(document,1,4.,.02,math.pi,'cpu')
    observation.zero_()
    assert (teacher(obs)*other(obs)).sum()<0


def test_algorithm_is_actor_only_without_a_ppo_phase():
    from rsl_rl.algorithms import PPO
    assert not issubclass(PlanImitation,PPO)
    assert not hasattr(PlanImitation,'compute_returns')
    assert not hasattr(PlanImitation,'act')


def test_pose_success_requires_geometric_dwell_without_any_weld(monkeypatch):
    cfg=load_env_cfg(TASK)
    env=SimpleNamespace(num_envs=2,device='cpu',step_dt=.02,common_step_counter=0,cfg=cfg)
    g={'delta':torch.zeros(2,3),'distance':torch.tensor([.001,.1]),'speed':torch.zeros(2),
       'dw':torch.zeros(2,3),'q0':torch.tensor([[1.,0.,0.,0.]]).expand(2,-1),
       'q1':torch.tensor([[1.,0.,0.,0.]]).expand(2,-1)}
    monkeypatch.setattr(plan_guided.mdp,'state',lambda env:SimpleNamespace(geometry=lambda:g,reset=lambda ids:None))
    for k in range(24):
        env.common_step_counter=k
        assert not plan_guided.pose_success(env).any()
    env.common_step_counter=24
    assert plan_guided.pose_success(env).tolist()==[True,False]
    env.common_step_counter=25
    assert plan_guided.pose_success(env).tolist()==[True,False]
    plan_guided.reset_guided(env,torch.tensor([0]))
    assert plan_guided.pose_success(env).tolist()==[False,False]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA plan demonstration pipeline')
@torch.inference_mode(False)
@torch.enable_grad()
def test_gpu_pure_imitation_resume_and_actor_only_play(tmp_path,monkeypatch):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    import zbot_rl_mjlab.planning_guidance as guidance
    torch.manual_seed(42)
    cfg=load_env_cfg(TASK);cfg.scene.num_envs=2
    core=ManagerBasedRlEnv(cfg,device='cuda:0');env=RslRlVecEnvWrapper(core,clip_actions=1.)
    agent=load_rl_cfg(TASK);agent.upload_model=False
    agent.algorithm.demo_envs=2;agent.algorithm.batch_size=64;agent.algorithm.validation_interval=100
    from rsl_rl.algorithms import PPO
    monkeypatch.setattr(PPO,'update',lambda *a:pytest.fail('PPO must never run'))
    try:
        runner=load_runner_cls(TASK)(env,asdict(agent),str(tmp_path),'cuda:0')
        runner.learn(2)
        assert runner.alg.completed_updates==2 and not hasattr(runner.alg,'critic')
        assert runner.alg.demo_report['teacher_success_rate']==1.
        assert runner.alg.demo_observations.shape[1]==59
        path=tmp_path/'resume.pt';runner.save(str(path))
        resumed=load_runner_cls(TASK)(env,asdict(agent),None,'cuda:0')
        resumed.load(str(path))
        assert resumed.alg.completed_updates==2
        torch.testing.assert_close(resumed.alg.demo_actions,runner.alg.demo_actions)
        monkeypatch.setattr(guidance,'collect_demonstrations',lambda *a:pytest.fail('Repeated demonstration collection'))
        monkeypatch.setattr(guidance,'load_plan',lambda *a:pytest.fail('Planner accessed during student-only use'))
        resumed.learn(1)
        player=load_runner_cls(TASK)(env,asdict(agent),None,'cuda:0')
        player.load(str(path),load_cfg={'actor':True})
        with torch.inference_mode():
            action=player.get_inference_policy()(env.get_observations())
        assert action.shape==(2,6) and torch.isfinite(action).all()
    finally:env.close()
