from types import SimpleNamespace

import pytest
import torch

from scripts.train_refined_gpu_incremental import make_pilot_cfg, make_initial_closure_cfg
from zbot_rl_mjlab.tasks.reconfiguration.full_batched_fsm import P
from zbot_rl_mjlab.tasks.reconfiguration import initial_closure_pilot as closure


def test_closure_success_requires_settled_closed_topology_and_no_failure(monkeypatch):
    fsm=SimpleNamespace(phase=torch.tensor([P['LOAD'],P['SETTLE_HEADTAIL'],P['LOAD'],P['LOAD'],P['LOAD']]),
                        stage=torch.tensor([0,0,0,1,0]),target=torch.tensor([0,5,0,0,0]))
    connected=torch.ones(5,6,dtype=torch.bool);connected[2,5]=False
    s=SimpleNamespace(fsm=fsm,connected=lambda:connected)
    monkeypatch.setattr(closure,'state',lambda env:s)
    monkeypatch.setattr(closure,'failed',lambda env:torch.tensor([False,False,False,False,True]))
    env=SimpleNamespace(step_dt=.02,episode_length_buf=torch.zeros(5),max_episode_length=1000)
    assert closure.initial_closure_success(env).tolist()==[True,False,False,False,False]
    assert closure.initial_closure_reward(env).tolist()==[50.,0.,0.,0.,0.]


def test_closure_configuration_is_independent():
    cfg=closure.configure_initial_closure(make_pilot_cfg(4096))
    assert cfg.scene.num_envs==4096 and cfg.sim.mujoco.timestep==.001
    assert cfg.full_stop_after_initial_closure and cfg.episode_length_s==20.
    assert cfg.terminations['success'].func is closure.initial_closure_success
    assert set(cfg.rewards)=={'penalty_failure','time','pose_nearness','penalty_elapsed_time'}
    assert cfg.rewards['penalty_elapsed_time'].weight==-.2
    assert cfg.rewards['pose_nearness'].func is closure.headtail_pose_nearness
    assert cfg.rewards['pose_nearness'].weight==.1
    assert cfg.rewards['penalty_failure'].weight==-20.
    assert cfg.rewards['time'].weight==1.
    assert cfg.full_headtail_alignment_position_tolerance==.001
    assert cfg.is_finite_horizon and not cfg.full_pose_progress_reward
    assert cfg.full_policy_curl_only and cfg.full_velocity_scale==.02
    assert cfg.observations['actor'].terms['closure'].func is closure.initial_closure_observation


def test_elapsed_time_reward_ranks_fast_safe_completion_above_reset(monkeypatch):
    from mjlab.managers.reward_manager import RewardManager
    cfg=make_initial_closure_cfg(1)
    cfg.rewards.pop('pose_nearness')  # isolate terminal time/failure semantics
    bad=torch.zeros(1,dtype=torch.bool);completed=torch.zeros_like(bad)
    monkeypatch.setattr(closure,'failed',lambda env:bad)
    monkeypatch.setattr(closure,'initial_closure_success',lambda env:completed&~bad&(env.episode_length_buf<env.max_episode_length))
    env=SimpleNamespace(num_envs=1,device='cpu',step_dt=.02,
                        episode_length_buf=torch.zeros(1,dtype=torch.long),max_episode_length=1000,
                        max_episode_length_s=20.)
    manager=RewardManager(cfg.rewards,env)
    def episode(steps,fail=False):
        manager.reset();bad.zero_();completed.zero_();rewards=[]
        for step in range(1,steps+1):
            env.episode_length_buf.fill_(step)
            bad.fill_(fail and step==steps)
            completed.fill_(not fail and step==steps)
            rewards.append(float(manager.compute(dt=.02)[0]))
        return sum(rewards),sum(value*.997**i for i,value in enumerate(rewards))
    fast,fast_discounted=episode(282)
    slow,slow_discounted=episode(293)
    last_safe,last_safe_discounted=episode(999)
    early_failure,_=episode(1,fail=True)
    timeout,timeout_discounted=episode(1000)
    assert fast==pytest.approx(20/5.64-.2*5.64) and slow==pytest.approx(20/5.86-.2*5.86)
    assert early_failure==pytest.approx(-20.004) and timeout==pytest.approx(-24.)
    assert fast>slow>last_safe>early_failure>timeout
    assert fast_discounted>slow_discounted>last_safe_discounted>timeout_discounted


def test_waiting_cannot_collect_more_guidance_than_elapsed_cost(monkeypatch):
    from mjlab.managers.reward_manager import RewardManager
    cfg=make_initial_closure_cfg(1)
    monkeypatch.setattr(closure,'initial_closure_success',lambda env:torch.zeros(1,dtype=torch.bool))
    monkeypatch.setattr(closure,'failed',lambda env:torch.zeros(1,dtype=torch.bool))
    # Worst-case guidance income: perfectly near the target at every step.
    cfg.rewards['pose_nearness'].func=lambda env:torch.ones(1)
    env=SimpleNamespace(num_envs=1,device='cpu',step_dt=.02,max_episode_length_s=20.,
                        max_episode_length=1000,episode_length_buf=torch.ones(1,dtype=torch.long))
    manager=RewardManager(cfg.rewards,env)
    assert float(manager.compute(dt=.02)[0])==pytest.approx(-.002)


def test_dense_pose_reward_increases_only_toward_headtail_capture(monkeypatch):
    fsm=SimpleNamespace(phase=torch.tensor([P['CURL'],P['CURL'],P['CAPTURE_HEADTAIL'],P['LOAD'],P['CURL']]),
                        target=torch.tensor([5,5,5,0,0]))
    pose={'gap':torch.tensor([.30,.15,0.,0.,0.]),'angle':torch.tensor([1.5,.75,0.,0.,0.])}
    monkeypatch.setattr(closure,'state',lambda env:SimpleNamespace(fsm=fsm,capture_geometry=lambda:pose))
    monkeypatch.setattr(closure,'failed',lambda env:torch.zeros(5,dtype=torch.bool))
    reward=closure.headtail_pose_nearness(SimpleNamespace(episode_length_buf=torch.zeros(5),max_episode_length=1000))
    torch.testing.assert_close(reward,torch.tensor([torch.exp(torch.tensor(-4.)),torch.exp(torch.tensor(-2.)),1.,0.,0.]))
    assert reward[0]<reward[1]<reward[2]


def test_pose_progress_rewards_approach_and_penalizes_retreat(monkeypatch):
    before=torch.tensor([.2,.8,.5])
    current=torch.tensor([.3,.6,.9])
    pose={'gap':-.15*torch.log(current),'angle':torch.zeros(3)}
    action=SimpleNamespace(pose_potential_before=before,pose_progress_active=torch.tensor([True,True,False]))
    monkeypatch.setattr(closure,'state',lambda env:SimpleNamespace(capture_geometry=lambda:pose))
    env=SimpleNamespace(step_dt=.02,action_manager=SimpleNamespace(get_term=lambda name:action))
    torch.testing.assert_close(closure.headtail_pose_progress(env),torch.tensor([5.,-10.,0.]),atol=1e-5,rtol=0.)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_closure_holds_position_without_starting_release_torque():
    from mjlab.envs import ManagerBasedRlEnv
    cfg=closure.configure_initial_closure(make_pilot_cfg(1));cfg.auto_reset=False
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        with torch.inference_mode():
            env.reset();s=env._disconnect_state;a=env.action_manager.get_term('joint_torque')
            env.sim.data.eq_active[:,s.eq_ids[5]]=True
            s.fsm.phase.fill_(P['LOAD']);s.fsm.stage.zero_();s.fsm.target.zero_()
            a.loading.torque.fill_(5.)
            a._apply_actions_eager()
            gains=env.sim.model.actuator_gainprm._tensor[:,a.native_ids,0]
            torch.testing.assert_close(gains,torch.full_like(gains,80.))
            assert (a.policy_torque==0).all()
    finally:env.close()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_break_and_load_guard_are_penalized_and_reset_without_stopping_batch():
    import warp as wp
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_gpu_env import FullBatchedGpuEnv
    env=FullBatchedGpuEnv(make_initial_closure_cfg(3),device='cuda:0')
    previous=torch.cuda.current_stream(env.device)
    torch.cuda.set_stream(torch.cuda.ExternalStream(wp.get_stream(env.sim.wp_device).cuda_stream,device=env.device))
    try:
        with torch.inference_mode():
            env.reset();s=env._disconnect_state
            s.wrong_break[0]=True;s.safety_margin_violation[1]=True
            _,reward,terminated,truncated,extras=env.step(torch.zeros((3,7),device=env.device))
            assert (terminated|truncated).tolist()==[True,True,False]
            assert (reward[:2]<-19.9).all()
            log=extras['log']
            assert float(log['Metrics/failure_fraction'])==1.
            assert float(log['Penalty/failure'])==-20.
            assert float(log['Reward/time'])==0.
            assert float(log['Penalty/elapsed_time'])==pytest.approx(-.004)
            assert float(log['Reward/episode_return'])==pytest.approx(-20.004)
            assert float(log['Metrics/episode_duration_s'])==pytest.approx(.02)
            assert not any(key.startswith(('Closure/','Episode_Reward/')) for key in log)
            assert not s.wrong_break[0] and not s.safety_margin_violation[1]
            assert env.episode_length_buf.tolist()==[0,0,1]
            _,_,terminated,truncated,_=env.step(torch.zeros((3,7),device=env.device))
            assert not (terminated|truncated).any()
            from mjlab.rl import RslRlVecEnvWrapper
            wrapped=RslRlVecEnvWrapper(env)
            env.episode_length_buf[2]=env.max_episode_length-1
            _,reward,done,extras=wrapped.step(torch.zeros((3,7),device=env.device))
            assert done.tolist()==[0,0,1]
            assert float(reward[2])==pytest.approx(-20.004)
            assert 'time_outs' not in extras  # deadline failure must not bootstrap
    finally:
        env.close();torch.cuda.set_stream(previous)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_pruned_internal_pairs_preserve_early_motion_and_headtail_candidate():
    import warp as wp
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_gpu_env import FullBatchedGpuEnv
    original_cfg=make_initial_closure_cfg(2)
    original_cfg.full_initial_closure_collision_pruning=False
    original=FullBatchedGpuEnv(original_cfg,device='cuda:0')
    optimized=FullBatchedGpuEnv(make_initial_closure_cfg(2),device='cuda:0')
    previous=torch.cuda.current_stream(original.device)
    shared=torch.cuda.ExternalStream(wp.get_stream(original.sim.wp_device).cuda_stream,device=original.device)
    try:
        torch.cuda.set_stream(shared)
        with torch.inference_mode():
            original.reset();optimized.reset()
            before=original.sim.wp_model.nxn_pairid_filtered.numpy()
            after=optimized.sim.wp_model.nxn_pairid_filtered.numpy()
            removed={int(original.sim.mj_model.pair(f'open_face_{i}').id) for i in range(5)}
            assert len(before)-len(after)==5
            assert not any(int(pair[0]) in removed for pair in after)
            assert any(int(pair[0])==original.sim.mj_model.pair('open_face_5').id for pair in after)
            action=torch.zeros((2,7),device=original.device);action[1]=.2
            successes=torch.zeros(2,device=original.device,dtype=torch.long)
            original_successes=torch.zeros_like(successes)
            for _ in range(120):  # before first head-tail approach/collision
                obs1,r1,d1,t1,_=original.step(action)
                obs2,r2,d2,t2,_=optimized.step(action)
                # Contact ordering changes solver roundoff and subsequent
                # trajectories. The contract here is preserved candidates,
                # finite physics, successful topology and automatic reset.
                assert torch.isfinite(original.sim.data.qpos._tensor).all()
                assert torch.isfinite(optimized.sim.data.qpos._tensor).all()
                assert torch.isfinite(obs2['actor']).all()
                assert not original.termination_manager.get_term('failure').any()
                assert not optimized.termination_manager.get_term('failure').any()
                assert not t1.any() and not t2.any()
                original_successes+=original.termination_manager.get_term('success').long()
                successes+=optimized.termination_manager.get_term('success').long()
            assert torch.equal(successes,original_successes)
            assert not optimized._disconnect_state.phase_contacts.any()
    finally:
        original.close();optimized.close();torch.cuda.set_stream(previous)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_real_headtail_geometry_overlap_is_detected_by_gpu():
    import warp as wp
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_gpu_env import FullBatchedGpuEnv
    env=FullBatchedGpuEnv(make_initial_closure_cfg(1),device='cuda:0')
    previous=torch.cuda.current_stream(env.device)
    torch.cuda.set_stream(torch.cuda.ExternalStream(wp.get_stream(env.sim.wp_device).cuda_stream,device=env.device))
    try:
        with torch.inference_mode():
            env.reset();action=torch.zeros((1,7),device=env.device)
            env.step(action);env.step(action)
            s=env._disconnect_state;d=env.sim.data;m=env.sim.mj_model
            head=int(m.jnt_qposadr[m.joint('module_0/module_root').id])
            tail=int(m.jnt_qposadr[m.joint('module_5/module_root').id])
            d.qpos[:,tail:tail+7]=d.qpos[:,head:head+7]
            env.sim.forward()
            s.measurement_graph.replay()
            assert s.headtail_penetration.item()
            assert s.max_headtail_depth.item()>0
    finally:
        env.close();torch.cuda.set_stream(previous)

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_headtail_contact_requires_aligned_sites_and_preserves_other_collisions():
    import warp as wp
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_gpu_env import FullBatchedGpuEnv
    env=FullBatchedGpuEnv(make_initial_closure_cfg(5),device='cuda:0')
    previous=torch.cuda.current_stream(env.device)
    torch.cuda.set_stream(torch.cuda.ExternalStream(wp.get_stream(env.sim.wp_device).cuda_stream,device=env.device))
    try:
        with torch.inference_mode():
            env.reset();action=torch.zeros((5,7),device=env.device)
            env.step(action);env.step(action)
            s=env._disconnect_state;a=env.action_manager.get_term('joint_torque')
            d=env.sim.data;c=d.contact;m=env.sim.mj_model
            pair=m.pair('open_face_5').id;internal=m.pair('open_face_0').id
            source=int(s.capture_port_ids[5]);destination=int(s.port_ids[0,1])
            d.site_xpos[:,source]=d.site_xpos[:,destination]+torch.tensor([[.05,0,0],[.15,0,0],[0,0,0],[0,0,0],[0,0,0]],device=env.device)
            d.site_xmat[:,source]=d.site_xmat[:,destination]
            d.site_xmat[3,source]=d.site_xmat[3,destination]@torch.diag(torch.tensor([-1.,-1.,1.],device=env.device))
            c.geom[:5]=torch.tensor([[int(m.pair_geom1[pair]),int(m.pair_geom2[pair])],
                                     [int(m.pair_geom2[pair]),int(m.pair_geom1[pair])],
                                     [int(m.pair_geom1[pair]),int(m.pair_geom2[pair])],
                                     [int(m.pair_geom1[pair]),int(m.pair_geom2[pair])],
                                     [int(m.pair_geom1[internal]),int(m.pair_geom2[internal])]],device=env.device)
            c.worldid[:5]=torch.arange(5,device=env.device)
            c.dist[:5]=torch.tensor([-.001,-.003,-.00005,0.,-.003],device=env.device)
            c.type[:5]=torch.tensor([0,1,0,0,1],device=env.device)
            d.nacon[0]=5
            s.measurement_graph.replay()
            assert s.headtail_penetration.tolist()==[True,True,False,True,False]
            assert s.contact_violation.tolist()==[True,True,False,True,True]
            assert float(s.max_headtail_depth[1])==pytest.approx(.003)
            # A previously detected collision remains latched if sites later align.
            d.site_xpos[1,source]=d.site_xpos[1,destination]
            c.dist[1]=.001
            s.measurement_graph.replay()
            assert s.headtail_penetration[1]
            a.actuation_graph.replay()
            gains=env.sim.model.actuator_gainprm._tensor[:,a.native_ids,0]
            assert (gains[[0,1,3]]==0).all() and (gains[2]>0).all()
            _,reward,terminated,truncated,extras=env.step(action)
            assert (terminated|truncated).tolist()==[True,True,False,True,True]
            assert float(reward[0])==pytest.approx(-20.004)
            assert float(extras['log']['Metrics/headtail_collision_fraction'])==.75
            assert float(extras['log']['Metrics/headtail_penetration_batch_peak_mm'])==pytest.approx(3.)
            assert float(extras['log']['Reward/time'])==0
            assert not s.headtail_penetration[0] and float(s.max_headtail_depth[0])==0
            assert env.episode_length_buf.tolist()==[0,0,3,0,0]
            _,_,terminated,truncated,_=env.step(action)
            assert not (terminated|truncated).any()
    finally:
        env.close();torch.cuda.set_stream(previous)
