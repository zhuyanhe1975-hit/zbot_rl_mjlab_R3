"""Contracts for the independent torque task; no learning claims."""
import torch
from mjlab.tasks.registry import load_env_cfg as _load_env_cfg
import zbot_rl_mjlab
from zbot_rl_mjlab.tasks.reconfiguration.disconnect.model import load_snapshot,module_spec
from zbot_rl_mjlab.tasks.reconfiguration.disconnect.state import port_wrench

TASK='Mjlab-Zbot-6dof-Reconfigure-Disconnect-Torque'

def load_env_cfg(*args,**kwargs):
    # Retain the earlier direct-action physics contracts without the new baseline.
    cfg=_load_env_cfg(*args,**kwargs);cfg.jacobian_feedforward=False;cfg.separation_velocity_pd=False;cfg.separation_distance_jacobian=False
    from mjlab.actuator import IdealPdActuatorCfg
    for i,e in enumerate(cfg.scene.entities.values()):
        e.articulation.actuators=(IdealPdActuatorCfg(target_names_expr=(f'joint{i+1}',),stiffness=0.,damping=1.5,effort_limit=20.,armature=.001),)
    return cfg



def test_task_is_independent_pd_offset_control():
    cfg=_load_env_cfg(TASK)
    assert cfg.jacobian_feedforward
    assert len(cfg.scene.entities)==6
    assert cfg.target_connector==5
    assert set(cfg.actions)=={'joint_torque'}
    assert 'action_rate' not in cfg.rewards
    assert cfg.break_force==80. and cfg.connector_radius==.05
    assert cfg.minimum_other_margin==.2
    assert cfg.separation_distance==.05
    assert cfg.post_separation_wait_s==2.
    assert 'success' not in cfg.terminations
    assert 'separation_wait_done' in cfg.terminations
    assert cfg.enable_weld_failure
    assert 'target_separation' in cfg.rewards
    assert 'success' in cfg.rewards
    assert cfg.torque_action_scale==20.
    assert cfg.torque_limit==20.
    assert cfg.continuous_torque_limit==14.
    assert cfg.torque_overload_duration_s==1.
    assert cfg.loading_reward_cap==1.2
    assert cfg.joint_stiffness==0. and cfg.joint_damping==1.5
    assert cfg.sim.mujoco.jacobian=='dense'
    assert cfg.sim.mujoco.timestep==.005
    assert cfg.decimation==4
    assert cfg.sim.mujoco.timestep*cfg.decimation==.02


def test_transferred_modules_compile_with_one_free_root_and_one_motor_joint():
    snapshot=load_snapshot()
    for i in range(6):
        model=module_spec(i).compile()
        assert model.nq==8 and model.nv==7 and model.njnt==2
        assert abs(model.qpos0[-1]-snapshot['modules'][i]['joint_position'])<1e-8


def test_port_wrench_rotates_root_moment_and_shifts_origin():
    # Root local +X moment becomes world +Y under a 90-degree Z rotation.
    q=torch.tensor([[2**-.5,0.,0.,2**-.5]])
    generalized=torch.tensor([[0.,1.,0.,1.,0.,0.]])
    force,moment=port_wrench(generalized,q,torch.zeros(1,3),
                            torch.tensor([[1.,0.,0.]]),torch.tensor([[1.,0.,0.,0.]]))
    assert torch.allclose(force,torch.tensor([[0.,1.,0.]]),atol=1e-6)
    assert torch.allclose(moment,torch.tensor([[0.,1.,-1.]]),atol=1e-6)


def test_start_is_stationary_zero_gap_ik_without_servo_preload():
    import numpy as np
    from zbot_rl_mjlab.tasks.reconfiguration.planning import ClosurePlanner
    snapshot=load_snapshot()
    assert snapshot['initialization']=='stationary_zero_gap_ik'
    q=np.array([m['joint_position'] for m in snapshot['modules']])
    planner=ClosurePlanner()
    assert np.linalg.norm(planner.error(q,snapshot['tail_registration_quarter'],gap=0.))<1e-8
    planner.data.qpos[:7]=snapshot['modules'][0]['root_state'][:7]
    data=planner.forward(q)
    for module in snapshot['modules']:
        body=planner.model.body(module['root_name']).id
        assert np.allclose(module['root_state'][:3],data.xpos[body],atol=1e-9)
        assert np.allclose(module['root_state'][3:7],data.xquat[body],atol=1e-9)
        assert module['root_state'][7:]==[0.]*6
        assert module['joint_velocity']==0.
        assert module['holding_torque']==0.


def test_zero_offset_pd_loop_keeps_all_ports_rigid_on_gpu(monkeypatch):
    import pytest
    from mjlab.envs import ManagerBasedRlEnv
    from zbot_rl_mjlab.tasks.reconfiguration.disconnect.state import state
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    cfg=load_env_cfg(TASK,play=True)
    cfg.scene.num_envs=2
    cfg.auto_reset=False
    cfg.enable_weld_failure=True
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        with torch.inference_mode():
            env.reset()
            connectors=state(env)
            for _ in range(50):
                _,_,terminated,_,_=env.step(torch.zeros(2,6,device=env.device))
                assert not terminated.any()
                assert connectors.connected().all()
                for i in range(6):
                    geometry=connectors.geometry(i)
                    assert (geometry['distance']<.0001).all()
                    assert (geometry['angle']<torch.pi/180).all()
            metrics=dict(env.metrics_manager.get_active_iterable_terms(0))
            from zbot_rl_mjlab.tasks.reconfiguration.disconnect.state import connector_load
            for i in range(6):
                name=f'connector_{i}_load_N'
                assert name in metrics
                assert abs(metrics[name][0]-connector_load(env,i)[0].item())<1e-5
            action=env.action_manager.get_term('joint_torque')
            requested=torch.ones(2,6,device=env.device)*.5
            action.process_actions(requested)
            assert torch.allclose(action.feedforward,torch.full_like(requested,10.))
            action.process_actions(requested)
            assert torch.allclose(action.feedforward,torch.full_like(requested,10.))
            action.apply_actions()
            from mjlab.actuator import IdealPdActuator
            from mjlab.actuator.actuator import ActuatorCmd
            for module in action.modules:
                actuator=module.actuators[0]
                assert isinstance(actuator,IdealPdActuator)
                assert (actuator.stiffness==0.).all()
                assert (actuator.damping==1.5).all()
                dq=module.data.joint_vel
                cmd=ActuatorCmd(position_target=module.data.joint_pos+100.,
                                velocity_target=torch.zeros_like(dq),
                                effort_target=module.data.joint_effort_target,
                                pos=module.data.joint_pos,vel=dq)
                assert torch.allclose(actuator.compute(cmd),(10.-1.5*dq).clamp(-20.,20.))
                cmd.vel=torch.ones_like(dq)*100
                assert (actuator.compute(cmd)==-20.).all()
            action.reset(torch.tensor([0],device=env.device))
            assert (action.feedforward[0]==0.).all()
            assert (action.feedforward[1]==10.).all()
            from zbot_rl_mjlab.tasks.reconfiguration.disconnect.state import observations
            obs=observations(env)
            assert obs.shape==(2,31)
            assert torch.allclose(obs[:,:6],torch.cat([m.data.joint_pos for m in action.modules],-1)/torch.pi)
            assert torch.allclose(obs[:,6:12],torch.cat([m.data.joint_vel for m in action.modules],-1))
            assert torch.allclose(obs[:,12:18],connectors.utilization)
            assert torch.allclose(obs[:,18:24],action.torque/cfg.continuous_torque_limit)
            assert torch.equal(obs[:,24:30],connectors.connected().float())
            assert torch.allclose(obs[:,30],connectors.geometry()['distance'])
            # Drive the actual passive-break update with known port wrenches:
            # 79N stays connected, exactly 80N opens target only.
            from zbot_rl_mjlab.tasks.reconfiguration.disconnect import state as state_module
            def known_wrench(generalized,*args):
                force=torch.zeros(2,6,3,device=env.device)
                moment=torch.zeros_like(force)
                force[:,:,0]=10000.  # shear is excluded
                force[:,:,2]=-10000.  # compression is excluded
                moment[:,:,2]=10000.  # twist is excluded
                force[:,5,2]=torch.tensor([39.,40.],device=env.device)
                moment[:,5,0]=2.
                return force.flatten(0,1),moment.flatten(0,1)
            monkeypatch.setattr(state_module,'port_wrench',known_wrench)
            env._sim_step_counter+=1
            connectors.update()
            assert connectors.connected()[0].all()
            assert connectors.connected()[1,:5].all()
            assert not connectors.connected()[1,5]
            assert not connectors.wrong_break.any()
            assert torch.allclose(connectors.break_utilization[1,5],torch.tensor(1.,device=env.device))
            broken_obs=observations(env)
            assert broken_obs[1,17]==0.  # no historical rupture load in current observation
            assert broken_obs[1,29]==0.
            assert connector_load(env,5)[1]==0.
            assert torch.allclose(connector_load(env,5)[0],torch.tensor(79.,device=env.device))
            assert torch.allclose(broken_obs[0,17],torch.tensor(79./80.,device=env.device))


    finally:
        env.close()


def test_epa_capacity_is_model_scoped_and_restored(monkeypatch):
    from types import SimpleNamespace
    import pytest
    from mujoco_warp._src import collision_convex,collision_driver
    from zbot_rl_mjlab.tasks.reconfiguration.disconnect.collision import install_epa_adapter
    default=collision_convex.MJ_MAX_EPAHORIZON
    def allocation(model,fail=False):
        capacity=collision_convex.MJ_MAX_EPAHORIZON
        if fail:
            raise RuntimeError('capture failed')
        return capacity
    monkeypatch.setattr(collision_driver,'convex_narrowphase',allocation)
    install_epa_adapter()
    patched=collision_driver.convex_narrowphase
    selected=SimpleNamespace(opt=SimpleNamespace(_zbot_epa_horizon_capacity=64))
    other=SimpleNamespace(opt=SimpleNamespace())
    assert patched(selected)==64
    assert collision_convex.MJ_MAX_EPAHORIZON==default
    assert patched(other)==default
    with pytest.raises(RuntimeError,match='capture failed'):
        patched(selected,fail=True)
    assert collision_convex.MJ_MAX_EPAHORIZON==default


def test_safe_tilt_loading_reward_reaches_1p2_and_ignores_old_break_peak():
    from types import SimpleNamespace
    from zbot_rl_mjlab.tasks.reconfiguration.disconnect.state import target_loading
    cfg=SimpleNamespace(target_connector=5,loading_reward_cap=1.2,minimum_other_margin=.2,break_force=80.)
    loads=torch.zeros(4,6);loads[:,5]=torch.tensor([.6,1.1,1.5,.9])
    s=SimpleNamespace(cfg=cfg,peak_utilization=loads,break_utilization=torch.zeros_like(loads),
                      peak_target_tilt_load=loads[:,5]*80.,
                      wrong_break=torch.tensor([False,False,False,True]),
                      other_margin=lambda:torch.tensor([.2,.1,.3,.5]))
    env=SimpleNamespace(_disconnect_state=s)
    assert torch.allclose(target_loading(env),torch.tensor([.6,.55,1.2,0.]))
    s.break_utilization[:,5]=1.15
    s.peak_utilization.zero_();s.peak_target_tilt_load.zero_()
    assert (target_loading(env)==0.).all()


def test_peel_load_combines_tension_and_tilt_but_excludes_shear_twist_compression():
    from zbot_rl_mjlab.tasks.reconfiguration.disconnect.state import peel_load
    force=torch.tensor([[0.,0.,80.],[0.,0.,40.],[10000.,-10000.,0.],
                        [0.,0.,-1000.],[0.,0.,-1000.],[0.,0.,40.]])
    moment=torch.tensor([[0.,0.,0.],[2.,0.,0.],[0.,0.,10000.],
                         [0.,0.,0.],[3.,4.,9999.],[1.2,1.6,0.]])
    load=peel_load(force,moment,.05)
    assert torch.allclose(load,torch.tensor([80.,80.,0.,0.,100.,80.]))
    assert not (load[[0,1,5]]>80.).any()
    assert torch.equal(load>=80.,torch.tensor([True,True,False,False,True,True]))
    assert torch.allclose(peel_load(force,moment,.1),torch.tensor([80.,60.,0.,0.,50.,60.]))


def test_weld_reaction_tension_is_positive_local_z():
    import mujoco
    import numpy as np
    # Source below receiver; both port frames have +Z pointing into receiver.
    # Pull source DOWN: weld reaction on source must be +Z (tension), while
    # pushing source UP gives a -Z compressive reaction.
    model=mujoco.MjModel.from_xml_string('''<mujoco>
      <option gravity="0 0 0"/>
      <worldbody>
       <body name="source" pos="0 0 0"><freejoint/>
        <geom type="sphere" size=".01" mass="1" contype="0" conaffinity="0"/>
        <site name="out" pos="0 0 .05"/></body>
       <body name="receiver" pos="0 0 .1"><freejoint/>
        <geom type="sphere" size=".01" mass="1" contype="0" conaffinity="0"/>
        <site name="in" pos="0 0 -.05"/></body>
      </worldbody>
      <equality><weld site1="out" site2="in"/></equality></mujoco>''')
    for direction in (-1.,1.):
        data=mujoco.MjData(model)
        data.xfrc_applied[model.body('source').id,2]=direction*100.
        data.xfrc_applied[model.body('receiver').id,2]=-direction*100.
        mujoco.mj_forward(model,data)
        reaction=data.qfrc_constraint[:3]
        assert np.sign(reaction[2])==-direction


def test_all_source_out_ports_have_outward_positive_z():
    import mujoco
    import numpy as np
    from zbot_rl_mjlab.tasks.reconfiguration.disconnect.model import TIP_NAMES
    for i in range(6):
        model=module_spec(i).compile();data=mujoco.MjData(model)
        mujoco.mj_forward(model,data)
        site=model.site('port_out').id
        normal=data.site_xmat[site].reshape(3,3)[:,2]
        # COM of the source tip lies inside the source side of the port plane.
        assert np.dot(data.site_xpos[site]-data.xipos[model.body(TIP_NAMES[i]).id],normal)>0


def test_separation_wait_latches_at_strictly_over_5cm_and_counts_full_two_seconds():
    from types import SimpleNamespace
    from zbot_rl_mjlab.tasks.reconfiguration.disconnect.state import ConnectorState,time_out
    s=ConnectorState.__new__(ConnectorState)
    s.max_target_gap=torch.zeros(2)
    s.cfg=SimpleNamespace(target_connector=5,separation_distance=.05,
                          post_separation_wait_s=2.,minimum_other_margin=.2)
    s.env=SimpleNamespace(physics_dt=.005)
    s.other_ids=list(range(5))
    s.waiting=torch.zeros(2,dtype=torch.bool)
    s.wait_count=torch.zeros(2,dtype=torch.long)
    s.gap_progress=torch.zeros(2)
    s.separation_progress_delta=torch.zeros(2)
    s.succeeded=torch.zeros(2,dtype=torch.bool)
    s.just_succeeded=torch.zeros(2,dtype=torch.bool)
    s.wrong_break=torch.zeros(2,dtype=torch.bool)
    s.safety_margin_violation=torch.zeros(2,dtype=torch.bool)
    gap=torch.tensor([.05,.05001])
    s.geometry=lambda:{'distance':gap}
    s.other_margin=lambda:torch.tensor([.9,.1])  # cutoff is independent of safety
    active=torch.ones(2,6,dtype=torch.bool);active[:,5]=False
    s.update_separation(active)
    assert s.waiting.tolist()==[False,True]
    assert not s.succeeded.any()
    assert s.safety_margin_violation.tolist()==[False,True]
    assert s.wait_count.tolist()==[0,0]
    s.other_margin=lambda:torch.tensor([.9,.2])
    gap.zero_()  # crossing back does not resume torque or restart the timer
    for _ in range(399):
        s.update_separation(active)
    assert not s.wait_done().any()
    s.update_separation(active)
    assert s.wait_done().tolist()==[False,True]
    assert s.safety_margin_violation.tolist()==[False,True]  # recovery cannot erase a violation
    env=SimpleNamespace(_disconnect_state=s,episode_length_buf=torch.tensor([200,200]),
                        max_episode_length=200)
    assert time_out(env).tolist()==[True,False]
    s.safety_margin_violation.zero_()
    s.other_margin=lambda:1.-torch.tensor([.8,.8])
    s.update_separation(active)
    assert not s.safety_margin_violation.any()  # exactly 64N remains allowed
    s.other_margin=lambda:torch.tensor([.199,.199])
    s.update_separation(active)
    assert s.safety_margin_violation.all()  # applies before waiting as well



def test_wait_masks_policy_torque_and_auto_resets_only_after_two_seconds_on_gpu(monkeypatch):
    import pytest
    from mjlab.envs import ManagerBasedRlEnv
    from zbot_rl_mjlab.tasks.reconfiguration.disconnect.state import state
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    cfg=load_env_cfg(TASK)
    # Keep explicit legacy lifecycle coverage while the active task is static.
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from zbot_rl_mjlab.tasks.reconfiguration.disconnect import state as mdp
    cfg.enable_weld_failure=True
    cfg.terminations['separation_wait_done']=TerminationTermCfg(func=mdp.wait_done)
    cfg.rewards['success']=RewardTermCfg(func=mdp.success_reward,weight=10.)
    cfg.metrics['success']=MetricsTermCfg(func=mdp.success,reduce='max')
    cfg.scene.num_envs=2
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        with torch.inference_mode():
            env.reset()
            s=state(env)
            original_geometry=s.geometry
            def geometry(index=None):
                g=original_geometry(index)
                if index is None:
                    g['distance']=torch.tensor([.05,.051],device=env.device)
                return g
            monkeypatch.setattr(s,'geometry',geometry)
            env.sim.data.eq_active[1,s.eq_ids[5]]=False
            s.update_separation(s.connected())
            # Isolate the rollout lifecycle from uncontrolled physical rupture.
            monkeypatch.setattr(s,'update',lambda:s.update_separation(s.connected()))
            action=env.action_manager.get_term('joint_torque')
            # Synthetic geometry/rupture is not a motor-envelope test. Isolate
            # these guards here; actual torque monitoring is injected below.
            monkeypatch.setattr(action,'monitor_torque',lambda:None)
            monkeypatch.setattr(action,'check_torque_demand',lambda:None)
            requests=torch.zeros(2,6,device=env.device);requests[1]=1.
            env.episode_length_buf.fill_(env.max_episode_length-1)
            for i in range(100):
                _,reward,terminated,truncated,_=env.step(requests)
                if i<99:
                    assert not terminated[1] and not truncated[1]
                    assert s.waiting[1]
                    assert reward[1]<1.  # no success bonus during the wait
                    assert (action.feedforward[1]==0.).all()
                    for module in action.modules:
                        assert (module.data.joint_effort_target[1]==0.).all()
                        assert (module.actuators[0].damping[1]==1.5).all()
                else:
                    assert terminated[1] and not truncated[1]
                    assert reward[1]>9.9
                    assert env.extras['log']['Episode_Metrics/success']==1.
                    assert not s.waiting[1] and s.wait_count[1]==0
                    assert s.connected()[1].all()
                if i==0:
                    assert truncated[0] and not s.waiting[0]
            # Waiting ends immediately with the existing -10 failure reward
            # when another face's margin falls below .2, even without rupture.
            s.waiting[1]=True
            s.utilization[1,0]=.801
            _,reward,terminated,truncated,_=env.step(requests)
            assert terminated[1] and not truncated[1]
            assert reward[1]<-9.9
            assert env.extras['log']['Episode_Metrics/success']==0.
            assert not s.waiting[1] and not s.safety_margin_violation[1]
            assert s.wait_count[1]==0 and s.connected()[1].all()
            action.process_actions(requests*.5);action.apply_actions()
            assert (action.feedforward[1]==10.).all()  # control resumes after reset
            assert not s.waiting[1]
            s.utilization[1,0]=.801
            _,reward,terminated,truncated,_=env.step(torch.zeros_like(requests))
            assert terminated[1] and not truncated[1] and reward[1]<-9.9
            assert not s.safety_margin_violation[1]  # auto-reset cleared the latch
            # Controlled actual torque isolates the 1s overload/reset lifecycle.
            measured=torch.zeros(2,6,device=env.device);measured[1,0]=20.
            monkeypatch.setattr(action,'monitor_torque',lambda:action.record_torque(measured))
            for i in range(50):
                _,reward,terminated,truncated,_=env.step(torch.zeros_like(requests))
                if i<49:
                    assert not terminated[1] and not truncated[1]
                    assert action.overload_count[1,0]==4*(i+1)
                else:
                    assert terminated[1] and not truncated[1]
                    assert reward[1]<-9.9
                    assert not action.torque_failure[1]
                    assert (action.overload_count[1]==0).all()
            # Reset only one world: the other world's latch/counter is untouched.
            s.waiting[:]=True;s.wait_count[:]=7
            s.reset(torch.tensor([0],device=env.device))
            assert s.waiting.tolist()==[False,True]
            assert s.wait_count.tolist()==[0,7]
    finally:
        env.close()


def test_safety_penalty_has_no_positive_reward_and_all_costs_are_labelled():
    from types import SimpleNamespace
    from zbot_rl_mjlab.tasks.reconfiguration.disconnect.state import safety_penalty
    margin=torch.tensor([1.,.4,.3,.2,.199,0.])
    s=SimpleNamespace(cfg=SimpleNamespace(minimum_other_margin=.2,safety_penalty_buffer=.2),
                      other_ids=list(range(5)),
                      other_mask=lambda: torch.tensor([True,True,True,True,True,False])[None].expand(len(margin),-1),
                      peak_utilization=(1.-margin)[:,None].expand(-1,6))
    result=safety_penalty(SimpleNamespace(_disconnect_state=s))
    assert torch.allclose(result,torch.tensor([0.,0.,.25,1.,1.010025,4.]),atol=1e-6)
    cfg=load_env_cfg(TASK)
    assert 'other_safety_margin' not in cfg.rewards
    assert cfg.rewards['penalty_safety_margin'].weight==-.2
    assert all(name.startswith('penalty_') for name,term in cfg.rewards.items() if term.weight<0)


def test_continuous_and_peak_torque_envelope_is_per_joint_and_latched():
    from types import SimpleNamespace
    from zbot_rl_mjlab.tasks.reconfiguration.disconnect.env_cfg import TorqueAction,torque_cost
    action=TorqueAction.__new__(TorqueAction)
    action._env=SimpleNamespace(cfg=SimpleNamespace(continuous_torque_limit=14.,
        torque_limit=20.,torque_overload_duration_s=1.),physics_dt=.005)
    action.overload_count=torch.zeros(3,6,dtype=torch.long)
    action.torque_failure=torch.zeros(3,dtype=torch.bool)
    action.peak_overload_cost=torch.zeros(3)
    torque=torch.zeros(3,6)
    torque[0]=14.;torque[1,0]=-20.;torque[2,1]=15.
    for i in range(199):
        if i==100:
            torque[2,1]=14.  # returning to the continuous envelope clears its timer
        action.record_torque(torque)
    assert not action.torque_failure.any()
    assert (action.overload_count[0]==0).all()
    assert action.overload_count[1,0]==199
    assert (action.overload_count[2]==0).all()
    assert action.peak_overload_cost[0]==0.
    assert torch.allclose(action.peak_overload_cost[1],torch.tensor(1/6))
    action.record_torque(torque)
    assert action.torque_failure.tolist()==[False,True,False]
    torque.zero_();torque[0,2]=20.001
    action.record_torque(torque)
    assert action.torque_failure.tolist()==[True,True,False]
    env=SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda _:action))
    assert torch.equal(torque_cost(env),action.peak_overload_cost)


def test_torque_demand_guard_checks_before_native_clipping():
    from types import SimpleNamespace
    from zbot_rl_mjlab.tasks.reconfiguration.disconnect.env_cfg import TorqueAction
    action=TorqueAction.__new__(TorqueAction)
    action._env=SimpleNamespace(cfg=SimpleNamespace(torque_limit=20.))
    action.requested_torque=torch.zeros(3,6)
    action.torque_failure=torch.zeros(3,dtype=torch.bool)
    action.requested_torque[:,0]=torch.tensor([20.,-20.,20.001])
    action.check_torque_demand()
    assert action.torque_failure.tolist()==[False,False,True]
    action.requested_torque.zero_();action.check_torque_demand()
    assert action.torque_failure.tolist()==[False,False,True]


def test_static_loading_keeps_welds_above_capacity_and_preserves_safety_reset_on_gpu(monkeypatch):
    import pytest
    from mjlab.envs import ManagerBasedRlEnv
    from zbot_rl_mjlab.tasks.reconfiguration.disconnect import state as mdp
    if not torch.cuda.is_available():
        pytest.skip('CUDA required')
    cfg=load_env_cfg(TASK);cfg.scene.num_envs=2;cfg.enable_weld_failure=False
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        with torch.inference_mode():
            env.reset()
            for _ in range(10):
                _,_,terminated,truncated,_=env.step(torch.zeros(2,6,device=env.device))
                assert not terminated.any() and not truncated.any()
            s=mdp.state(env)
            with_tilt=[False]
            def known_wrench(generalized,*args):
                f=torch.zeros(2,6,3,device=env.device);m=torch.zeros_like(f)
                f[:,5,2]=120.  # direct tension does not earn target_loading
                if with_tilt[0]:m[:,5,0]=6.
                f[1,0,2]=65.  # unsafe other face: reset without weld removal
                return f.flatten(0,1),m.flatten(0,1)
            monkeypatch.setattr(mdp,'port_wrench',known_wrench)
            s.begin_step()  # isolate synthetic load from the previous physical step peak
            env._sim_step_counter+=1;s.update()
            assert s.connected().all()
            assert not s.waiting.any() and not s.succeeded.any()
            assert not s.break_utilization.any()
            assert s.safety_margin_violation.tolist()==[False,True]
            assert mdp.failure(env).tolist()==[False,True]
            assert mdp.target_loading(env)[0]==0.
            obs=mdp.observations(env)
            assert obs.shape==(2,31)
            assert (obs[:,17]==1.5).all()
            assert (obs[:,24:30]==1.).all()
            with_tilt[0]=True
            env._sim_step_counter+=1;s.update()
            assert torch.allclose(mdp.target_loading(env)[0],torch.tensor(1.2,device=env.device))
            assert (mdp.target_tilt_load(env)==120.).all()
            assert (mdp.target_direct_tension(env)==120.).all()
            assert s.connected().all()
            s.begin_step()
            assert not s.peak_target_tilt_load.any()

            _,reward,terminated,truncated,_=env.step(torch.zeros(2,6,device=env.device))
            assert not terminated[0] and terminated[1] and not truncated[1]
            assert reward[1]<-9.9
            assert s.connected().all()
            assert not s.safety_margin_violation[1]
    finally:
        env.close()
