"""Unified residual control: real constraint linearization and trainable loading."""
from dataclasses import asdict
import numpy as np
import pytest
import torch
import mujoco


def test_gpu_unified_loading_residual_mask_and_checkpoint(tmp_path):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    import zbot_rl_mjlab
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
    from zbot_rl_mjlab.tasks.reconfiguration.disconnect.state import state,observations,critic_observations
    from zbot_rl_mjlab.tasks.reconfiguration.disconnect.jacobian import constraint_response,reference_wrenches
    T='Mjlab-Zbot-6dof-Reconfigure-Disconnect-Torque'
    cfg=load_env_cfg(T);cfg.scene.num_envs=4
    core=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        # Analytic constraint-Jacobian response vs actual MuJoCo forward differences.
        m=core.sim.mj_model;d=mujoco.MjData(m);d.qpos[:]=m.qpos0
        A,w0,_=constraint_response(m);fd=[]
        for k in range(6):
            jid=m.joint(f'module_{k}/joint{k+1}').id
            aid=int(np.where(m.actuator_trnid[:,0]==jid)[0][0])
            d.ctrl[:]=0.;d.ctrl[aid]=1e-4;mujoco.mj_forward(m,d);plus=reference_wrenches(m,d)
            d.ctrl[aid]=-1e-4;mujoco.mj_forward(m,d);minus=reference_wrenches(m,d)
            fd.append((plus-minus)/2e-4)
        assert np.allclose(A,np.stack(fd,-1),atol=1e-6,rtol=1e-5)
        env=RslRlVecEnvWrapper(core,clip_actions=1.)
        agent=asdict(load_rl_cfg(T));agent['num_steps_per_env']=4
        agent['algorithm'].update(num_learning_epochs=1,num_mini_batches=2)
        assert 'teacher_checkpoint' not in agent
        runner=load_runner_cls(T)(env,agent,device='cuda:0');alg=runner.alg
        ctrl=core.action_manager.get_term('joint_torque')
        assert ctrl.jacobian_report['method']=='constraint_J_M_R_active_set'
        assert max(ctrl.jacobian_report['predicted_loads_N'][:5])<=60.+1e-5
        with torch.inference_mode():
            obs,_=env.reset()
            assert obs['actor'].shape==(4,43) and obs['critic'].shape==(4,45)
            assert torch.allclose(obs['actor'][:,31:37]*cfg.torque_limit,ctrl.jacobian_command())
            for t in range(4):
                if t==2:
                    state(core).waiting.fill_(True);obs=env.get_observations().clone()
                    obs['actor']=observations(core);obs['critic']=critic_observations(core)
                action=alg.act(obs)
                if t==0:
                    assert alg.pending_mask.all()  # includes the closed-loop loading step
                    assert torch.equal(action,alg.transition.actions)
                else:assert not alg.pending_mask.any() and (action==0).all()
                obs,reward,done,extras=env.step(action)
                alg.process_env_step(obs,reward,done,extras)
                if t==0:
                    assert state(core).connected()[:,:5].all()
                    # Mask contract uses a controlled pre-action phase; the full
                    # physical transition is checked separately below.
                    core.sim.data.eq_active[:,state(core).eq_ids[5]]=False
                    obs['actor']=observations(core);obs['critic']=critic_observations(core)
                    assert not ctrl.jacobian_command().any()
            assert alg.policy_masks[:,0].tolist()==[True,False,False,False]
            alg.compute_returns(obs)
        loss=alg.update();assert loss['student_fraction']==.25
        assert loss['actor_batches']>0
        assert 'loading_teacher_action' not in alg.save()
        path=tmp_path/'model.pt';runner.save(path)
        restored=load_runner_cls(T)(env,asdict(load_rl_cfg(T)),device='cuda:0')
        restored.load(path,load_cfg={'actor':True},map_location='cuda:0')
        policy=restored.get_inference_policy()
        with torch.inference_mode():
            reset_obs,_=env.reset()
            assert torch.equal(policy(reset_obs),restored.alg.actor(reset_obs))
            state(core).waiting.fill_(True);wait_obs=env.get_observations().clone()
            wait_obs['actor']=observations(core);wait_obs['critic']=critic_observations(core)
            assert (policy(wait_obs)==0.).all() and not ctrl.jacobian_command().any()
        with pytest.raises(ValueError):
            restored.load('outputs/reconfiguration/disconnect_torque/static_gaussian/penalty02/verification_model.pt')
    finally:core.close()


def test_gpu_velocity_pd_switch_native_law_and_protection():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    import zbot_rl_mjlab
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.actuator.actuator import ActuatorCmd
    from zbot_rl_mjlab.tasks.reconfiguration.disconnect.state import state,observations
    cfg=load_env_cfg('Mjlab-Zbot-6dof-Reconfigure-Disconnect-Torque')
    cfg.scene.num_envs=2;cfg.auto_reset=False;cfg.separation_distance_jacobian=False
    from mjlab.actuator import IdealPdActuatorCfg
    for i,e in enumerate(cfg.scene.entities.values()):
        e.articulation.actuators=(IdealPdActuatorCfg(target_names_expr=(f'joint{i+1}',),stiffness=0.,damping=1.5,effort_limit=20.,armature=.001),)
    core=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        with torch.inference_mode():
            core.reset();s=state(core);ctrl=core.action_manager.get_term('joint_torque')
            q=torch.cat([m.data.joint_pos for m in ctrl.modules],-1).clone()
            ctrl.process_actions(torch.full_like(q,.2))
            core.sim.data.eq_active[1,s.eq_ids[5]]=False
            ctrl.apply_actions()
            assert ctrl.pd_active.tolist()==[False,True]
            assert torch.equal(ctrl.position_target[1],q[1])
            assert not ctrl.velocity_command[1].any()  # no loading-action reinterpretation
            assert not ctrl.feedforward[1].any()
            assert ctrl.feedforward[0].any()
            assert ctrl.modules[0].actuators[0].stiffness[:,0].tolist()==[0.,15.]
            ctrl.process_actions(torch.full_like(q,.5))
            for _ in range(cfg.decimation):ctrl.apply_actions()
            delta=.5*cfg.separation_max_target_speed*core.step_dt
            assert torch.allclose(ctrl.position_target[1]-q[1],torch.full_like(q[1],delta),atol=2e-7)
            assert torch.allclose(observations(core)[1,-6:]*torch.pi,ctrl.position_target[1]-q[1])
            actual=[]
            for module in ctrl.modules:
                a=module.actuators[0]
                cmd=ActuatorCmd(position_target=module.data.joint_pos_target,
                    velocity_target=module.data.joint_vel_target,effort_target=module.data.joint_effort_target,
                    pos=module.data.joint_pos,vel=module.data.joint_vel)
                actual.append(a.compute(cmd))
            assert torch.allclose(torch.cat(actual,-1),ctrl.requested_torque,atol=1e-6)
            # Target integration is bounded by joint limits and over-demand cuts P.
            ctrl.position_target[1]+=2.
            ctrl.apply_actions()
            assert ctrl.torque_failure[1] and not ctrl.feedforward[1].any()
            assert ctrl.modules[0].actuators[0].stiffness[1]==0.
            ctrl.reset(torch.tensor([1],device=core.device))
            assert not ctrl.pd_active[1] and not ctrl.velocity_command[1].any()
            assert not ctrl.torque_failure[1]
            # Forced wait must remove position hold even if target error exists.
            ctrl.pd_active[1]=True;ctrl.position_target[1]=q[1]+.2;s.waiting[1]=True
            ctrl.apply_actions()
            assert not ctrl.feedforward[1].any()
            assert ctrl.modules[0].actuators[0].stiffness[1]==0.
            assert torch.allclose(ctrl.requested_torque[1],-cfg.joint_damping*torch.cat([m.data.joint_vel for m in ctrl.modules],-1)[1])
    finally:core.close()



def test_gpu_distance_jacobian_native_pd_full_separation():
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    import zbot_rl_mjlab
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.actuator.builtin_actuator import BuiltinPositionActuator
    from zbot_rl_mjlab.tasks.reconfiguration.disconnect.state import state,failure,success
    from zbot_rl_mjlab.tasks.reconfiguration.planning import ClosurePlanner
    cfg=load_env_cfg('Mjlab-Zbot-6dof-Reconfigure-Disconnect-Torque')
    cfg.scene.num_envs=2;cfg.auto_reset=False
    core=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        with torch.inference_mode():
            core.reset();s=state(core);ctrl=core.action_manager.get_term('joint_torque')
            assert all(isinstance(m.actuators[0],BuiltinPositionActuator) for m in ctrl.modules)
            # Independent finite differences on the original serial kinematic model.
            q=torch.cat([m.data.joint_pos for m in ctrl.modules],-1)[0].cpu().numpy().astype(float)
            planner=ClosurePlanner();d=planner.forward(q)
            normal=d.site_xmat[planner.tail].reshape(3,3)[:,2].copy()
            fd=[]
            for i in range(6):
                plus=q.copy();minus=q.copy();plus[i]+=1e-5;minus[i]-=1e-5
                dp=planner.forward(plus);p=dp.site_xpos[planner.tail].copy()-dp.site_xpos[planner.head]
                dm=planner.forward(minus);m=dm.site_xpos[planner.tail].copy()-dm.site_xpos[planner.head]
                fd.append(-normal@(p-m)/2e-5)
            row,_=ctrl.distance_controller.jacobian(s)
            assert np.allclose(row[0].cpu(),fd,atol=1e-5)
            peak=torch.zeros(2,device=core.device);completed=torch.zeros(2,dtype=torch.bool,device=core.device)
            for _ in range(300):
                _,_,terminated,timed_out,_=core.step(torch.zeros(2,6,device=core.device))
                assert not failure(core).any()
                assert s.connected()[:,:5].all()
                peak=torch.maximum(peak,s.max_target_gap)
                completed |= success(core)
                assert (ctrl.velocity_command.abs()<=cfg.separation_max_target_speed+1e-6).all()
                if s.waiting.any():
                    assert not ctrl.velocity_command[s.waiting].any()
                    for aid in ctrl.native_ids:
                        assert not core.sim.model.actuator_biasprm[s.waiting,aid,1].any()
                if completed.all():break
                ended=(terminated|timed_out).nonzero(as_tuple=True)[0]
                if ended.numel():core.reset(env_ids=ended)
            assert completed.all() and (peak>.05).all()
            ctrl.reset(torch.tensor([1],device=core.device))
            assert not ctrl.distance_controller.command[1].any()
            assert not ctrl.pd_active[1]
    finally:core.close()
