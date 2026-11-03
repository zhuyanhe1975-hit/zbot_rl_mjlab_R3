import pytest
import torch
from mjlab.envs import ManagerBasedRlEnv
from zbot_rl_mjlab.tasks.reconfiguration.full_batched_env import make_full_batched_env_cfg
from zbot_rl_mjlab.tasks.reconfiguration.full_batched_fsm import P
from zbot_rl_mjlab.tasks.reconfiguration.batched_state import BatchedState

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA graph required')
def test_graph_replays_live_modes_measurements_and_reset():
    with torch.inference_mode():
        env=ManagerBasedRlEnv(make_full_batched_env_cfg(4),device='cuda:0');env.reset()
        try:
            actions=torch.zeros((4,7),device=env.device)
            env.step(actions);env.step(actions)
            s=env._disconnect_state;a=env.action_manager.get_term('joint_torque')
            assert a.actuation_graph is not None and s.measurement_graph is not None
            # Measurement graph reads current world/target values rather than
            # the values present when the graph was originally captured.
            s.fsm.target[:]=torch.tensor([0,1,4,5],device=env.device)
            s.measurement_graph.replay();torch.cuda.synchronize()
            expected={k:getattr(s,k).clone() for k in ('force','moment','utilization','max_target_gap')}
            s.last_substep=None;BatchedState.update(s);torch.cuda.synchronize()
            for k,v in expected.items():torch.testing.assert_close(getattr(s,k),v,atol=2e-5,rtol=2e-5)
            # Compare eager and graph execution from identical mutable action
            # buffers, including different modes in the same batch.
            s.fsm.target.fill_(0);env.sim.data.eq_active[:,s.eq_ids[5]]=True
            s.fsm.phase[:]=torch.tensor([P['CURL'],P['LOAD'],P['LOAD_HEADTAIL'],P['FINAL_STRAIGHT_HOLD']],device=env.device)
            s.fsm.target[2]=5;a.policy[:,:6]=.2;a.loading.torque.fill_(.5);a.motion_velocity.fill_(.04);a.torque_failure.zero_()
            baseline=a.position_target.clone();a._apply_actions_eager();torch.cuda.synchronize()
            gain=env.sim.model.actuator_gainprm._tensor[:,a.native_ids,0].clone();bias=env.sim.model.actuator_biasprm._tensor[:,a.native_ids,:3].clone()
            requested=a.requested_torque.clone();position=a.position_target.clone();ctrl=torch.cat([m.data.joint_pos_target for m in a.modules],-1).clone()
            a.position_target.copy_(baseline);a.torque_failure.zero_();a.actuation_graph.replay();torch.cuda.synchronize()
            torch.testing.assert_close(env.sim.model.actuator_gainprm._tensor[:,a.native_ids,0],gain)
            torch.testing.assert_close(env.sim.model.actuator_biasprm._tensor[:,a.native_ids,:3],bias)
            torch.testing.assert_close(a.requested_torque,requested,atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(a.position_target,position,atol=2e-6,rtol=2e-6)
            torch.testing.assert_close(torch.cat([m.data.joint_pos_target for m in a.modules],-1),ctrl,atol=2e-6,rtol=2e-6)
            env.reset(env_ids=torch.tensor([1,3],device=env.device));env.step(actions)
            assert not s.connected()[[1,3],5].any()
        finally:env.close()

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA graph required')
def test_graph_target_break_immediately_removes_loading_torque():
    from mjlab.utils.lab_api.math import quat_apply
    with torch.inference_mode():
        env=ManagerBasedRlEnv(make_full_batched_env_cfg(1),device='cuda:0');env.reset()
        try:
            action=torch.zeros((1,7),device=env.device);env.step(action);env.step(action)
            s=env._disconnect_state;a=env.action_manager.get_term('joint_torque');d=env.sim.data
            s.fsm.phase.fill_(P['LOAD']);s.fsm.target.zero_();d.eq_active[:,s.eq_ids[5]]=True
            env.sim.forward();eq=s.eq_ids[0]
            efc=d.efc;rows=torch.nonzero((efc.type[0]==0)&(efc.id[0]==eq)).flatten()[:6]
            assert rows.numel()==6
            J=efc.J[0,rows][:,s.root_dofs[0]]
            port,frame=s.outgoing_port_frames();root=s.modules[0].data.root_link_pos_w
            rootq=s.modules[0].data.root_link_quat_w
            force=quat_apply(frame[:,0],torch.tensor([[0.,0.,85.]],device=env.device))
            arm=port[:,0]-root;world_moment=torch.cross(arm,force,dim=-1)
            from mjlab.utils.lab_api.math import quat_apply_inverse
            gen=torch.cat((force,quat_apply_inverse(rootq,world_moment)),-1)[0]
            d.efc.force[0].zero_();d.efc.force[0,rows]=torch.linalg.solve(J.T,gen)
            s.measurement_graph.replay();torch.cuda.synchronize()
            assert not s.connected()[0,0] and not s.wrong_break[0]
            a.loading.torque.fill_(5.);a.policy[:,:6]=1.;a.actuation_graph.replay();torch.cuda.synchronize()
            assert not a.policy_torque.any()
            assert (env.sim.model.actuator_gainprm._tensor[0,a.native_ids,0]>1.).all()
        finally:env.close()

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_async_loading_solver_matches_existing_solver():
    with torch.inference_mode():
        env=ManagerBasedRlEnv(make_full_batched_env_cfg(4),device='cuda:0');env.reset()
        try:
            s=env._disconnect_state;a=env.action_manager.get_term('joint_torque')
            env.sim.data.eq_active[:,s.eq_ids[5]]=True;env.sim.forward()
            ids=torch.arange(4,device=env.device)
            a.loading.async_linalg=False;A,w,p=a.loading.response(s,ids)
            a.loading.async_linalg=True;B,v,q=a.loading.response(s,ids)
            assert p.all() and q.all()
            torch.testing.assert_close(B,A,atol=1e-9,rtol=1e-8)
            torch.testing.assert_close(v,w,atol=1e-9,rtol=1e-8)
        finally:env.close()

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_shared_warp_stream_matches_default_stream_physics():
    import warp as wp
    with torch.inference_mode():
        first=ManagerBasedRlEnv(make_full_batched_env_cfg(2),device='cuda:0')
        second=ManagerBasedRlEnv(make_full_batched_env_cfg(2),device='cuda:0')
        previous=torch.cuda.current_stream(first.device)
        shared=torch.cuda.ExternalStream(wp.get_stream(first.sim.wp_device).cuda_stream,device=first.device)
        try:
            first.reset();second.reset();action=torch.zeros((2,7),device=first.device)
            for _ in range(6):
                torch.cuda.set_stream(previous);first.step(action)
                torch.cuda.set_stream(shared);second.step(action)
            torch.cuda.synchronize()
            torch.testing.assert_close(first.sim.data.qpos._tensor,second.sim.data.qpos._tensor,atol=2e-5,rtol=2e-5)
            torch.testing.assert_close(first.sim.data.qvel._tensor,second.sim.data.qvel._tensor,atol=2e-4,rtol=2e-4)
            assert torch.equal(first.sim.data.eq_active._tensor,second.sim.data.eq_active._tensor)
        finally:
            torch.cuda.set_stream(previous);first.close();second.close()

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA graph required')
def test_welded_face_contacts_are_filtered_in_simulation_graph():
    """The graph captured by MJLab must include the installed contact filter."""
    with torch.inference_mode():
        cfg=make_full_batched_env_cfg(4)
        cfg.auto_reset=False
        env=ManagerBasedRlEnv(cfg,device='cuda:0')
        try:
            env.reset()
            env.step(torch.zeros((4,7),device=env.device))
            d=env.sim.data;c=d.contact;n=int(d.nacon[0])
            pair=env.sim.mj_model.pair('open_face_2').id
            a=int(env.sim.mj_model.pair_geom1[pair]);b=int(env.sim.mj_model.pair_geom2[pair])
            geom=c.geom[:n];kind=c.type[:n].int();world=c.worldid[:n].long()
            match=((geom[:,0]==a)&(geom[:,1]==b))|((geom[:,0]==b)&(geom[:,1]==a))
            assert match.any(), 'test needs actual narrowphase contacts for this welded face'
            assert env._disconnect_state.connected()[world[match],2].all()
            assert not ((kind[match]&1)!=0).any()
        finally:
            env.close()

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA graph required')
def test_margin_guard_stops_actuators_on_next_physics_substep():
    with torch.inference_mode():
        cfg=make_full_batched_env_cfg(2)
        cfg.full_enforce_other_margin=True
        env=ManagerBasedRlEnv(cfg,device='cuda:0')
        try:
            env.reset();env.step(torch.zeros((2,7),device=env.device))
            a=env.action_manager.get_term('joint_torque');s=env._disconnect_state
            assert a.actuation_graph is not None
            s.safety_margin_violation[0]=True
            a.actuation_graph.replay();torch.cuda.synchronize()
            gains=env.sim.model.actuator_gainprm._tensor[:,a.native_ids,0]
            assert (gains[0]==0).all()
            assert (gains[1]>0).all()
        finally:env.close()

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA graph required')
def test_terminal_success_rechecks_post_step_speed():
    from zbot_rl_mjlab.tasks.reconfiguration.full_batched_env import success
    with torch.inference_mode():
        cfg=make_full_batched_env_cfg(1)
        cfg.auto_reset=False;cfg.full_terminal_recheck=True;cfg.full_final_angle_tolerance=.04
        env=ManagerBasedRlEnv(cfg,device='cuda:0')
        try:
            env.reset();s=env._disconnect_state;a=env.action_manager.get_term('joint_torque')
            s.fsm.phase.fill_(P['DONE'])
            env.sim.data.qvel[:,a.joint_dof_ids]=.2
            assert not success(env).item()
            env.step(torch.zeros((1,7),device=env.device))
            assert s.fsm.phase.item()!=P['DONE']
        finally:env.close()
