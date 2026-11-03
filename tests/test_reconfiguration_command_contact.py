"""Requested registration, goal-scoped bonds, and premature end pressure."""
from types import SimpleNamespace

import pytest
import torch
from mjlab.tasks.registry import load_env_cfg

from zbot_rl_mjlab.tasks.reconfiguration import contact,mdp
from zbot_rl_mjlab.tasks.reconfiguration.command import DockingCommand,DockingCommandCfg

TASK='Mjlab-Zbot-6dof-Reconfigure-Head-Tail-Docking-PBT'


def test_goal_is_sampled_per_episode_and_manual_input_is_scoped():
    env=SimpleNamespace(num_envs=1000,device='cpu',cfg=SimpleNamespace(allowed_dock_quarters=(0,1,2,3)))
    term=DockingCommand(DockingCommandCfg(),env)
    ids=torch.arange(1,1000)
    term._resample_command(ids)
    assert term.quarter[0] == 2 and set(term.quarter[ids].tolist()) == {0,1,2,3}
    before=term.quarter.clone()
    term.set_target(270,torch.tensor([0]))
    assert term.quarter[0] == 3
    torch.testing.assert_close(term.quarter[1:],before[1:])
    assert term.command.shape == (1000,4)
    term._resample_command(torch.tensor([0]))
    assert term.quarter[0] == 3  # user input persists across reset
    with pytest.raises(ValueError):term.set_target(45)
    with pytest.raises(ValueError):DockingCommand(DockingCommandCfg(angles_deg=(0,90),target_yaw_deg=180),env)


def test_collision_cost_requires_premature_pressure_and_exempts_valid_pose(monkeypatch):
    geometry={'distance':torch.tensor([.02,.02,.001,.001,.02]),
              'angle':torch.tensor([.5,.5,0.,0.,.5]),'speed':torch.zeros(5),
              'angular_speed':torch.zeros(5),'quarter':torch.full((5,),2)}
    bonds=torch.zeros(5,4,dtype=torch.bool);bonds[3,2]=True
    cfg=SimpleNamespace(capture_distance=.004,capture_angle=.14,capture_speed=.1,capture_angular_speed=1.)
    dock=SimpleNamespace(geometry=lambda:geometry,cfg=cfg,enabled=lambda:bonds)
    monkeypatch.setattr(contact.mdp,'state',lambda env:dock)
    monkeypatch.setattr(contact,'end_contact_force',lambda env:torch.tensor([0.,100.,100.,100.,1.]))
    cost=contact.premature_contact_cost(None)
    assert cost[0] == cost[2] == cost[3] == cost[4] == 0
    assert cost[1] > 2.
    # Being bonded in a different requested registration does not count as success.
    bonds[1,0]=True
    assert contact.premature_contact_cost(None)[1] > 0
    assert mdp.connected_reward(None).tolist() == [0.,0.,0.,1.,0.]


def test_contact_history_counts_peak_normal_force_not_friction():
    history=torch.zeros(2,4,1,3);history[0,1,0,0]=50.;history[1,2,0,1]=200.
    env=SimpleNamespace(num_envs=2,scene={'end_to_end_contact':SimpleNamespace(data=SimpleNamespace(force_history=history))})
    torch.testing.assert_close(contact.end_contact_force(env),torch.tensor([50.,0.]))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA real sensor')
def test_real_end_body_collision_is_measured_and_goal_change_preserves_bond():
    from mjlab.envs import ManagerBasedRlEnv
    cfg=load_env_cfg(TASK,play=True);cfg.scene.num_envs=2
    cfg.docking_mode='registered_weld'
    with torch.inference_mode():
        env=ManagerBasedRlEnv(cfg,device='cuda:0')
        try:
            env.reset();robot=env.scene['robot']
            q=torch.tensor([-1.7788335,1.6934241,-1.5820078,1.4501922,-1.5534124,1.7137595],device=env.device).expand(2,-1)
            root=robot.data.default_root_state.clone();root[:,:3]=env.scene.env_origins+root.new_tensor((0.,0.,1.));root[:,7:]=0
            robot.write_root_state_to_sim(root);robot.write_joint_state_to_sim(q,torch.zeros_like(q))
            target=q.new_tensor([-2.019,2.302,-2.772,2.674,-2.299,1.825]).expand_as(q)
            robot.set_joint_position_target(target);env.scene.write_data_to_sim()
            env.sim.forward();env.scene.update(env.physics_dt)
            for _ in range(3):env.sim.step();env.scene.update(env.physics_dt)
            assert (contact.end_contact_force(env) > 2.).all()
            assert (contact.premature_contact_cost(env) > 0).all()
            dock=mdp.state(env);dock.env.sim.data.eq_active[:,dock.eq_ids[2]]=True
            before=env.sim.data.qpos[:].clone()
            env.command_manager.get_term('dock').set_target(90,torch.tensor([0],device=env.device))
            assert dock.enabled().sum(-1).tolist() == [1,1]
            torch.testing.assert_close(env.sim.data.qpos[:],before)
            assert mdp.connected_reward(env).tolist() == [0.,1.]
            assert dock.geometry()['quarter'].tolist() == [1,2]
        finally:env.close()


def test_requested_positive_quarter_turns_use_the_correct_site_frame_sign():
    import mujoco
    import numpy as np
    import math
    from zbot_rl_mjlab.tasks.reconfiguration.model import head_tail_spec
    spec=head_tail_spec()
    sole=next(site for site in spec.sites if site.name=='foot_1_sole')
    inverse=np.array(sole.quat);inverse[1:]*=-1
    for quarter in range(4):
        dock=next(site for site in spec.sites if site.name==f'tail_dock_{quarter}')
        offset=np.empty(4);mujoco.mju_mulQuat(offset,inverse,dock.quat)
        desired=np.array([math.cos(quarter*math.pi/4),0.,0.,math.sin(quarter*math.pi/4)])
        matched=np.empty(4);mujoco.mju_mulQuat(matched,desired,offset)
        assert abs(matched[0]) > 1-1e-12
        np.testing.assert_allclose(matched[1:],0.,atol=1e-12)


def test_compiled_straight_chain_head_axis_is_inward_and_tail_axis_outward():
    import mujoco
    import numpy as np
    from zbot_rl_mjlab.tasks.reconfiguration.model import head_tail_spec
    model = head_tail_spec().compile()
    data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    h = model.site('foot_0_sole').id; t = model.site('foot_1_sole').id
    zh = data.site_xmat[h].reshape(3,3)[:,2]
    zt = data.site_xmat[t].reshape(3,3)[:,2]
    separation = data.site_xpos[t] - data.site_xpos[h]
    assert np.dot(zh,zt) > .99999
    assert np.dot(zh,separation) > 0 and np.dot(zt,separation) > 0


def test_normal_pose_geometry_ignores_arbitrary_twist_and_twist_velocity(monkeypatch):
    from zbot_rl_mjlab.tasks.reconfiguration.magnet import MagneticDock
    dock=object.__new__(MagneticDock)
    angles=torch.tensor([0.,.4,1.7,3.0]);q=torch.zeros(4,4)
    q[:,0]=(angles/2).cos();q[:,3]=(angles/2).sin()
    identity=torch.tensor([1.,0.,0.,0.]).expand(4,-1)
    dock.cfg=SimpleNamespace(docking_mode='normal_pose')
    dock.env=SimpleNamespace(num_envs=4,device='cpu')
    dock.sites=[0,1];dock.dock_sites=torch.tensor([2,3,4,5]);dock.allowed=torch.ones(4,dtype=torch.bool)
    monkeypatch.setattr('zbot_rl_mjlab.tasks.reconfiguration.magnet.target_quarter',lambda env:torch.tensor([0,1,2,3]))
    quats=torch.cat((identity[:,None],q[:,None],identity[:,None].expand(-1,4,-1)),dim=1)
    velocity=torch.zeros(4,6,3);velocity[:,1,2]=10. # free spin does not tilt normals
    dock.robot=SimpleNamespace(data=SimpleNamespace(site_quat_w=quats,site_pos_w=torch.zeros(4,6,3),
        site_lin_vel_w=torch.zeros(4,6,3),site_ang_vel_w=velocity))
    g=dock.geometry()
    torch.testing.assert_close(g['angle'],torch.zeros(4))
    torch.testing.assert_close(g['angular_speed'],torch.zeros(4))
    assert not g['quarter'].any()


def test_unused_twist_command_does_not_reset_normal_pose_success_or_dwell():
    env=SimpleNamespace(num_envs=2,device='cpu',cfg=SimpleNamespace(
        allowed_dock_quarters=(0,1,2,3),docking_mode='normal_pose'))
    dock=SimpleNamespace(capture_time=torch.tensor([.5,.2]),hold_time=torch.tensor([.5,.2]),
        succeeded=torch.tensor([True,False]),just_succeeded=torch.tensor([True,False]))
    env._magnetic_dock=dock
    before={name:value.clone() for name,value in vars(dock).items()}
    DockingCommand(DockingCommandCfg(),env).set_target(90)
    for name,value in before.items():torch.testing.assert_close(getattr(dock,name),value)
