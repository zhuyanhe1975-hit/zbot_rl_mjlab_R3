"""PPO actor masking, teacher routing, and future credit through the forced wait."""
from dataclasses import asdict
from types import SimpleNamespace
import pytest
import torch
from tensordict import TensorDict
from zbot_rl_mjlab.teacher_separation import TeacherMaskedPPO,TeacherStudentPolicy,TeacherSeparationRunner


def test_wait_terminal_reward_reaches_student_through_gae():
    class Critic(torch.nn.Module):
        def forward(self,obs):return torch.zeros(1,1)
        def get_hidden_state(self):return None
        def reset(self,**kwargs):pass
    alg=TeacherMaskedPPO.__new__(TeacherMaskedPPO)
    alg.gamma=.995;alg.lam=.95;alg.normalize_advantage_per_mini_batch=False;alg.critic=Critic()
    alg.storage=SimpleNamespace(num_transitions_per_env=4,values=torch.zeros(4,1,1),
        rewards=torch.tensor([0.,0.,0.,10.]).reshape(4,1,1),dones=torch.tensor([0,0,0,1]).reshape(4,1,1),
        returns=torch.zeros(4,1,1),advantages=torch.zeros(4,1,1))
    alg.policy_masks=torch.tensor([[True],[False],[False],[False]])
    alg.compute_returns(TensorDict({'critic':torch.zeros(1,33)},batch_size=[1]))
    assert torch.allclose(alg.storage.returns[0,0,0],torch.tensor(10*(.995*.95)**3))
    assert not alg.storage.advantages[1:].any()


def test_inference_routes_teacher_student_and_forced_wait():
    class Actor(torch.nn.Module):
        def forward(self,obs,**kwargs):return torch.ones(3,6)*.01
        def reset(self,*args,**kwargs):pass
    policy=TeacherStudentPolicy(Actor(),torch.ones(6)*.2,5)
    obs={'actor':torch.zeros(3,31),'critic':torch.zeros(3,33)}
    obs['actor'][0,29]=1.;obs['critic'][2,31]=1.
    action=policy(obs)
    assert (action[0]==.2).all() and (action[1]==.01).all() and (action[2]==0).all()


def test_gpu_policy_masks_block_all_actor_updates_and_checkpoint_is_self_contained(tmp_path):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    import zbot_rl_mjlab
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
    from zbot_rl_mjlab.tasks.reconfiguration.disconnect.state import state
    task='Mjlab-Zbot-6dof-Reconfigure-Disconnect-Torque'
    cfg=load_env_cfg(task);cfg.scene.num_envs=8;cfg.jacobian_feedforward=False;cfg.separation_velocity_pd=False;cfg.separation_distance_jacobian=False
    from mjlab.actuator import IdealPdActuatorCfg
    for i,e in enumerate(cfg.scene.entities.values()):
        e.articulation.actuators=(IdealPdActuatorCfg(target_names_expr=(f'joint{i+1}',),stiffness=0.,damping=1.5,effort_limit=20.,armature=.001),)
    core=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        env=RslRlVecEnvWrapper(core,clip_actions=1.)
        agent=asdict(load_rl_cfg(task));agent['num_steps_per_env']=4
        agent['algorithm']['class_name']='zbot_rl_mjlab.teacher_separation:TeacherMaskedPPO'
        agent['teacher_checkpoint']='outputs/reconfiguration/disconnect_torque/static_gaussian/penalty02/verification_model.pt'
        agent['algorithm'].update(num_learning_epochs=2,num_mini_batches=2)
        runner=TeacherSeparationRunner(env,agent,device='cuda:0');runner.initialize_teacher()
        alg=runner.alg;teacher=alg.teacher_action.clone()
        with torch.inference_mode():
            obs,_=env.reset()
            assert obs['actor'].shape==(8,31) and obs['critic'].shape==(8,33)
            for t in range(4):
                if t==2:
                    state(core).waiting.fill_(True)
                    obs=env.get_observations()
                actions=alg.act(obs)
                if t==0:
                    assert torch.equal(actions,teacher.expand(8,-1))
                    assert not alg.pending_mask.any()
                elif t==1:
                    assert alg.pending_mask.all()
                    assert torch.equal(actions,alg.transition.actions)
                else:
                    assert not alg.pending_mask.any() and (actions==0).all()
                obs,reward,done,extras=env.step(actions)
                alg.process_env_step(obs,reward,done,extras)
            masks=alg.policy_masks.clone()
            assert masks[:,0].tolist()==[False,True,False,False]
            alg.compute_returns(obs)
        # With NO actor-controlled transitions, even entropy/Adam/normalizers
        # must leave the actor exactly unchanged while the critic can learn.
        def actor_state():return {k:v.clone() for k,v in alg.actor.state_dict().items()}
        before=actor_state();alg.policy_masks.zero_()
        loss=alg.update()
        assert loss['actor_batches']==0. and loss['entropy']==0.
        assert all(torch.equal(v,before[k]) for k,v in alg.actor.state_dict().items())
        # Valid student transitions update the policy, then another masked batch
        # must remain frozen despite nonzero Adam momentum from that update.
        alg.policy_masks.copy_(masks);alg.storage.step=4
        with torch.inference_mode():alg.compute_returns(obs)
        active_loss=alg.update()
        assert active_loss['actor_batches']>0 and active_loss['student_fraction']==.25
        assert any(not torch.equal(v,before[k]) for k,v in alg.actor.state_dict().items())
        before=actor_state();alg.policy_masks.zero_();alg.storage.step=4
        with torch.inference_mode():alg.compute_returns(obs)
        alg.update()
        assert all(torch.equal(v,before[k]) for k,v in alg.actor.state_dict().items())
        assert torch.equal(alg.teacher_action,teacher)
        path=tmp_path/'new.pt';runner.save(path)
        restored_agent=asdict(load_rl_cfg(task));restored_agent['algorithm']['class_name']='zbot_rl_mjlab.teacher_separation:TeacherMaskedPPO';restored_agent['teacher_checkpoint']=str(tmp_path/'missing.pt')
        restored=TeacherSeparationRunner(env,restored_agent,device='cuda:0')
        restored.load(path,load_cfg={'actor':True},map_location='cuda:0')
        assert torch.equal(restored.alg.teacher_action,teacher)
        policy=restored.get_inference_policy()
        with torch.inference_mode():
            reset_obs,_=env.reset()
            assert torch.equal(policy(reset_obs),teacher.expand(8,-1))
        with pytest.raises(ValueError):
            restored.load('outputs/reconfiguration/disconnect_torque/static_gaussian/penalty02/verification_model.pt')
    finally:core.close()
