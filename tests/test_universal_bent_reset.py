"""弯曲初态仅扩展部署训练，保留双足及规范蛇形并缓存碰撞包络。"""
from types import SimpleNamespace

import pytest
import torch

from zbot_rl_mjlab.tasks.universal import deployment
from zbot_rl_mjlab.tasks.universal.env_cfg import universal_env_cfg


def fixture(monkeypatch, *, fraction=1., training=True):
    class Scene(dict):
        pass

    n=8
    data=SimpleNamespace(joint_pos=torch.zeros(n,6),default_root_state=torch.zeros(n,13))
    data.default_root_state[:,3]=1
    data.joint_pos[:2]=.3
    writes=[]
    def write_joints(q,qd,env_ids):
        data.joint_pos[env_ids]=q
        writes.append(('joint',env_ids.clone()))
    def write_roots(root,env_ids):
        assert torch.isfinite(root).all()
        assert (root[:,2]>.003).all()
        torch.testing.assert_close(root[:,3:7].norm(dim=-1),torch.ones(len(root)))
        writes.append(('root',env_ids.clone()))
    robot=SimpleNamespace(data=data,write_joint_state_to_sim=write_joints,write_root_state_to_sim=write_roots)
    scene=Scene(robot=robot);scene.env_origins=torch.zeros(n,3)
    env=SimpleNamespace(scene=scene,device='cpu',num_envs=n,cfg=SimpleNamespace(commands={'transform':SimpleNamespace(training=training)}))
    calls=[]
    def envelope(pose):
        calls.append(pose)
        return [(-.05,-.05),(.05,-.05),(.05,.05),(-.05,.05)],-.5,.5
    def reset(env,ids):
        selected=torch.arange(n) if ids is None else ids
        env.scene['robot'].data.joint_pos[selected]=0
        env.scene['robot'].data.joint_pos[selected[selected<2]]=.3
    monkeypatch.setattr(deployment,'collision_radial_envelope',envelope)
    monkeypatch.setattr(deployment.mdp,'reset_universal',reset)
    params={'bend_fraction':fraction,'bend_limit':.2,'roll_limit_deg':40.}
    term=deployment.BentSnakeReset(SimpleNamespace(params=params),env)
    return env,term,params,writes,calls


def test_subset_and_standing_are_preserved_with_finite_clearance(monkeypatch):
    env,term,params,writes,calls=fixture(monkeypatch)
    term(env,torch.tensor([1,3,5]),**params)
    q=env.scene['robot'].data.joint_pos
    assert (q[:2]==.3).all()
    assert q[[3,5]].abs().amax()<=.2 and (q[[3,5]].abs().amax(dim=-1)>0).all()
    assert (q[[2,4,6,7]]==0).all()
    assert all(set(ids.tolist())<= {3,5} for _,ids in writes)
    assert len(calls)==32
    term(env,torch.tensor([1,3,5]),**params)
    assert len(calls)==32


@pytest.mark.parametrize(('fraction','training'),[(0.,True),(1.,False)])
def test_canonical_and_play_reset_do_not_build_bend_bank(monkeypatch,fraction,training):
    env,term,params,writes,calls=fixture(monkeypatch,fraction=fraction,training=training)
    term(env,None,**params)
    assert not writes and not calls
    assert (env.scene['robot'].data.joint_pos[2:]==0).all()


def test_only_deployment_enables_straightening_and_bent_reset():
    base=universal_env_cfg()
    assert not base.straighten_before_stand
    deployment.configure_deployment(base)
    assert base.straighten_before_stand
    assert base.events['reset_transition'].func is deployment.BentSnakeReset
    assert base.observations['actor'].history_length==5


@pytest.mark.parametrize('bad',[{'bend_fraction':-1.},{'bend_limit':0.},{'bend_limit':float('nan')},{'roll_limit_deg':100.}])
def test_invalid_pose_ranges_fail_before_reset(bad):
    params={'bend_fraction':.5,'bend_limit':.2,'roll_limit_deg':40.}|bad
    with pytest.raises(ValueError):
        deployment.BentSnakeReset(SimpleNamespace(params=params),None)
