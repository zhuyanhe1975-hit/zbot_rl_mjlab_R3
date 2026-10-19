"""部署前进课程在EMA为零时仍有训练覆盖，且不改变回放及变形调度。"""
from types import SimpleNamespace

import pytest
import torch

from zbot_rl_mjlab.tasks.universal.commands import UniversalCommandCfg
from zbot_rl_mjlab.tasks.universal.deployment import DeploymentCommandCfg


def make_command(**kwargs):
    env=SimpleNamespace(num_envs=4096,device='cpu',step_dt=.02,common_step_counter=1)
    cfg=DeploymentCommandCfg(resampling_time_range=(6.,12.),training=True,**kwargs)
    return env,cfg.build(env)


@pytest.mark.parametrize(('stage','maximum'),[(0,.08),(1,.10),(2,.12),(3,.15),(4,.20)])
def test_forward_sampling_reaches_stage_with_zero_ema_and_retains_lower_speeds(stage,maximum):
    _,term=make_command(translation_training_fraction=1.)
    term.set_forward_speed_stage(stage)
    assert term.tracking_ema.eq(0).all()
    torch.manual_seed(29)
    term._sample_velocity(torch.arange(term.num_envs))
    speeds=term.requested_velocity
    assert torch.isclose(speeds[:,0],torch.tensor(maximum)).sum()>200
    assert speeds[:,0].max()<=maximum+1e-6
    assert speeds[:,2].eq(0).all()
    assert (speeds[:,0]<0).sum()>500
    assert (speeds[:,1]>0).sum()>500
    assert (speeds[:,1]<0).sum()>500
    assert ((speeds[:,0]>.04)&(speeds[:,0]<.08)).any()
    assert (speeds.norm(dim=-1)==0).sum()>500
    if stage:
        assert torch.isclose(speeds[:,0],torch.tensor(.08)).any()


def test_default_mixture_preserves_snake_and_other_biped_axes():
    _,term=make_command()
    term.command[2048:,0]=-1
    term.set_forward_speed_stage(4)
    torch.manual_seed(13)
    term._sample_velocity(torch.arange(term.num_envs))
    assert term.requested_velocity[:2048,0].max()>.19
    assert term.requested_velocity[2048:,0].abs().max()<=.05+1e-6
    assert term.requested_velocity[:2048,1:].abs().max()>0
    assert (term.requested_velocity[:2048,0]<0).any()


def test_manual_request_is_not_limited_by_training_stage():
    _,term=make_command()
    term.set_target('stand');term.set_velocity(.2,0.,0.)
    term._resample_command(torch.arange(term.num_envs))
    torch.testing.assert_close(term.requested_velocity[:,0],torch.full((term.num_envs,),.2))
    assert term.forward_speed_stage==0


def test_pretraining_keeps_original_four_scenarios():
    _,term=make_command()
    term.set_forward_speed_stage(4);term.set_pretraining(True)
    ids=torch.arange(term.num_envs)
    term._resample_command(ids)
    assert term.requested_velocity[:,0].max()<=.08+1e-6
    assert term.requested_velocity[ids%4==0,0].eq(.08).all()
    assert term.requested_velocity[ids%4==1,1].eq(.05).all()
    assert term.requested_velocity[ids%4>=2].eq(0).all()


def test_pending_transformation_cannot_be_skipped_by_speed_sampling():
    env,term=make_command()
    term.requested_velocity[:]=.01
    term.start_step[:]=0
    assert env.common_step_counter==1 and term.transition_mask.all()
    before=term.requested_velocity.clone()
    term._resample_command(torch.arange(term.num_envs))
    torch.testing.assert_close(term.requested_velocity,before)


def test_base_command_retains_original_low_ema_behavior():
    env,_=make_command()
    term=UniversalCommandCfg(resampling_time_range=(6.,12.)).build(env)
    term._sample_velocity(torch.arange(term.num_envs))
    assert term.requested_velocity[:,0].abs().max()<=.05+1e-6


@pytest.mark.parametrize('levels',[(),(.1,.08),(.08,.08),(.08,float('nan')),(.08,.3),(0.,.08)])
def test_invalid_speed_levels_are_rejected(levels):
    with pytest.raises(ValueError,match='档位'):
        make_command(forward_speed_levels=levels)


@pytest.mark.parametrize('stage',[-1,5,True,1.5])
def test_invalid_stage_cannot_change_active_curriculum(stage):
    _,term=make_command()
    with pytest.raises(ValueError,match='档位'):
        term.set_forward_speed_stage(stage)
    assert term.forward_speed_stage==0


@pytest.mark.parametrize('direction',['backward','left','right'])
def test_each_direction_has_keyboard_reachable_independent_stages(direction):
    _,term=make_command(translation_training_fraction=1.)
    assert term.course_levels(direction)==pytest.approx((.05,.075,.10))
    term.set_motion_speed_stage(direction,2)
    assert term.forward_speed_stage==0
    assert sum(term.motion_speed_stages.values())==2
    torch.manual_seed(47)
    term._sample_velocity(torch.arange(term.num_envs))
    values=term.requested_velocity
    measured=-values[:,0] if direction=='backward' else values[:,1]*(1 if direction=='left' else -1)
    assert torch.isclose(measured,torch.tensor(.10)).sum()>200
    if direction!='backward':assert values[:,0].min()>=-.05-1e-6
    assert values[:,0].max()<=.08+1e-6


def test_episode_reset_does_not_reset_global_direction_curriculum():
    _,term=make_command()
    term.set_motion_speed_stage('left',2)
    term.motion_speed_passes['left']=1
    term._clear(torch.arange(term.num_envs))
    assert term.motion_speed_stages['left']==2
    assert term.motion_speed_passes['left']==1
