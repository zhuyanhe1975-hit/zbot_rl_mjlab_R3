import numpy as np
import pytest
import torch
from zbot_rl_mjlab.tasks.reconfiguration.batched_teacher_reference import BatchedTeacherReference
from zbot_rl_mjlab.tasks.reconfiguration.full_cpu_student import PHASES


def test_reference_independent_queries_and_absent_keys(tmp_path):
    path=tmp_path/'reference.npz'
    np.savez(path,phase_names=PHASES,phase=[1,1,2],target=[5,5,0],stage=[0,0,1],phase_time=[0.,.02,.0],q=np.array([[1]*6,[2]*6,[3]*6]),qvel=np.zeros((3,6)),qref=np.zeros((3,6)))
    ref=BatchedTeacherReference(path,'cpu')
    data=ref.sample(torch.tensor([1,1,2,2,1]),torch.tensor([5,5,0,1,5]),torch.tensor([0,0,1,1,0]),torch.tensor([.001,.019,1.,0.,float('nan')]))
    assert data['valid'].tolist()==[True,True,True,False,False]
    assert data['q'][:,0].tolist()==[1,2,3,0,0]


def test_reference_rejects_bad_shapes(tmp_path):
    path=tmp_path/'reference.npz'
    np.savez(path,phase_names=PHASES,phase=[0],target=[5],stage=[0],phase_time=[0.],q=np.zeros((1,6)),qvel=np.zeros((1,6)),qref=np.zeros((1,6)))
    ref=BatchedTeacherReference(path,'cpu')
    with pytest.raises(ValueError):ref.sample(torch.tensor([0,0]),torch.tensor([5]),torch.tensor([0]),torch.tensor([0.]))


def test_loading_replay_does_not_treat_post_break_position_as_torque(tmp_path):
    path=tmp_path/'control.npz'
    controls=np.array([[0.]*6,[1.]*6,[99.]*6])
    connected=np.ones((3,6),dtype=bool);connected[2,0]=False
    np.savez(path,phase_names=PHASES,phase=[4]*3,target=[0]*3,stage=[0]*3,
             phase_time=[0.,.02,.04],q=np.zeros((3,6)),qvel=np.zeros((3,6)),
             qref=np.zeros((3,6)),ctrl=controls,connected=connected)
    ref=BatchedTeacherReference(path,'cpu')
    sample=ref.sample(torch.tensor([4]),torch.tensor([0]),torch.tensor([0]),torch.tensor([.04]))
    assert sample['ctrl'][0,0].item()==99.
    assert sample['load_ctrl'][0,0].item()==pytest.approx(2.)
    later=ref.sample(torch.tensor([4]),torch.tensor([0]),torch.tensor([0]),torch.tensor([.06]))
    assert later['load_ctrl_extended'][0,0].item()==pytest.approx(3.)


def test_headtail_replay_preserves_loading_ramp_after_break_sample(tmp_path):
    path=tmp_path/'headtail.npz';connected=np.ones((3,6),dtype=bool);connected[-1,5]=False
    np.savez(path,phase_names=PHASES,phase=[9]*3,target=[5]*3,stage=[4]*3,
             phase_time=[0.,.02,.04],q=np.zeros((3,6)),qvel=np.zeros((3,6)),
             qref=np.zeros((3,6)),ctrl=np.array([[1.]*6,[1.1]*6,[-2.]*6]),connected=connected)
    ref=BatchedTeacherReference(path,'cpu')
    sample=ref.sample(torch.tensor([9]),torch.tensor([5]),torch.tensor([4]),torch.tensor([.04]))
    assert sample['load_ctrl'][0,0].item()==pytest.approx(1.2)


def test_reference_interpolates_within_the_same_phase_only(tmp_path):
    path=tmp_path/'interpolate.npz'
    np.savez(path,phase_names=PHASES,phase=[1,1,2],target=[5,5,0],stage=[0,0,1],
             phase_time=[0.,.02,0.],q=np.zeros((3,6)),qvel=np.zeros((3,6)),
             qref=np.array([[0.]*6,[2.]*6,[7.]*6]))
    ref=BatchedTeacherReference(path,'cpu')
    out=ref.sample(torch.tensor([1,2]),torch.tensor([5,0]),torch.tensor([0,1]),torch.tensor([.01,0.]))
    torch.testing.assert_close(out['qref_interp'][:,0],torch.tensor([1.,7.]))
    torch.testing.assert_close(out['qref_slope'][:,0],torch.tensor([100.,0.]))
