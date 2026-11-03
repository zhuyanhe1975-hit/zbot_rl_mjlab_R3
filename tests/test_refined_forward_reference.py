import numpy as np
import torch

from zbot_rl_mjlab.tasks.reconfiguration.batched_teacher_reference import BatchedTeacherReference
from zbot_rl_mjlab.tasks.reconfiguration.full_batched_env import make_full_batched_env_cfg
from zbot_rl_mjlab.tasks.reconfiguration.full_batched_fsm import FullBatchedFSM,P
from zbot_rl_mjlab.tasks.reconfiguration.full_cpu_student import PHASES

REFERENCE='assets/reconfiguration/refined_forward_velocity_reference.npz'


def test_real_reference_velocity_and_phase_coverage():
    with np.load(REFERENCE,allow_pickle=False) as d:
        assert str(d['reference_version'])=='refined_forward_90_overlap_v1'
        assert int(d['headtail_registration_deg'])==90
        assert d['qvel'].shape==(len(d['phase']),6)
        assert np.isfinite(d['qvel']).all() and abs(d['qvel']).max()>3
        for target,stage in ((0,0),(1,1),(3,2),(4,3)):
            for name in ('SEPARATE','APPROACH'):
                assert np.any((d['phase']==PHASES.index(name))&(d['target']==target)&(d['stage']==stage))
        assert not np.any(d['phase']==PHASES.index('ROTATE'))
        index=np.flatnonzero(d['phase']==PHASES.index('CURL'))[10]
        expected=d['qvel'][index].copy();t=d['phase_time'][index]
    ref=BatchedTeacherReference(REFERENCE,'cpu')
    got=ref.sample(torch.tensor([P['CURL']]),torch.tensor([5]),torch.tensor([0]),torch.tensor([t]))
    assert got['valid'].item()
    np.testing.assert_allclose(got['qvel'][0].numpy(),expected,atol=1e-6)


def test_refined_state_machine_overlaps_rotation_and_preserves_release_hold():
    f=FullBatchedFSM(1,'cpu',refined=True)
    connected=torch.ones(1,6,dtype=torch.bool);reg=torch.zeros(1,5,dtype=torch.bool)
    loads=torch.zeros(1,6);g={name:torch.zeros(1) for name in ('gap','angle','linear_speed','angular_speed')}
    done=torch.ones(1,dtype=torch.bool);false=torch.zeros(1,dtype=torch.bool)
    def update():return f.update(connected,reg,loads,g,done,false,false,false)
    f.phase[:]=P['SEPARATE'];f.target[:]=0;connected[0,0]=False;g['gap'][:]=.021
    update();assert f.phase.item()==P['APPROACH']
    f.phase[:]=P['LOAD_HEADTAIL'];f.target[:]=5;f.stage[:]=4;connected[0,0]=True;connected[0,5]=False
    update();assert f.phase.item()==P['SETTLE_HEADTAIL']
    for _ in range(24):update()
    assert f.phase.item()==P['SETTLE_HEADTAIL']
    update();assert f.phase.item()==P['UNFOLD']


def test_checkpoint_reference_identity_changes_with_data(tmp_path):
    from scripts.train_full_batched_student import REFINED_MODE, reference_identity
    source=np.load(REFERENCE,allow_pickle=False)
    payload={key:source[key].copy() for key in source.files};source.close()
    first=tmp_path/'teacher.npz';second=tmp_path/'changed.npz'
    np.savez_compressed(first,**payload)
    payload['qvel'][0,0]+=.01
    np.savez_compressed(second,**payload)
    mode_a,sha_a=reference_identity(first);mode_b,sha_b=reference_identity(second)
    assert mode_a==mode_b==REFINED_MODE and sha_a!=sha_b


def test_refined_registered_site_matches_cpu_model():
    from mjlab.scene import Scene
    import scripts.view_headtail_search as source
    from types import SimpleNamespace
    cfg=make_full_batched_env_cfg(1,REFERENCE)
    assert cfg.full_refined_reference and not cfg.full_curl_symmetric
    gpu_model=Scene(cfg.scene,device='cpu').compile()
    cpu=source.mod.FullCpuReconfiguration(source.make_config(SimpleNamespace(speed_deg_s=136.5,accel_deg_s2=1656.2)))
    try:
        source.prepare(cpu,90,[0]*6)
        a=gpu_model.site('module_5/port_out_refined90').id;b=cpu.model.site('module_5/search_port').id
        np.testing.assert_allclose(gpu_model.site_pos[a],cpu.model.site_pos[b],atol=1e-12)
        np.testing.assert_allclose(gpu_model.site_quat[a],cpu.model.site_quat[b],atol=1e-12)
        assert gpu_model.eq_obj1id[gpu_model.equality('connector_5').id]==a
    finally:cpu.close()
