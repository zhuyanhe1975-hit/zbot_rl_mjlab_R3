"""Connected kinematic FSM invariants, registration signs and full geometry path."""
import numpy as np
import pytest
from zbot_rl_mjlab.tasks.reconfiguration.kinematic_mixed_closed import MixedClosedKinematics,TARGET_DEG
from zbot_rl_mjlab.tasks.reconfiguration.kinematic_rz270 import KinematicPlanningError


def test_mixed_target_rejects_changes_to_preserved_connections():
    with pytest.raises(ValueError):MixedClosedKinematics(order=(0,1,2,4))
    k=MixedClosedKinematics(steps_per_phase=2)
    assert np.isinf(k.bounds).all() and not k.model.jnt_limited[k.joints].any()
    k.record('INITIAL_CLOSED',-1)
    for face in (2,5):
        k.active[k.old_eq[face]]=False
        with pytest.raises(KinematicPlanningError):k.record('INVALID',face)
        assert len(k.frames)==1
        k.active[k.old_eq[face]]=True
    k.active[k.old_eq[0]]=False;k.record('DISCONNECT',0)
    k.active[k.old_eq[1]]=False
    with pytest.raises(KinematicPlanningError,match='Disconnected robot'):k.record('INVALID',1)
    assert len(k.frames)==2


def test_full_mixed_kinematic_fsm_preserves_single_chain_and_rebuilds_target():
    k=MixedClosedKinematics(steps_per_phase=20)
    k.plan();checks=k.target_checks()
    assert k.events[0]['phase']=='INITIAL_CLOSED' and k.events[-1]['phase']=='DONE'
    assert [e['connection'] for e in k.events if e['phase']=='REBUILD_WELD']==[1,2,4,5]
    assert checks['internal_angles_deg']==list(TARGET_DEG)
    assert checks['target_angles_match'] and checks['variant_selection_matches']
    assert checks['head_tail_original_closed'] and checks['middle_connection_original']
    assert checks['connected_all_frames'] and checks['min_connected_faces']==5 and checks['final_connected_faces']==6
    assert checks['max_active_weld_position_error_m']<1e-6
    assert checks['max_active_weld_angle_error_rad']<1e-5
    for f in k.frames:
        assert f.eq_active[k.old_eq[5]] and not f.eq_active[k.new_eq[5]]
        assert f.eq_active[k.old_eq[2]] and not f.eq_active[k.new_eq[2]]
    assert len(k.stage_errors)==12 and abs(k.time-12.)<1e-8
