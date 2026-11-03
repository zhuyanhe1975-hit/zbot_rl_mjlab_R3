"""Kinematic graph and target geometry; does not certify the complete IK route."""
import numpy as np
import pytest
from zbot_rl_mjlab.tasks.reconfiguration.kinematic_rz270 import KinematicRz270,KinematicPlanningError


def test_no_fragment_can_be_recorded_when_two_ring_edges_are_open():
    k=KinematicRz270(steps_per_phase=2)
    assert np.isinf(k.bounds).all()
    assert not k.model.jnt_limited[k.joints].any()
    k.record('INITIAL_CLOSED',-1)
    assert k.connectivity()=={'components':1,'connected_faces':6}
    k.active[k.old_eq[5]]=False;k.record('DISCONNECT',5)
    assert k.connectivity()=={'components':1,'connected_faces':5}
    before=len(k.frames)
    k.active[k.old_eq[0]]=False
    with pytest.raises(KinematicPlanningError,match='Disconnected robot is forbidden'):
        k.record('INVALID',0)
    assert len(k.frames)==before


def test_goal_geometry_matches_original_rz270_model_not_path_reachability():
    k=KinematicRz270(steps_per_phase=2)
    # Define the endpoint independently of whether the closed IK path reaches it.
    k.registration[:5]=np.pi/2;k.q[:]=0.
    k.active[k.old_eq]=False;k.active[k.new_eq[:5]]=True
    k.roots,_,_=k.fk(k.q,5,k.roots[0].copy())
    k.record('TARGET_GEOMETRY',5)
    graph=k.connectivity()
    assert graph=={'components':1,'connected_faces':5}
    errors=k.validate_reference();weld=k.active_constraint_errors()
    assert errors['max_root_position_error_m']<1e-8
    assert errors['max_root_angle_error_rad']<1e-8
    assert errors['joint_zero_error_rad']==0.
    assert all(errors['registered_270']) and errors['head_tail_open']
    assert weld['max_active_weld_position_error_m']<1e-8
    assert weld['max_active_weld_angle_error_rad']<1e-8
