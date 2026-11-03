import numpy as np
from zbot_rl_mjlab.tasks.reconfiguration.static_loading_milp import solve_static_loading


def test_minimizes_worst_other_load_for_fixed_target():
    A=np.zeros((6,6,6));w=np.zeros((6,6))
    A[0,2,0]=10.;A[1,2,0]=8.;A[1,2,1]=-8.;A[2,2,1]=8.
    result=solve_static_loading(A,w,0)
    assert result['success']
    np.testing.assert_allclose(result['torque'][:2],[10.,5.],atol=.03)
    assert result['predicted_loads'][0]>=100.-1e-5
    assert result['other_peak']<=40.1


def test_infeasible_target_returns_failure():
    A=np.zeros((6,6,6));w=np.zeros((6,6));A[0,2,0]=1.
    result=solve_static_loading(A,w,0)
    assert not result['success']


def test_actual_elliptic_peel_stays_within_other_limit():
    A=np.zeros((6,6,6));w=np.zeros((6,6));A[0,2,0]=10.
    # A fixed diagonal moment is near the polygon approximation boundary.
    w[1,3:5]=[3.5,3.5]
    result=solve_static_loading(A,w,0,other_limit=100.)
    assert result['success']
    assert result['predicted_loads'][1]<100.
