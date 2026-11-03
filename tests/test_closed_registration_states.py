"""Discrete closure certificates and independent Jacobian verification."""
import numpy as np
import pytest
from zbot_rl_mjlab.tasks.reconfiguration.closure_states import ClosedRegistrationAudit


@pytest.fixture(scope='module',params=[90,270])
def audit(request):return ClosedRegistrationAudit(request.param)


def test_analytic_loop_jacobian_matches_independent_finite_differences(audit):
    for q,bits in [(audit.robot.qref,[0]*6),(np.array([.1,.2,-.3,.4,-.5,.6]),[1,0,1,0,0,1])]:
        h=1e-6
        fd=np.stack([(audit.evaluate(q+v,bits)-audit.evaluate(q-v,bits))/(2*h)
                     for v in np.eye(6)*h],axis=1)
        assert np.allclose(fd,audit.evaluate(q,bits,True),atol=1e-6,rtol=1e-5)


def test_both_uniform_registrations_have_mujoco_verified_static_solutions(audit):
    for bits in ([0]*6,[1]*6):
        rng=np.random.default_rng(180270+sum(b<<i for i,b in enumerate(bits)))
        seeds=[audit.robot.qref,np.zeros(6)]+[rng.uniform(-np.pi,np.pi,6) for _ in range(8)]
        result=audit.solve(bits,seeds)
        assert result['status']=='solution_found'
        assert result['mujoco_verification']['max_active_weld_position_error_m']<audit.position_tolerance
        assert result['mujoco_verification']['max_active_weld_angle_error_rad']<audit.angle_tolerance


def test_numerical_miss_is_reported_as_not_found_not_proved_infeasible(audit):
    result=audit.solve([1,1,1,1,1,0],[np.zeros(6)],max_nfev=1)
    assert result['status']=='not_found'
    assert 'infeasible' not in result['status']


def test_alternating_registration_jacobian_and_five_face_closure():
    audit=ClosedRegistrationAudit(face_angles_deg=[270,90,270,90,270,90])
    q=np.array([.1,.2,-.3,.4,-.5,.6]);bits=[1]*6;h=1e-6
    fd=np.stack([(audit.evaluate(q+v,bits)-audit.evaluate(q-v,bits))/(2*h)
                 for v in np.eye(6)*h],axis=1)
    assert np.allclose(fd,audit.evaluate(q,bits,True),atol=1e-6,rtol=1e-5)
    rng=np.random.default_rng(20261028)
    seeds=[audit.robot.qref,np.zeros(6)]+[rng.uniform(-np.pi,np.pi,6) for _ in range(32)]
    result=audit.solve([1,1,1,1,1,0],seeds,max_nfev=300)
    assert result['status']=='solution_found'
    assert result['mujoco_verification']['max_active_weld_position_error_m']<=audit.position_tolerance
    assert result['mujoco_verification']['max_active_weld_angle_error_rad']<=audit.angle_tolerance
