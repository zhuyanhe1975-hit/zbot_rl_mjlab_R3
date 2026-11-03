"""Basic physical sanity checks for the isolated tapered-pin load study."""
import pytest

from scripts.diagnose_taper_contact import run


@pytest.mark.parametrize('gap_m', [.0005,.001,.0015])
def test_unloaded_aligned_pins_do_not_touch_or_drift(gap_m):
    row=run(gap_m,.001,load_N=0.,duration_s=.05)
    assert row['initial_contacts']==0 and row['contact_steps']==0
    assert row['max_abs_relative_x_mm'] < 1e-8
    assert row['max_gap_drift_mm'] < 1e-8
    assert row['finite'] and row['warnings']==0


@pytest.mark.parametrize('gap_m', [.0005,.001,.0015])
def test_lateral_load_reacts_and_final_state_is_similar_at_smaller_step(gap_m):
    one=run(gap_m,.001)
    half=run(gap_m,.0005)
    assert one['contact_steps']>0 and one['peak_pin_wall_contacts']>0
    assert one['peak_sum_pin_wall_normal_force_N']>0
    assert one['finite'] and half['finite'] and one['warnings']==half['warnings']==0
    assert abs(one['final_relative_xyz_mm'][0]-half['final_relative_xyz_mm'][0]) < .005
    assert abs(one['final_relative_xyz_mm'][2]-half['final_relative_xyz_mm'][2]) < .005
    # Checks this explicitly damped diagnostic only, not a real safety limit.
    assert 0 < one['max_pin_wall_penetration_mm'] < .06
