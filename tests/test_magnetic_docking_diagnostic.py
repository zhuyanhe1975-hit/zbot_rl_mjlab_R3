"""Small, independent diagnostics; do not claim real hardware stability."""
import numpy as np
import pytest

from scripts.diagnose_magnetic_docking import run


def test_magnetic_free_approach_and_timestep_sensitivity():
    # Before contact, one conservative force sample should attract the
    # two free plates. Refining the step should improve the integration of
    # magnetic plus kinetic energy for this short zero-gravity flight.
    rows=[run('approach',timestep=dt,duration=.04,force_n=.05)
          for dt in (.001,.0005,.00025)]
    assert all(r['finite'] and r['warnings']==0 and r['first_contact_s'] is None for r in rows)
    assert all(0 < r['final_gap_mm'] < r['initial_gap_mm'] for r in rows)
    drift=[abs(r['magnetic_plus_kinetic_energy_J_initial_final'][1]-
               r['magnetic_plus_kinetic_energy_J_initial_final'][0]) for r in rows]
    assert drift[2] < drift[1] < drift[0]
    assert rows[0]['assumptions']['physical_calibration'] is False


def test_attached_phase_uses_actual_mujoco_contact_and_tilt():
    # At the nominal joined pose, actual contact should be reported without
    # any weld. This is a short diagnostic, not a safety or capture guarantee.
    r=run('tilt', timestep=.001, duration=.15, force_n=5.,
          contact_timeconst=.002,tilt_nm=.8)
    assert r['finite'] and r['warnings']==0
    assert r['first_contact_s'] is not None and r['peak_contacts'] > 0
    assert r['peak_penetration_mm'] > 0
    assert np.isfinite(r['max_relative_tilt_deg'])


def test_invalid_contact_impedance_is_rejected():
    with pytest.raises(ValueError):
        run('approach',contact_impedance=(.99,.9))
