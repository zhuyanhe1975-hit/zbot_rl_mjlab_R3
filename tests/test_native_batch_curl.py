import numpy as np
import mujoco

from scripts.search_native_batch_curl import IK_GOAL, run, _connection_losses
from zbot_mjbatch.native_batch import make_native_batch_model
from zbot_mjbatch.native_safety import NativeSafetyAudit


def test_parallel_search_audits_initial_and_each_integrated_state():
    result = run(count=2, threads=2, seconds=.002, seed=7, max_offset=.1)
    assert not result['complete_fsm_success']
    assert not result['curl_success']
    assert len(result['outcomes']) == 2
    for candidate in result['outcomes']:
        # Rejection is evaluated on the integrated state rather than a stale
        # pre-integration contact buffer.  A candidate may be rejected early.
        assert 1 <= candidate['audit']['samples'] <= 3
        if candidate['status'] == 'rejected':
            assert candidate['reason']['time_s'] <= .002
        else:
            assert candidate['status'] == 'unfinished'
            assert candidate['audit']['samples'] == 3
    assert np.max(np.abs(IK_GOAL)) < 6.28
    assert result['outcomes'][0]['offset_rad'] == [0.] * 6


def test_inlet_shell_clearance_only_cuts_inward():
    model, data = make_native_batch_model(inlet_shell_clearance_m=.0001)
    for i in range(6):
        for side, direction, part, expected in [('in', 1, 'a', .0021),
                                                 ('out', -1, 'b', .002)]:
            g = model.geom(f'module_{i}/{part}{i+1}_geom_clipped_shell').id
            s = model.site(f'module_{i}/port_{side}').id
            mesh = model.geom_dataid[g]
            address, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            vertices = (model.mesh_vert[address:address+count]
                        @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g])
            normal = direction * data.site_xmat[s].reshape(3, 3)[:, 2]
            depth = np.min((vertices - data.site_xpos[s]) @ normal)
            assert abs(depth - expected) < 1e-8


def test_native_batch_uses_only_connector_collision_geometry():
    model, _ = make_native_batch_model()
    assert model.npair == 0
    for i in range(model.ngeom):
        name = model.geom(i).name or ''
        if name.startswith('module_'):
            assert model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0
        elif name.startswith('physical_connector_'):
            assert model.geom_contype[i] or model.geom_conaffinity[i]
        else:
            assert name == 'terrain'
            assert model.geom_contype[i] and model.geom_conaffinity[i]


def test_connector_only_initial_and_first_step_audited():
    # Module shells are disabled in this model; this is a narrow check of the
    # connector-only audit, not evidence for the effect of shell clearance.
    model, data = make_native_batch_model(inlet_shell_clearance_m=.0001)
    audit = NativeSafetyAudit(model, data)
    assert audit.sample() == []
    mujoco.mj_step(model, data)
    assert audit.sample() == []


def test_connection_loss_requires_fifty_consecutive_missing_samples():
    elapsed = np.zeros(5)
    for _ in range(49):
        assert len(_connection_losses(elapsed, [0, 1, 1, 1, 1, 0], .001)) == 0
    # One proxy contact remains sufficient, and restores the timer.
    assert len(_connection_losses(elapsed, [1, 1, 1, 1, 1, 0], .001)) == 0
    assert elapsed[0] == 0
    for _ in range(49):
        assert len(_connection_losses(elapsed, [0, 1, 1, 1, 1, 0], .001)) == 0
    assert _connection_losses(elapsed, [0, 1, 1, 1, 1, 0], .001).tolist() == [0]
