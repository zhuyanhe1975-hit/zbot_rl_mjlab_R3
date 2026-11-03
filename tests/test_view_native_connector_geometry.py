import mujoco
import numpy as np
import pytest

from scripts.view_native_connector_geometry import make_preview


@pytest.mark.parametrize('face', [0, 3, 5])
def test_exploded_preview_moves_only_selected_modules(face):
    model, data, center = make_preview(face, .02)
    base = mujoco.MjData(model)
    base.qpos[:] = model.qpos0
    mujoco.mj_forward(model, base)
    target = (face + 1) % 6
    port = model.site(f'module_{target}/port_in').id
    normal = base.site_xmat[port].reshape(3, 3)[:, 2]
    for module in range(6):
        adr = int(model.joint(f'module_{module}/module_root').qposadr[0])
        shift = data.qpos[adr:adr+3] - base.qpos[adr:adr+3]
        expected = (-.01 if module == face else .01 if module == target else 0.) * normal
        np.testing.assert_allclose(shift, expected, atol=1e-12)
        np.testing.assert_allclose(data.qpos[adr+3:adr+7], base.qpos[adr+3:adr+7], atol=0)
    assert data.time == 0.
    assert np.isfinite(center).all()
    # The preview retains all four pins and both sets of magnetic proxies.
    for i in range(4):
        assert model.geom(f'physical_connector_{face}_pin_{i}').id >= 0
        for side in ('source', 'target'):
            geom = model.geom(f'physical_connector_{face}_adhesion_{side}_{i}').id
            assert model.geom_rgba[geom, 3] > 0
    other = model.geom(f'physical_connector_{(face+1)%6}_pin_0').id
    assert model.geom_rgba[other, 3] == 0


@pytest.mark.parametrize('face, gap', [(-1, .02), (6, .02), (0, -.01), (0, float('nan'))])
def test_invalid_preview_arguments_rejected(face, gap):
    with pytest.raises(ValueError):
        make_preview(face, gap)
