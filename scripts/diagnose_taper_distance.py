"""Offline-only comparison of MuJoCo distance with a convex-hull QP.

This diagnostic uses the isolated four-pin model.  It is too slow for the
1 ms control loop, returns zero for overlapping solids (no penetration depth),
and must not be used for safety or as a replacement for collision detection.
"""
import argparse
import time

import mujoco
import numpy as np
from scipy.optimize import minimize

from zbot_mjbatch.taper_pin_prototype import tapered_pin_model


def world_mesh_vertices_mm(model, data, geom):
    """Get compiled convex-mesh vertices in world coordinates, in mm."""
    if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
        raise ValueError('diagnostic accepts mesh geoms only')
    mesh = int(model.geom_dataid[geom])
    first = int(model.mesh_vertadr[mesh])
    vertices = model.mesh_vert[first:first + int(model.mesh_vertnum[mesh])]
    rot = data.geom_xmat[geom].reshape(3, 3)
    return (vertices @ rot.T + data.geom_xpos[geom]) * 1000.


def nonnegative_convex_distance_mm(model, data, a, b):
    """Distance between convex hulls via barycentric QP, mm; zero on overlap.

    SLSQP success and tolerances are not safety certification.  Unlike a signed
    distance, a convex-hull distance cannot report overlap penetration depth.
    """
    va, vb = world_mesh_vertices_mm(model, data, a), world_mesh_vertices_mm(model, data, b)
    na, nb = len(va), len(vb)
    k = np.concatenate((va.T, -vb.T), axis=1)
    constraints = np.zeros((2, na + nb))
    constraints[0, :na] = 1.
    constraints[1, na:] = 1.
    initial = np.r_[np.full(na, 1. / na), np.full(nb, 1. / nb)]
    def objective(weights):
        delta = k @ weights
        return .5 * delta @ delta
    result = minimize(
        objective, initial, method='SLSQP', jac=lambda w: k.T @ (k @ w),
        bounds=[(0., 1.)] * (na + nb),
        constraints={'type': 'eq', 'fun': lambda w: constraints @ w - 1.,
                     'jac': lambda w: constraints},
        options={'ftol': 1e-12, 'maxiter': 2000},
    )
    if not result.success:
        raise RuntimeError(f'QP did not converge: {result.message}')
    return float(np.linalg.norm(k @ result.x))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repeats', type=int, default=10)
    args = p.parse_args()
    if args.repeats < 1:
        p.error('repeats must be positive')
    model = tapered_pin_model()
    for mode, x in [('separated', 0.), ('overlap', .0003)]:
        data = mujoco.MjData(model)
        body = model.body('pin_plate').id
        adr = int(model.jnt_qposadr[model.body_jntadr[body]])
        data.qpos[adr] = x
        data.qpos[adr + 2] = .0005
        mujoco.mj_forward(model, data)
        first, second = model.geom('pin_0').id, model.geom('socket_piece_8').id
        fromto = np.zeros(6)
        engine = mujoco.mj_geomDistance(model, data, first, second, .01, fromto)
        start = time.perf_counter()
        for _ in range(args.repeats):
            exact = nonnegative_convex_distance_mm(model, data, first, second)
        elapsed = (time.perf_counter() - start) * 1000. / args.repeats
        print(f'{mode}: mujoco_mm={engine * 1000:.9g}, '
              f'fromto_distance_mm={np.linalg.norm(fromto[:3] - fromto[3:]) * 1000:.9g}, '
              f'qp_nonnegative_mm={exact:.9g}, mean_qp_ms={elapsed:.3f}, '
              f'solver_contacts={data.ncon}', flush=True)
    print('Offline diagnostic only: a zero QP distance can mean overlap; it is not signed.')


if __name__ == '__main__':
    main()
