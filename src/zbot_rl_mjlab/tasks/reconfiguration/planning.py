"""Offline joint-space closure planning; no state writes during policy execution."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from ...robot import ASSET_PATH, JOINT_NAMES
from .basic.model import head_tail_spec

DEFAULT_PLAN = Path(__file__).resolve().parents[4] / 'assets/reconfiguration/head_tail_joint_plan.json'


class ClosurePlanner:
    def __init__(self, target_limit=2., penetration_tolerance=1e-4):
        self.model = head_tail_spec().compile()
        self.data = mujoco.MjData(self.model)
        self.indices = np.array([self.model.joint(name).qposadr[0] for name in JOINT_NAMES])
        physical = np.array([self.model.joint(name).range for name in JOINT_NAMES])
        self.bounds = np.stack((np.maximum(physical[:, 0], -target_limit),
                                np.minimum(physical[:, 1], target_limit)), axis=-1)
        self.head = self.model.site('foot_0_sole').id
        self.tail = self.model.site('foot_1_sole').id
        self.docks = [self.model.site(f'tail_dock_{i}').id for i in range(4)]
        self.penetration_tolerance = penetration_tolerance

    def forward(self, q):
        self.data.qpos[self.indices] = q
        mujoco.mj_forward(self.model, self.data)
        return self.data

    def error(self, q, quarter=2, gap=.001):
        from scipy.spatial.transform import Rotation
        d = self.forward(q)
        rh = d.site_xmat[self.head].reshape(3, 3)
        rt = d.site_xmat[self.docks[quarter]].reshape(3, 3)
        position = d.site_xpos[self.tail] - d.site_xpos[self.head] + gap * rh[:, 2]
        return np.concatenate((position / .1, Rotation.from_matrix(rh.T @ rt).as_rotvec()))

    def valid(self, q):
        if not np.isfinite(q).all() or (q < self.bounds[:, 0]).any() or (q > self.bounds[:, 1]).any():
            return False
        d = self.forward(q)
        return all(c.dist >= -self.penetration_tolerance for c in d.contact[:d.ncon])

    def edge_valid(self, start, end, resolution=.03):
        count = max(1, math.ceil(float(np.max(np.abs(end-start))) / resolution))
        return all(self.valid(start + (end-start)*f) for f in np.linspace(0., 1., count+1))

    def connect(self, start, goal, seed=42, max_iterations=2500):
        """Try interpolation first, then bidirectional sampling with checked edges."""
        if not self.valid(start) or not self.valid(goal):
            raise ValueError('Planning start or goal is invalid')
        if self.edge_valid(start, goal):
            return np.stack((start, goal))
        rng = np.random.default_rng(seed)
        trees = [([start.copy()], [-1]), ([goal.copy()], [-1])]

        def extend(tree, target):
            points, parents = tree
            nearest = int(np.argmin(np.linalg.norm(np.array(points)-target, axis=-1)))
            delta = target-points[nearest]
            point = points[nearest] + delta * min(1., .15/max(np.linalg.norm(delta), 1e-12))
            if not self.edge_valid(points[nearest], point):
                return None
            points.append(point); parents.append(nearest)
            return len(points)-1

        def trace(tree, index):
            path = []
            while index >= 0:
                path.append(tree[0][index]); index = tree[1][index]
            return path[::-1]

        for iteration in range(max_iterations):
            a = iteration % 2; b = 1-a
            sample = rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
            ia = extend(trees[a], sample)
            if ia is None:
                continue
            target = trees[a][0][ia]
            while True:
                ib = extend(trees[b], target)
                if ib is None:
                    break
                if np.linalg.norm(trees[b][0][ib]-target) < 1e-8:
                    first, second = trace(trees[a], ia), trace(trees[b], ib)
                    path = first + second[-2::-1]
                    return np.array(path if a == 0 else path[::-1])
        raise RuntimeError('No checked joint-space path found within the sampling budget')

    def solve(self, starts=8, seed=42, quarter=2, gap=.001):
        from scipy.optimize import least_squares
        rng = np.random.default_rng(seed)
        pattern = np.array([-1., 1., -1., 1., -1., 1.])
        seeds = [pattern*1.4, -pattern*1.4]
        seeds += [rng.uniform(self.bounds[:, 0]+.05, self.bounds[:, 1]-.05) for _ in range(max(0, starts-2))]
        solutions = []
        for initial in seeds:
            result = least_squares(lambda q: self.error(q, quarter, gap), initial,
                                   bounds=(self.bounds[:, 0]+1e-4, self.bounds[:, 1]-1e-4),
                                   max_nfev=400, ftol=1e-11, xtol=1e-11, gtol=1e-11)
            q = result.x
            if np.linalg.norm(self.error(q, quarter, gap)) > 1e-4 or not self.valid(q):
                continue
            if any(np.linalg.norm(q-item['waypoints'][-1]) < .1 for item in solutions):
                continue
            try:
                path = self.connect(np.zeros(6), q, seed+len(solutions))
            except RuntimeError:
                continue
            d = self.forward(q)
            solutions.append({'waypoints': path.tolist(), 'quarter': quarter,
                              'gap_m': float(np.linalg.norm(d.site_xpos[self.tail]-d.site_xpos[self.head])),
                              'axis_angle_error_rad': float(np.linalg.norm(self.error(q, quarter, gap)[3:]))})
        if not solutions:
            raise RuntimeError('IK found no closure with a valid joint-space path')
        return solutions


def generate_plan(path=DEFAULT_PLAN, starts=8, seed=42):
    planner = ClosurePlanner()
    document = {'version': 1, 'asset_sha256': hashlib.sha256(ASSET_PATH.read_bytes()).hexdigest(),
                'target_limit': 2., 'penetration_tolerance_m': planner.penetration_tolerance,
                'seed': seed, 'solutions': planner.solve(starts, seed)}
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2)+'\n')
    return document


def load_plan(path=DEFAULT_PLAN):
    document = json.loads(Path(path).read_text())
    if document.get('version') != 1 or not document.get('solutions'):
        raise ValueError('Unsupported or empty joint plan')
    if document['asset_sha256'] != hashlib.sha256(ASSET_PATH.read_bytes()).hexdigest():
        raise ValueError('Robot asset changed; regenerate and validate the joint plan')
    for solution in document['solutions']:
        points = np.asarray(solution['waypoints'])
        if points.ndim != 2 or points.shape[1] != 6 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError('Invalid plan waypoints')
        if (np.abs(points) > document['target_limit']+1e-7).any():
            raise ValueError('Plan exceeds target limits')
    return document
