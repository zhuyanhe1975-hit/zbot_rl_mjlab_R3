"""Old 180-degree assembly plus initially inactive end-to-end loop constraints.

The compiled serial tree never changes. Activating one equality adds a real
closed-loop edge; no body transform or qpos is rewritten at capture. Internal
module connections remain fixed in experiment 1.
"""

import math

import mujoco
import numpy as np

from ...transformation.env_cfg import full_body_spec

DOCK_SITES = tuple(f'tail_dock_{i}' for i in range(4))
WELD_NAMES = tuple(f'head_tail_weld_{i}' for i in range(4))
TORQUE_SCALE = .05


def head_tail_spec():
    spec = full_body_spec()
    for joint in spec.joints:
        if joint.name.startswith("joint"):
            joint.range = (-math.pi, math.pi)
    tail = next(body for body in spec.bodies if body.name == 'foot_1')
    sole = next(site for site in spec.sites if site.name == 'foot_1_sole')
    for i, (site_name, weld_name) in enumerate(zip(DOCK_SITES, WELD_NAMES, strict=True)):
        angle = i * math.pi / 2
        quat = np.empty(4)
        mujoco.mju_mulQuat(quat, sole.quat, np.array((math.cos(angle / 2), 0., 0., math.sin(angle / 2))))
        tail.add_site(name=site_name, pos=sole.pos, quat=quat, size=(.006,) * 3,
                      rgba=(1., .6, .1, 0.))
        spec.add_equality(name=weld_name, type=mujoco.mjtEq.mjEQ_WELD,
                          objtype=mujoco.mjtObj.mjOBJ_SITE,
                          name1='foot_0_sole', name2=site_name, active=False,
                          data=(*([0.] * 10), TORQUE_SCALE),
                          solref=(.02, 1.), solimp=(.9, .95, .001, .5, 2.))
    return spec
