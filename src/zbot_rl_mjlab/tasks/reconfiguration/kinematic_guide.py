"""Read-only joint reference from the existing connected kinematic FSM."""
import contextlib
import hashlib
import io
from functools import lru_cache

import numpy as np
import torch

from .kinematic_mixed_closed import MixedClosedKinematics
from .state_machine import Phase


@lru_cache(maxsize=4)
def planned_reference(duration):
    with contextlib.redirect_stdout(io.StringIO()):
        planner = MixedClosedKinematics(steps_per_phase=20, phase_duration=duration)
        frames = planner.plan()
    checks = planner.target_checks()
    if not checks['target_angles_match'] or not checks['connected_all_frames']:
        raise RuntimeError('Kinematic guide failed target/connectivity validation')
    segments = {}
    for connector in planner.target_order:
        for phase in (Phase.SEPARATE, Phase.ROTATE, Phase.APPROACH):
            selected = [(i, f) for i, f in enumerate(frames) if f.connector == connector and f.phase == phase.name]
            start = frames[selected[0][0]-1].q
            samples = np.stack([start, *(f.q for _, f in selected)])
            segments[(connector, int(phase))] = samples
    digest = hashlib.sha256(b''.join(v.tobytes() for v in segments.values())).hexdigest()
    return segments, {'method': 'existing_connected_mixed_kinematic_state_machine', 'sha256': digest,
                       'steps_per_phase': 20, 'duration_s': duration,
                       'target_clocking_deg': [270,270,180,90,90],
                       'collision_free': False, 'usage': 'soft joint velocity guide; physical safety guards remain active'}


class KinematicGuide:
    def __init__(self, env):
        samples, self.report = planned_reference(env.cfg.motion_duration_s)
        self.samples = {k: torch.as_tensor(v, device=env.device, dtype=torch.float32) for k,v in samples.items()}
        self.duration = env.cfg.motion_duration_s

    def reference(self, controller, position):
        key = (controller.env.cfg.target_connector, int(controller.phase.item()))
        if key not in self.samples:
            return position.clone(), torch.zeros_like(position)
        samples = self.samples[key]
        # Lookahead compensates for integration/servo lag; phase transitions
        # and physical weld activation still depend on measured state.
        time = (controller.phase_time + .10).clamp(0., self.duration)
        t = time / self.duration * (len(samples)-1)
        index = t.long().clamp(0, len(samples)-2)
        fraction = (t-index).clamp(0., 1.)[:, None]
        q = samples[index] + fraction * (samples[index+1]-samples[index])
        velocity = (samples[index+1]-samples[index]) * ((len(samples)-1)/self.duration)
        velocity = torch.where((time < self.duration)[:, None], velocity, 0.)
        return q, velocity
