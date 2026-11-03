"""Experimental parallel native-adhesion CURL search with 1 ms serial audit.

The former closed-chain IK solution supplies only a target configuration.  It
does not supply dynamics, welds or a reference trajectory.  Each candidate is
integrated by mjbatch.Batch; the current integration state is then independently
forwarded and audited at every physical step because Batch does not expose
MuJoCo's variable-length contact array.  This is a CURL experiment, not the
complete reconfiguration state machine.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import mujoco
import numpy as np

from zbot_mjbatch.native_batch import NativeAdhesionBatch
from zbot_mjbatch.native_safety import NativeSafetyAudit

# Geometry-only solution of the former six-joint closed-chain IK.  It is a
# candidate target, not evidence that this new contact model can reach it.
IK_GOAL = np.array([-.7151050279933058, 2.663656161069831,
                    2.341859434055353, -1.093611347372558,
                    1.8997523394142295, -2.8300742095870532])


def _connections(model, scratch):
    """Count paired native magnetic proxy contacts, not just nearby sites."""
    pairs = [set() for _ in range(6)]
    names = {model.geom(f'physical_connector_{i}_adhesion_{side}_{j}').id: (i, side, j)
             for i in range(6) for side in ('source', 'target') for j in range(4)}
    for contact in scratch.contact:
        a = names.get(int(contact.geom1)); b = names.get(int(contact.geom2))
        if a is None or b is None or a[0] != b[0] or a[1] == b[1]:
            continue
        # A contact in the native gap is an adhesive constraint candidate.
        if contact.dist <= .003 + 1e-12:
            pairs[a[0]].add((a[2], b[2]) if a[1] == 'source' else (b[2], a[2]))
    def maximum_matching(p):
        # Distinct sources must pair with distinct targets; merely seeing all
        # four sources can overstate connected magnets if one target is shared.
        assigned = {}
        def visit(source, seen):
            for s, target in p:
                if s != source or target in seen:
                    continue
                seen.add(target)
                if target not in assigned or visit(assigned[target], seen):
                    assigned[target] = source
                    return True
            return False
        return sum(visit(source, set()) for source in range(4))
    return [maximum_matching(p) for p in pairs]


def _connection_losses(loss_s, contacts, dt, confirm_s=.05):
    """Track continuous absence of all native proxy contacts on faces 0–4."""
    loss_s[:] = np.where(np.asarray(contacts[:5]) > 0, 0., loss_s + dt)
    return np.flatnonzero(loss_s + 1e-12 >= confirm_s)


def run(count, threads, seconds, seed, max_offset, min_duration=2., max_duration=8.):
    if (count < 1 or threads < 1 or not np.isfinite([seconds, max_offset, min_duration, max_duration]).all()
            or seconds <= 0 or max_offset < 0 or not 0 < min_duration <= max_duration):
        raise ValueError('count/threads/duration/offset must be positive')
    start_wall = time.perf_counter()
    env = NativeAdhesionBatch(count, threads)
    model = env.model
    qadr = np.array([model.jnt_qposadr[model.joint(f'module_{i}/joint{i+1}').id] for i in range(6)])
    init = env.qpos[:, qadr].copy()
    rng = np.random.default_rng(seed)
    offsets = rng.uniform(-max_offset, max_offset, (count, 6))
    offsets[0] = 0.
    durations = np.linspace(min_duration, max_duration, count)
    audits = [NativeSafetyAudit(model, mujoco.MjData(model)) for _ in range(count)]
    loss_s = np.zeros((count, 5))
    active = np.ones(count, dtype=bool)
    outcomes = [None] * count
    state_type = mujoco.mjtState.mjSTATE_INTEGRATION

    def sample(i):
        audit = audits[i]; data = audit.data
        mujoco.mj_setState(model, data, env.state[i], state_type)
        data.actuator_force[:] = env.actuator_force[i]
        data.warning.lastinfo[:] = env.warning[i, :, 0]
        data.warning.number[:] = env.warning[i, :, 1]
        issues = audit.sample()
        contacts = _connections(model, audit.scratch)
        lost = _connection_losses(loss_s[i], contacts, .001 if audit.samples > 1 else 0.)
        if len(lost):
            issues.append({'kind': 'non_target_magnetic_contact_lost',
                           'time_s': float(data.time), 'faces': lost.tolist(),
                           'proxy_contacts': contacts,
                           'continuous_loss_s': loss_s[i].tolist()})
        return issues, contacts

    # Check initial state as well as all integrated states.
    for i in range(count):
        issues, contacts = sample(i)
        if issues:
            active[i] = False
            outcomes[i] = {'status': 'rejected', 'reason': issues[0], 'contacts': contacts}
    nsteps = round(seconds / .001)
    for step in range(nsteps):
        ids = np.flatnonzero(active)
        if not len(ids):
            break
        t = (step + 1) * .001
        progress = np.minimum(t / durations[ids], 1.)
        # A smooth bounded 6-D parameterized position command; candidate controls
        # are generated online, with no previously recorded trajectory.
        p = progress[:, None]
        target = init[ids] + (IK_GOAL - init[ids]) * p + offsets[ids] * np.sin(np.pi * p)
        env.ctrl[ids] = np.clip(target, -6.28, 6.28)
        env.step(ids, nstep=1)
        for i in ids:
            issues, contacts = sample(int(i))
            if issues:
                active[i] = False
                outcomes[i] = {'status': 'rejected', 'reason': issues[0], 'contacts': contacts}
    for i in range(count):
        if outcomes[i] is None:
            # This search never asserts capture: closed orientation, physical
            # contact, holding and the remainder of the FSM need validation.
            outcomes[i] = {'status': 'unfinished', 'reason': 'no closure/hold success test',
                           'contacts': _connections(model, audits[i].scratch)}
        outcomes[i].update({'candidate': i, 'command_duration_s': float(durations[i]),
                            'offset_rad': offsets[i].tolist(), 'audit': audits[i].report()})
    return {'method': 'mjbatch.Batch forward dynamics; serial MuJoCo contact audit every 1 ms',
            'complete_fsm_success': False, 'curl_success': False,
            'wall_time_s': time.perf_counter() - start_wall,
            'count': count, 'threads': min(threads, count), 'horizon_s': seconds,
            'seed': seed, 'duration_range_s': [min_duration, max_duration],
            'max_offset_rad': max_offset, 'outcomes': outcomes,
            'connection_policy': 'any native proxy contact per non-target interface; '
                                 'continuous absence for 0.05 s indicates lost connection',
            'limitations': '1 ms discrete checks only; filtered contacts and tunnelling are not excluded; '
                           'passive adhesion and geometric dimensions are uncalibrated assumptions; '
                           'no successful closure or full reconstruction is asserted.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--count', type=int, default=8)
    p.add_argument('--threads', type=int, default=len(os.sched_getaffinity(0)))
    p.add_argument('--seconds', type=float, default=8.)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--max-offset', type=float, default=.3)
    p.add_argument('--min-duration', type=float, default=2.)
    p.add_argument('--max-duration', type=float, default=8.)
    args = p.parse_args()
    result = run(args.count, args.threads, args.seconds, args.seed, args.max_offset,
                 args.min_duration, args.max_duration)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
