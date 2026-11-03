"""Run the experimental six-module native-contact reconstruction controller.

Pure MuJoCo scene construction and serial live physics; later-stage local
response proposals use mjbatch. Safety audit has documented limits.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid

import mujoco
import numpy as np
from zbot_mjbatch.native_reconfiguration import NativeReconfiguration, Config
from zbot_mjbatch.native_batch import DEFAULT_CONTACT_FORCE_N


def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration-s',type=float,default=None,help='optional maximum simulated duration in seconds')
    parser.add_argument('--joint-target-speed',type=float,default=.2,help='experimental target velocity in rad/s')
    parser.add_argument('--contact-force-n',type=float,default=DEFAULT_CONTACT_FORCE_N,help='assumed uncalibrated native adhesion capacity per contact (N)')
    parser.add_argument('--output',type=Path,default=Path('outputs/reconfiguration/native_fsm/result.json'))
    parser.add_argument('--no-index', action='store_true', help='do not update the latest-execution index')
    args=parser.parse_args()
    if args.duration_s is not None and not 0<args.duration_s<float('inf'):parser.error('--duration-s must be finite and positive')
    if not 0<args.joint_target_speed<float('inf'):parser.error('--joint-target-speed must be finite and positive')
    if not 0<args.contact_force_n<float('inf'):parser.error('--contact-force-n must be finite and positive')
    sim=NativeReconfiguration(Config(joint_target_speed=args.joint_target_speed,contact_force_n=args.contact_force_n))
    # Record the actual integrated state, including failures and the initial
    # frame. Unique filenames keep an interrupted or concurrent run from
    # pairing a report with another execution's model or trajectory.
    times=[]; qpos=[]; qvel=[]; ctrl=[]; phases=[]; targets=[]
    def record():
        times.append(float(sim.data.time)); qpos.append(sim.data.qpos.copy())
        qvel.append(sim.data.qvel.copy()); ctrl.append(sim.data.ctrl.copy())
        phases.append(sim.phase); targets.append(sim.target + 1)
    record()
    last=0;next_progress=10.
    interrupted=False
    try:
        while sim.phase not in ('DONE','FAILED') and (args.duration_s is None or sim.data.time<args.duration_s):
            before=float(sim.data.time)
            sim.step()
            if sim.data.time > before:
                record()
            elif sim.phase in ('DONE','FAILED'):
                # A terminal transition can happen without another physical step.
                phases[-1]=sim.phase; targets[-1]=sim.target + 1
            if len(sim.events)>last:
                event=sim.events[-1]
                print(f"NATIVE_FSM {event['time_s']:.3f}s {event['phase']} interface={event['target_interface']}",flush=True)
                last=len(sim.events)
            if sim.data.time>=next_progress and sim.phase not in ('DONE','FAILED'):
                status=sim.interfaces()[sim.target]
                print(f"NATIVE_PROGRESS {sim.data.time:.3f}s {sim.phase} interface={sim.target+1} "
                      f"gap={status['position_m']:.6f}m angle={status['angle_rad']:.6f}rad "
                      f"contacts={status['contacts']}",flush=True)
                next_progress+=10.
    except KeyboardInterrupt:
        interrupted=True
        print('NATIVE_FSM interrupted; saving recorded trajectory',flush=True)
    finally:
        sim.close()
    result=sim.report();result['interrupted']=interrupted
    result['timeout']=not interrupted and sim.phase not in ('DONE','FAILED')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    run_id=uuid.uuid4().hex
    trajectory=args.output.parent / f'{args.output.stem}.{run_id}.trajectory.npz'
    model=args.output.parent / f'{args.output.stem}.{run_id}.model.mjb'
    mujoco.mj_saveModel(sim.model, str(model))
    np.savez_compressed(trajectory, time=np.asarray(times), qpos=np.asarray(qpos),
                        qvel=np.asarray(qvel), ctrl=np.asarray(ctrl),
                        phase=np.asarray(phases), target_interface=np.asarray(targets))
    result.update({'run_id':run_id, 'trajectory_file':trajectory.name, 'model_file':model.name,
                   'trajectory_sha256':_sha256(trajectory), 'model_sha256':_sha256(model),
                   'trajectory_samples':len(times), 'trajectory_sampling_s':sim.cfg.dt,
                   'mujoco_version':mujoco.__version__})
    serialized=json.dumps(result,indent=2)+'\n'
    # The immutable report lets an indexed execution remain paired with its
    # own snapshots even when multiple runs share the same --output path.
    archived=args.output.parent / f'{args.output.stem}.{run_id}.report.json'
    archived.write_text(serialized)
    temporary=args.output.parent / f'.{args.output.name}.{run_id}.tmp'
    temporary.write_text(serialized)
    os.replace(temporary,args.output)
    if not args.no_index:
        index_dir=Path(__file__).resolve().parents[1] / 'outputs/reconfiguration/native_fsm'
        index_dir.mkdir(parents=True,exist_ok=True)
        index_tmp=index_dir / f'.latest_execution.{run_id}.tmp'
        index_tmp.write_text(json.dumps({'result_path':str(archived.resolve()),'run_id':run_id})+'\n')
        os.replace(index_tmp,index_dir / 'latest_execution.json')
    print('NATIVE_RESULT',json.dumps({k:result[k] for k in ('success','phase','failure','time_s','timeout','max_penetration_m','worst_contact')}),flush=True)
    print(f'REPORT {args.output}',flush=True)
    return 130 if interrupted else (0 if result['success'] else 1)

if __name__=='__main__':raise SystemExit(main())
