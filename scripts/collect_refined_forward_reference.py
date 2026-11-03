"""Record the actual forward refined 1 ms FSM at the 20 ms control rate."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from zbot_rl_mjlab.tasks.reconfiguration.full_cpu_student import PHASES
import scripts.view_headtail_mjbatch_refined as refined

DEFAULT=Path('assets/reconfiguration/refined_forward_velocity_reference.npz')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args()
    f=refined.create_refined('forward');rows=[];omitted={}
    try:
        substeps=round(f.cfg.control_dt/f.cfg.dt)
        while f.phase not in ('DONE','FAILED') and f.data.time<120.:
            # This physical waiting phase has no corresponding action in the
            # legacy GPU FSM and provides no moving velocity reference.
            if f.phase not in PHASES:
                omitted[f.phase]=omitted.get(f.phase,0)+1
            else:
                rows.append((float(f.data.time),PHASES.index(f.phase),f.target,f.stage,float(f.phase_time),
                             f.data.qpos[f.qadr].copy(),f.data.qvel[f.dofs].copy(),f.qref.copy(),
                             f.data.ctrl[f.actuators].copy(),f.loads.copy(),f.connected().copy()))
            for _ in range(substeps):
                f.step()
                if f.phase in ('DONE','FAILED'):break
        if f.phase!='DONE':raise RuntimeError(f'forward source failed: {f.failure}')
        columns=list(zip(*rows));args.output.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.output,sim_time=columns[0],phase=columns[1],target=columns[2],stage=columns[3],
            phase_time=columns[4],q=columns[5],qvel=columns[6],qref=columns[7],ctrl=columns[8],loads=columns[9],
            connected=columns[10],phase_names=np.asarray(PHASES),control_dt=f.cfg.control_dt,
            reference_version=np.asarray('refined_forward_90_overlap_v1'),headtail_registration_deg=np.asarray(90),
            synchronized_separation_rotation=np.asarray(True),physics_dt=np.asarray(f.cfg.dt))
        report=f.report();report['reference']={'path':str(args.output),'sha256':sha(args.output),'samples':len(rows),
            'source_script':'scripts/view_headtail_mjbatch_refined.py','source_sha256':sha('scripts/view_headtail_mjbatch_refined.py'),
            'forward_unfold_reference_sha256':sha(refined.REFERENCE),'omitted_nonmoving_phases':omitted,
            'velocity_definition':'actual simulated joint qvel in rad/s at the control sampling instants',
            'headtail_registration_deg':90,'synchronized_separation_rotation':True}
        args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
        print('REFERENCE',args.output,'samples',len(rows),'duration',f.data.time,'other_peak',f.peak_other,'sha256',report['reference']['sha256'],flush=True)
    finally:f.close()

if __name__=='__main__':main()
