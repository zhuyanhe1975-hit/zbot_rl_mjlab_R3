"""Record real full-FSM MuJoCo dynamics at the 50 Hz control rate."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,json
from pathlib import Path
import numpy as np
from zbot_rl_mjlab.tasks.reconfiguration.full_cpu import FullCpuReconfiguration
from zbot_rl_mjlab.tasks.reconfiguration.full_cpu_student import PHASES

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('outputs/reconfiguration/full_cpu_student/teacher_reference.npz'));p.add_argument('--max-time',type=float,default=120.);a=p.parse_args()
    f=FullCpuReconfiguration();rows=[]
    try:
        while f.phase not in ('DONE','FAILED') and f.data.time<a.max_time:
            rows.append((f.data.time,PHASES.index(f.phase),f.target,f.stage,f.phase_time,f.data.qpos[f.qadr].copy(),f.data.qvel[f.dofs].copy(),f.qref.copy(),f.data.ctrl[f.actuators].copy(),f.loads.copy(),f.connected().copy()))
            for _ in range(round(f.cfg.control_dt/f.cfg.dt)):
                f.step()
                if f.phase in ('DONE','FAILED'):break
        if f.phase!='DONE':raise RuntimeError(f'teacher did not complete: {f.phase}: {f.failure}')
        cols=list(zip(*rows));a.output.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(a.output,sim_time=cols[0],phase=cols[1],target=cols[2],stage=cols[3],phase_time=cols[4],q=cols[5],qvel=cols[6],qref=cols[7],ctrl=cols[8],loads=cols[9],connected=cols[10],phase_names=np.array(PHASES),control_dt=f.cfg.control_dt)
        report=f.report();a.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f'REFERENCE {a.output} samples={len(rows)} duration={f.data.time:.3f}s other_peak={f.peak_other:.3f}N',flush=True)
    finally:f.close()
if __name__=='__main__':main()
