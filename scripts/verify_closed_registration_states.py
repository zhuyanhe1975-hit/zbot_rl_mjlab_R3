"""Audit all 64 static 180/270 registration assignments, no dynamics or motion."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from zbot_rl_mjlab.tasks.reconfiguration.closure_states import ClosedRegistrationAudit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target-angle',type=int,choices=[90,270],default=270)
    p.add_argument('--face-targets',type=int,nargs=6,choices=[90,270])
    p.add_argument('--starts',type=int,default=24)
    p.add_argument('--refine-starts',type=int,default=96)
    p.add_argument('--max-nfev',type=int,default=200)
    p.add_argument('--output',type=Path,default=None)
    args=p.parse_args()
    if args.starts<2 or args.refine_starts<args.starts:p.error('require refine-starts >= starts >= 2')
    if args.output is None:
        if args.face_targets==[270,90,270,90,270,90]:name='alternating_270_90'
        elif args.face_targets:name='mixed_'+'_'.join(map(str,args.face_targets))
        else:name=f'180_{args.target_angle}'
        args.output=Path(f'outputs/reconfiguration/discrete_closure_{name}/report.json')
    audit=ClosedRegistrationAudit(args.target_angle,args.face_targets);states={};begin=time.perf_counter()
    report={'target_angle_deg':None if args.face_targets else args.target_angle,'face_targets_deg':audit.face_angles_deg,'mode':'static_geometric_closure','face_order':[0,1,2,3,4,5],
      'face_5':'temporary head-tail brace; each bit selects the corresponding face_targets_deg entry',
      'joint_limits_ignored':True,'collision_checks':False,'dynamics_checked':False,
      'position_tolerance_m':audit.position_tolerance,'angle_tolerance_rad':audit.angle_tolerance,
      'not_found_is_not_a_proof_of_infeasibility':True,'states':states}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        report['elapsed_wall_s']=time.perf_counter()-begin
        args.output.write_text(json.dumps(report,indent=2))
    masks=sorted(range(64),key=lambda m:(m.bit_count(),m))
    for mask in masks:
        bits=[(mask>>i)&1 for i in range(6)]
        seeds=[audit.robot.qref.copy(),np.zeros(6)]
        for i in range(6):
            other=mask^(1<<i)
            entry=states.get(str(other))
            if entry and entry['status']=='solution_found':seeds.insert(0,np.array(entry['joint_angle_rad']))
        rng=np.random.default_rng(180270+mask)
        while len(seeds)<args.starts:seeds.append(rng.uniform(-np.pi,np.pi,6))
        result=audit.solve(bits,seeds,args.max_nfev)
        if result['status']=='not_found' and args.refine_starts>len(seeds):
            refinement=[rng.uniform(-np.pi,np.pi,6) for _ in range(args.refine_starts-len(seeds))]
            refined=audit.solve(bits,refinement,args.max_nfev)
            total=result['attempts']+refined['attempts']
            if refined['weighted_residual']<result['weighted_residual'] or refined['status']=='solution_found':result=refined
            result['attempts']=total
        states[str(mask)]={'mask':mask,'bits':bits,'registration_deg':[180+(angle-180)*b for angle,b in zip(audit.face_angles_deg,bits)],**result}
        save();print('STATE',mask,bits,result['status'],'p',result['position_error_m'],'angle_deg',result['angle_error_deg'],'starts',result['attempts'],flush=True)
    # Prefixes for the natural internal-face order, with either original or re-clocked brace.
    report['prefix_original_brace']=[(1<<n)-1 for n in range(6)]
    report['prefix_reclocked_brace']=[32|((1<<n)-1) for n in range(6)]
    report['prefix_all_six']=[(1<<n)-1 for n in range(7)]
    report['summary']={'solution_found':sum(s['status']=='solution_found' for s in states.values()),
                       'not_found':sum(s['status']=='not_found' for s in states.values())}
    # A path here certifies discrete states only, not continuous reconfiguration.
    reached={0:None};queue=[0]
    for current in queue:
        for i in range(6):
            if current&(1<<i):continue
            nxt=current|(1<<i)
            if nxt not in reached and states[str(nxt)]['status']=='solution_found':reached[nxt]=current;queue.append(nxt)
    path=None
    if 63 in reached:
        path=[];n=63
        while n is not None:path.append(n);n=reached[n]
        path.reverse()
    report['monotone_discrete_path_all_six']=path
    save();print('SUMMARY',report['summary'],'PATH',path,'OUTPUT',args.output,flush=True)

if __name__=='__main__':main()
