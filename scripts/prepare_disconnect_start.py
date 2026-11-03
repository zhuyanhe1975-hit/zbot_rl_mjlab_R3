"""Build a stationary, zero-gap IK loop for the independent torque task."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

import zbot_rl_mjlab
from zbot_rl_mjlab.tasks.reconfiguration.planning import ClosurePlanner,load_plan,DEFAULT_PLAN
from zbot_rl_mjlab.tasks.reconfiguration.disconnect.model import ROOT_NAMES,SNAPSHOT_PATH

DEFAULT_CHECKPOINT=Path('logs/rsl_rl_reconfiguration_plan_imitation/zbot_reconfiguration_head_tail_docking_plan_guided/2026-10-26_20-17-04/model_99.pt')


def prepare(checkpoint,output):
    plan=load_plan();solution=plan['solutions'][0];quarter=solution['quarter']
    planner=ClosurePlanner()
    result=least_squares(lambda q:planner.error(q,quarter,gap=0.),
                         solution['waypoints'][-1],bounds=planner.bounds.T,
                         max_nfev=1000,ftol=1e-13,xtol=1e-13,gtol=1e-13)
    if not result.success or np.linalg.norm(planner.error(result.x,quarter,0.))>1e-8:
        raise RuntimeError('Zero-gap loop IK failed')
    # Retain the established lying-on-ground orientation, but derive every
    # module pose from the same original-chain FK at the solved joint angles.
    previous=json.loads(SNAPSHOT_PATH.read_text())
    planner.data.qpos[:7]=previous['modules'][0]['root_state'][:7]
    data=planner.forward(result.x);model=planner.model
    # Seat the actual mesh surfaces on the plane, without initial penetration.
    lowest=float('inf')
    for geom in range(model.ngeom):
        if model.geom_type[geom]!=mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh=model.geom_dataid[geom];start=model.mesh_vertadr[mesh];count=model.mesh_vertnum[mesh]
        vertices=model.mesh_vert[start:start+count]
        world=vertices@data.geom_xmat[geom].reshape(3,3).T+data.geom_xpos[geom]
        lowest=min(lowest,float(world[:,2].min()))
    if not np.isfinite(lowest):
        raise RuntimeError('No robot mesh surfaces found')
    planner.data.qpos[2]+=.0001-lowest
    data=planner.forward(result.x)
    modules=[]
    for i,name in enumerate(ROOT_NAMES):
        body=model.body(name).id
        modules.append({'root_name':name,
                        'root_state':np.r_[data.xpos[body],data.xquat[body],np.zeros(6)].tolist(),
                        'joint_position':float(result.x[i]),'joint_velocity':0.,'holding_torque':0.})
    snapshot={'version':1,'initialization':'stationary_zero_gap_ik',
              'source_checkpoint':str(checkpoint.resolve()),
              'source_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'source_task':'Mjlab-Zbot-6dof-Reconfigure-Head-Tail-Docking-Plan-Guided',
              'plan_sha256':hashlib.sha256(DEFAULT_PLAN.read_bytes()).hexdigest(),
              'tail_registration_quarter':quarter,
              'source_gap_m':float(np.linalg.norm(data.site_xpos[planner.tail]-data.site_xpos[planner.head])),
              'source_normal_angle_rad':float(np.arccos(np.clip(np.dot(data.site_xmat[planner.head].reshape(3,3)[:,2],data.site_xmat[planner.tail].reshape(3,3)[:,2]),-1,1))),
              'modules':modules}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(snapshot,indent=2)+'\n')
    print('IK_START',json.dumps({'gap_m':snapshot['source_gap_m'],
                                'joint_angles':result.x.tolist(),'initial_motor_torque':[0.]*6}))
    return snapshot


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output',type=Path,default=SNAPSHOT_PATH)
    args=parser.parse_args()
    prepare(args.checkpoint,args.output)
