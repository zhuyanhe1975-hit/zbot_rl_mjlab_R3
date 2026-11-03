"""Independent experiment: closed180 -> sequential internal270 -> open snake."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
import torch
import zbot_rl_mjlab
from mjlab.envs import ManagerBasedRlEnv
from zbot_rl_mjlab.tasks.reconfiguration.sequential_rz270 import sequential_rz270_env_cfg,SequencePhase,EXPERIMENT_ID
from zbot_rl_mjlab.tasks.reconfiguration.disconnect.state import state
from zbot_rl_mjlab.tasks.reconfiguration.disconnect.model import ROOT_NAMES
from zbot_rl_mjlab.tasks.snake.rz270 import rz270_spec


def reference_errors(env):
    m=rz270_spec().compile();d=mujoco.MjData(m)
    modules=[env.scene[f'module_{i}'] for i in range(6)]
    free=int(np.where(m.jnt_type==mujoco.mjtJoint.mjJNT_FREE)[0][0]);adr=m.jnt_qposadr[free]
    q=modules[0].data.root_link_pose_w[0].cpu().numpy();d.qpos[adr:adr+7]=q
    for i,module in enumerate(modules):d.qpos[m.joint(f'joint{i+1}').qposadr]=module.data.joint_pos[0].cpu().numpy()
    mujoco.mj_forward(m,d);position=[];angle=[]
    for module,name in zip(modules,ROOT_NAMES):
        b=m.body(name).id
        p=module.data.root_link_pos_w[0].cpu().numpy();q=module.data.root_link_quat_w[0].cpu().numpy()
        position.append(float(np.linalg.norm(p-d.xpos[b])))
        angle.append(float(2*np.arccos(np.clip(abs(q@d.xquat[b]),0,1))))
    return {'root_position_error_m':position,'root_angle_error_rad':angle}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--viewer',action='store_true');p.add_argument('--headless',dest='viewer',action='store_false')
    p.set_defaults(viewer=False)
    p.add_argument('--output',type=Path,default=Path('outputs/reconfiguration/sequential_rz270/latest.json'))
    p.add_argument('--max-time',type=float,default=180.)
    p.add_argument('--order',type=int,nargs=5,default=[0,1,2,3,4])
    p.add_argument('--acceleration',type=float,default=8.)
    args=p.parse_args()
    if sorted(args.order)!=list(range(5)):p.error('--order must contain 0 1 2 3 4 once each')
    env=ManagerBasedRlEnv(sequential_rz270_env_cfg(order=args.order,acceleration=args.acceleration),device='cuda:0')
    try:
        with torch.inference_mode():
            env.reset();a=env.action_manager.get_term('joint_torque');c=a.distance_controller;s=state(env)
            zero=torch.zeros(1,6,device=env.device)
            previous=None
            def sample(t):
                q=torch.cat([m.data.joint_pos for m in a.modules],-1)
                return {'t':round(t,5),'sequence_index':c.index,'target_connector':env.cfg.target_connector,
                        'phase':SequencePhase(c.phase.item()).name,'phase_elapsed_s':c.phase_time.item(),
                        'gap_m':s.geometry()['distance'].item(),'orientation_error_deg':float(torch.rad2deg(c.angle_error)[0]),
                        'joint_angle_rad':q[0].tolist(),'actual_torque_Nm':a.torque[0].tolist(),
                        'loads_N':(s.utilization[0]*env.cfg.break_force).tolist(),'other_margin':s.other_margin().item(),
                        'connected':s.connected()[0].tolist(),'registered_270':env.sim.data.eq_active[0,s.variant_ids].bool().tolist(),
                        'failure_code':c.failure_code.item(),'safety_violation':s.safety_margin_violation.item(),'wrong_break':s.wrong_break.item(),
                        'torque_failure':a.torque_failure.item(),'peak_other_load_N':float(s.peak_utilization[0,s.other_ids].max()*env.cfg.break_force),'requested_torque_Nm':a.requested_torque[0].tolist(),'loading':a.loading.last_report}
            if args.viewer:
                from mjlab.viewer import NativeMujocoViewer
                def policy(obs):
                    nonlocal previous
                    key=(c.index,c.phase.item())
                    if key!=previous:print(json.dumps(sample(env._sim_step_counter*env.physics_dt)),flush=True);previous=key
                    return zero
                NativeMujocoViewer(env,policy).run();return 0
            trace=[]
            for k in range(int(args.max_time/env.step_dt)):
                env.step(zero);key=(c.index,c.phase.item())
                if key!=previous or k%25==0:
                    row=sample((k+1)*env.step_dt);trace.append(row);print(json.dumps(row),flush=True);previous=key
                if c.phase.item() in (SequencePhase.DONE,SequencePhase.FAILED):break
            terminal=sample((k+1)*env.step_dt)
            report={'experiment':EXPERIMENT_ID,'success':c.phase.item()==SequencePhase.DONE,'terminal':terminal,
                    'completed_connectors':c.history,'reference_rz270':reference_errors(env),'trace':trace}
            args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2))
            print('REPORT',args.output,report['success'],terminal['phase'],flush=True)
            return 0 if report['success'] else 1
    finally:env.close()

if __name__=='__main__':raise SystemExit(main())
