"""Deterministic mixed-clock dynamics rebuilt from the seven-module reference."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import zbot_rl_mjlab
from mjlab.envs import ManagerBasedRlEnv
from zbot_rl_mjlab.tasks.reconfiguration.reference_dynamic import reference_env_cfg,ReferencePhase,EXPERIMENT_ID
from zbot_rl_mjlab.tasks.reconfiguration.disconnect.state import state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--viewer',action='store_true');p.add_argument('--headless',dest='viewer',action='store_false');p.set_defaults(viewer=False)
    p.add_argument('--max-time',type=float,default=300.)
    p.add_argument('--speed-deg-s',type=float,default=45.);p.add_argument('--accel-deg-s2',type=float,default=180.)
    p.add_argument('--stiffness',type=float,default=400.);p.add_argument('--damping',type=float,default=4.)
    p.add_argument('--output',type=Path,default=Path('outputs/reconfiguration/mixed_closed_reference/latest.json'))
    args=p.parse_args()
    if min(args.max_time,args.speed_deg_s,args.accel_deg_s2,args.stiffness,args.damping)<=0:p.error('positive parameters required')
    cfg=reference_env_cfg();cfg.reference_speed_deg_s=args.speed_deg_s;cfg.reference_acceleration_deg_s2=args.accel_deg_s2
    cfg.separation_stiffness=args.stiffness;cfg.separation_damping=args.damping
    e=ManagerBasedRlEnv(cfg,device='cuda:0');start=time.monotonic()
    try:
        with torch.inference_mode():
            e.reset();a=e.action_manager.get_term('joint_torque');c=a.distance_controller;s=state(e)
            action=torch.zeros(1,6,device=e.device);trace=[];previous=None;next_sample=0.
            def sample():
                eq=s.constraint_ids()[0];active=s.connected()[0]
                ids=eq.cpu().numpy();site1=torch.as_tensor(e.sim.mj_model.eq_obj1id[ids].copy(),device=e.device)
                site2=torch.as_tensor(e.sim.mj_model.eq_obj2id[ids].copy(),device=e.device)
                position=(e.sim.data.site_xpos[0,site1]-e.sim.data.site_xpos[0,site2]).norm(dim=-1)
                R1=e.sim.data.site_xmat[0,site1].reshape(-1,3,3);R2=e.sim.data.site_xmat[0,site2].reshape(-1,3,3)
                relative=R1.transpose(-1,-2)@R2;angle=torch.acos(((relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1))
                return {'time_s':e._sim_step_counter*e.physics_dt,'stage':int(c.index[0]),'connection':int(s.target_indices()[0])+1,
                        'phase':ReferencePhase(c.phase.item()).name,'phase_time_s':float(c.phase_time[0]),
                        'planned_duration_s':None if c.curve is None else c.curve['duration'],
                        'gap_m':float(s.geometry()['distance'][0]),'goal_angle_error_deg':float(torch.rad2deg(c.angle_error[0])),
                        'q_rad':torch.cat([m.data.joint_pos for m in a.modules],-1)[0].tolist(),
                        'reference_q_rad':c.reference_q[0].tolist(),'reference_velocity_rad_s':c.reference_velocity[0].tolist(),
                        'torque_Nm':a.torque[0].tolist(),'requested_torque_Nm':a.requested_torque[0].tolist(),
                        'loads_N':(s.utilization[0]*cfg.break_force).tolist(),'loads_at_break_N':s.loads_at_break[0].tolist(),
                        'peak_other_load_N':s.max_other_load.item(),'peak_loads_N':s.max_loads_N[0].tolist(),
                        'native_stiffness':cfg.separation_stiffness,'native_damping':cfg.separation_damping,'connected':active.tolist(),
                        'new_registrations':e.sim.data.eq_active[0,s.variant_ids].bool().tolist(),
                        'max_active_weld_position_error_m':float(torch.where(active,position,0.).max()),
                        'max_active_weld_angle_error_deg':float(torch.rad2deg(torch.where(active,angle,0.).max())),
                        'failure_code':int(c.failure_code[0]),'failure_message':c.failure_message,
                        'safety_violation':bool(s.safety_margin_violation[0]),'torque_failure':bool(a.torque_failure[0]),
                        'contact_violation':bool(s.contact_violation[0]),'max_module_penetration_m':float(s.max_module_depth[0]),
                        'max_ground_penetration_m':float(s.max_ground_depth[0]),'support':s.last_support}
            def record(force=False):
                nonlocal previous,next_sample
                key=(int(c.index[0]),c.phase.item());t=e._sim_step_counter*e.physics_dt
                if force or key!=previous or t>=next_sample:
                    row=sample();trace.append(row);print(json.dumps(row),flush=True);previous=key;next_sample=t+.5
            if args.viewer:
                from mjlab.viewer import NativeMujocoViewer
                def policy(obs):record();return action
                class FaultPausingViewer(NativeMujocoViewer):
                    def _execute_step(self):
                        if c.phase.item() in (ReferencePhase.DONE,ReferencePhase.FAILED):
                            record(force=True);self.pause();return False
                        ok=super()._execute_step()
                        if c.phase.item() in (ReferencePhase.DONE,ReferencePhase.FAILED):
                            record(force=True);self.pause();return False
                        return ok
                FaultPausingViewer(e,policy).run()
            else:
                for _ in range(max(1,round(args.max_time/e.step_dt))):
                    e.step(action);record()
                    if c.phase.item() in (ReferencePhase.DONE,ReferencePhase.FAILED):break
            terminal=sample();success=c.phase.item()==ReferencePhase.DONE and int(c.index[0])==4
            substeps=[{'time_s':step*e.physics_dt,'phase':ReferencePhase(phase).name,
                'values':values.cpu().tolist()} for step,phase,values in s.substep_trace]
            report={'substep_diagnostics':{'dt_s':e.physics_dt,
                    'layout':'loads_N[6],connected[6],force_N[18],moment_Nm[18],joint_velocity[6],qvel[model.nv],actual_torque[6],requested_torque[6]',
                    'samples':substeps},'experiment':EXPERIMENT_ID,'success':success,'terminal':terminal,'trace':trace,'events':c.events,
                    'completed_connections':c.history,'planning':c.plan_diagnostics,'wall_time_s':time.monotonic()-start,
                    'configuration':{'reference_cuda_graphs':cfg.reference_cuda_graphs,'fsm_control_dt_s':e.step_dt if cfg.reference_cuda_graphs else e.physics_dt,'physics_dt_s':e.physics_dt,'control_dt_s':e.step_dt,'gap_goal_m':cfg.separation_distance,'reference_speed_deg_s':cfg.reference_speed_deg_s,
                        'reference_acceleration_deg_s2':cfg.reference_acceleration_deg_s2,'motion_stiffness':a.motion_stiffness,
                        'motion_damping':a.motion_damping ,'motion_controller':'fixed MJLab native position PD','break_force_N':cfg.break_force,'other_stop_load_N':cfg.break_force*(1-cfg.minimum_other_margin)},
                    'contact_rule':'Mating halves exempt while welded, retracting and approaching; enabled in separated rotation. Other contacts remain active.',
                    'scope':'Real dynamics; no learning and no reference root/qpos injection. Closed target, no final brace release.'}
            args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
            print('REPORT',args.output,'success=',success,'phase=',terminal['phase'],'reason=',terminal['failure_message'],flush=True)
            return 0 if args.viewer or success else 1
    finally:e.close()

if __name__=='__main__':raise SystemExit(main())
