"""Run one standalone load/open/separate/rotate/approach/reweld cycle; no PPO."""
import argparse
import json
from pathlib import Path
import torch
import zbot_rl_mjlab
from mjlab.envs import ManagerBasedRlEnv
from zbot_rl_mjlab.tasks.reconfiguration.state_machine import state_machine_env_cfg,Phase
from zbot_rl_mjlab.tasks.reconfiguration.disconnect.state import state


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--num-envs',type=int,default=1)
    p.add_argument('--viewer',action='store_true')
    p.add_argument('--headless',dest='viewer',action='store_false')
    p.set_defaults(viewer=False)
    p.add_argument('--output',type=Path,default=Path('outputs/reconfiguration/state_machine/latest.json'))
    args=p.parse_args()
    if args.num_envs<1:p.error('--num-envs must be positive')
    cfg=state_machine_env_cfg(args.num_envs);cfg.auto_reset=args.viewer
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
        with torch.inference_mode():
            env.reset();ctrl=env.action_manager.get_term('joint_torque');c=ctrl.distance_controller;s=state(env)
            action=torch.zeros(args.num_envs,6,device=env.device)
            if args.viewer:
                from mjlab.viewer import NativeMujocoViewer
                last_phase=None
                def policy(obs):
                    nonlocal last_phase
                    names=[Phase(x).name for x in c.phase.tolist()]
                    if names!=last_phase:
                        print('FSM',names,flush=True);last_phase=names
                    return action
                NativeMujocoViewer(env,policy).run()
                return
            trace=[];previous=None
            completed=torch.zeros(args.num_envs,dtype=torch.bool,device=env.device)
            terminal_phase=c.phase.clone()
            terminal_durations=c.phase_durations.clone()
            terminal_records=[None]*args.num_envs
            for k in range(int(env.cfg.episode_length_s/env.step_dt)):
                _,_,done,timeout,_=env.step(action)
                phase=c.phase.tolist()
                if phase!=previous or k%25==0 or done.any():
                    row={'t':round((k+1)*env.step_dt,4),'first_episode_active':(~completed).tolist(),'phase':[Phase(x).name for x in phase],
                         'gap_m':s.geometry()['distance'].tolist(),'angle_error_deg':torch.rad2deg(c.angle_error).tolist(),
                         'joint_angle_rad':torch.cat([m.data.joint_pos for m in ctrl.modules],-1).tolist(),
                         'target_error_rad':(ctrl.position_target-torch.cat([m.data.joint_pos for m in ctrl.modules],-1)).abs().amax(-1).tolist(),
                         'command_speed_rad_s':ctrl.velocity_command.abs().amax(-1).tolist(),'position_error_m':c.position_error.tolist(),
                         'other_margin':s.other_margin().tolist(),'welds':s.connected().tolist(),
                         'new_weld':env.sim.data.eq_active[:,s.new_eq].bool().tolist(),
                         'relative_speed_m_s':s.geometry()['dv'].norm(dim=-1).tolist(),'relative_angular_speed_rad_s':s.geometry()['dw'].norm(dim=-1).tolist(),'capture_time_s':c.capture_time.tolist(),
                         'loads_N':(s.utilization*env.cfg.break_force).tolist(),'failure_code':c.failure_code.tolist(),'phase_elapsed_s':c.phase_time.tolist()}
                    trace.append(row);print(json.dumps(row),flush=True);previous=phase
                ended=(done|timeout)&~completed
                for i in ended.nonzero(as_tuple=True)[0].tolist():
                    terminal_records[i]={key:(value[i] if isinstance(value,list) else value) for key,value in row.items()}
                terminal_phase.copy_(torch.where(ended,c.phase,terminal_phase))
                terminal_durations.copy_(torch.where(ended[:,None],c.phase_durations,terminal_durations))
                completed |= done|timeout
                if completed.all():break
                reset_ids=(done|timeout).nonzero(as_tuple=True)[0]
                if reset_ids.numel():env.reset(env_ids=reset_ids)
            args.output.parent.mkdir(parents=True,exist_ok=True)
            terminal_phase=torch.where(completed,terminal_phase,c.phase)
            terminal_durations=torch.where(completed[:,None],terminal_durations,c.phase_durations)
            configuration={name:getattr(env.cfg,name) for name in (
                'motion_duration_s','rotation_angle','rotation_speed','pose_linear_speed','approach_speed',
                'capture_distance','capture_angle','capture_speed','capture_angular_speed','capture_hold_s',
                'weld_ramp_s','settle_s','separation_stiffness','separation_damping',
                'separation_cartesian_speed','separation_max_target_speed','separation_joint_acceleration',
                'separation_target_error_limit','torque_limit','continuous_torque_limit','minimum_other_margin')}
            report={'mode':'standalone_reconfiguration_fsm_v1','actuator':'mjlab.BuiltinPositionActuatorCfg',
                    'configuration':configuration,'terminal':terminal_records,'success':(terminal_phase==Phase.DONE).tolist(),'phase':[Phase(x).name for x in terminal_phase.tolist()],
                    'elapsed_s':(k+1)*env.step_dt,'phase_durations_s':{phase.name:terminal_durations[:,phase].tolist() for phase in Phase},'trace':trace}
            args.output.write_text(json.dumps(report,indent=2));print('REPORT',args.output,report['phase'],flush=True)
            return 0 if all(report['success']) else 1
    finally:env.close()


if __name__=='__main__':raise SystemExit(main())
