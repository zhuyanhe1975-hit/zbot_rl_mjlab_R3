"""Fixed-shape GPU-mask reset prototype for the full batched task.

This helper is task-specific and must be constructed immediately after a full
reset. It avoids dynamic GPU-to-host index extraction and preserves all other
worlds. It does not replace MJLab's reset machinery for arbitrary tasks.
"""
import torch
import warp as wp

class FullBatchedGpuReset:
    def __init__(self,env):
        if not env.device.startswith('cuda'):raise ValueError('GPU reset requires CUDA')
        if env.cfg.commands or env.cfg.curriculum or env.cfg.recorders or env.scene.sensors:
            raise ValueError('GPU reset supports only the full task without commands/curricula/recorders')
        if set(env.cfg.events)!={'reset_scene','reset_connectors','reset_headtail_open'}:
            raise ValueError('GPU reset requires the full task reset events only')
        if any(env.observation_manager._group_obs_term_history_buffer.values()):
            raise ValueError('GPU reset does not support observation history')
        if env.reward_manager._class_term_cfgs or env.metrics_manager._class_term_cfgs:
            raise ValueError('GPU reset does not support stateful reward or metric terms')
        if not env.sim.use_cuda_graph or env.sim.reset_graph is None:
            raise ValueError('GPU reset requires the MuJoCo Warp reset graph')
        self.env=env;self.buffers=[];s=env._disconnect_state;a=env.action_manager.get_term('joint_torque')
        def remember(value):
            if hasattr(value,'_tensor'):value=value._tensor
            if not isinstance(value,torch.Tensor) or not value.ndim or value.shape[0]!=env.num_envs:
                raise ValueError('unexpected per-world reset buffer')
            self.buffers.append((value,value.clone()))
        # Warp reset initializes generic physics; the scene's calibrated root
        # poses and weld states must be restored from its real reset snapshot.
        for value in (env.sim.data.qpos,env.sim.data.qvel,env.sim.data.eq_active,
                      env.sim.model.eq_solref,env.sim.model.actuator_gainprm,env.sim.model.actuator_biasprm):remember(value)
        for name in ('force','moment','utilization','peak_utilization','break_utilization','peak_target_tilt_load',
                     'max_target_gap','gap_progress','separation_progress_delta','wait_count','waiting','succeeded',
                     'just_succeeded','wrong_break','safety_margin_violation','invalid_state','index','contact_violation',
                     'has_sample','max_module_depth','max_ground_depth','max_headtail_depth','headtail_penetration','max_other_load','loads_at_break','phase_contacts'):
            remember(getattr(s,name))
        for name in ('phase','phase_time','phase_elapsed','stage','target','capture_time','failure_code','active_load_phase'):
            remember(getattr(s.fsm,name))
        for name in ('_raw','residual','feedforward','requested_torque','overload_count','torque_failure','peak_overload_cost',
                     'pd_active','position_target','velocity_command','jacobian_gain','policy','nominal','nominal_vel',
                     'loading_qhold','teacher_headtail_ctrl','teacher_velocity','motion_velocity','curl_residual_velocity','curl_residual_offset','pose_potential_before','pose_progress_active','policy_torque','last_phase','progress_before'):
            remember(getattr(a,name))
        for name in ('torque','predicted_loads','feasible'):remember(getattr(a.loading,name))
        for name in ('_action','_prev_action','_prev_prev_action'):remember(getattr(env.action_manager,name))
        for module in s.modules:
            for name in ('joint_pos_target','joint_vel_target','joint_effort_target'):
                remember(getattr(module.data,name))
        remember(env.episode_length_buf);remember(env._manual_reset_pending)
        for value in env.reward_manager._episode_sums.values():remember(value)
        mm=env.metrics_manager
        for value in mm._episode_sums.values():remember(value)
        for value in mm._episode_max.values():remember(value)
        for value in mm._substep_accum:remember(value)
        for value in mm._substep_episode_sums:remember(value)
        for value in mm._substep_episode_max:
            if value is not None:remember(value)
        remember(mm._step_count)
    def reset(self,mask):
        if mask.dtype!=torch.bool or mask.shape!=(self.env.num_envs,):
            raise ValueError('reset mask must be a per-world CUDA boolean tensor')
        if mask.device!=torch.device(self.env.device):raise ValueError('reset mask must reside on simulator device')
        self.env.sim._reset_mask._tensor.copy_(mask)
        with wp.ScopedDevice(self.env.sim.wp_device):wp.capture_launch(self.env.sim.reset_graph)
        for buffer,initial in self.buffers:
            shape=(mask.shape[0],)+(1,)*(buffer.ndim-1)
            torch.where(mask.reshape(shape),initial,buffer,out=buffer)
