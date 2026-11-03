"""GPU-mask auto-reset prototype for the full reference student task."""
import torch
from mjlab.envs import ManagerBasedRlEnv
from .full_batched_gpu_reset import FullBatchedGpuReset

def record_peak_loads(values,peak_loads,invalid_world_steps,break_force):
    """Accumulate finite face peaks without hiding invalid-world diagnostics."""
    finite=torch.isfinite(values)
    invalid_world_steps.add_((~finite).any(dim=-1).sum())
    peak=torch.where(finite,values,0.).amax(dim=0)*break_force
    peak_loads.copy_(torch.maximum(peak_loads,peak))

class FullBatchedGpuEnv(ManagerBasedRlEnv):
    """Task-specific fixed-shape GPU reset; no commands, sensors or curriculum.

    Episode term counts stay in the termination manager. Per-episode manager
    summaries are reset on GPU; this prototype does not produce the standard
    MJLab per-episode summary dictionary.
    """
    def __init__(self,cfg,device):
        if not cfg.auto_reset:raise ValueError('GPU mask environment requires auto_reset')
        super().__init__(cfg,device=device);self._gpu_mask_reset=None
        self.interval_peak_loads=torch.zeros(6,device=self.device)
        self.interval_invalid_load_world_steps=torch.zeros((),device=self.device,dtype=torch.long)
        self.interval_peak_other=torch.zeros((),device=self.device)
    def reset(self,*,seed=None,env_ids=None,options=None):
        result=super().reset(seed=seed,env_ids=env_ids,options=options)
        if self._gpu_mask_reset is None:
            if env_ids is not None:raise ValueError('first reset must initialize every world')
            self._gpu_mask_reset=FullBatchedGpuReset(self)
        return result
    def step(self,action):
        if self._gpu_mask_reset is None:raise RuntimeError('call reset before stepping')
        self.extras['log']={};self.action_manager.process_action(action.to(self.device))
        for _ in range(self.cfg.decimation):
            self._sim_step_counter+=1;self.action_manager.apply_action();self.scene.write_data_to_sim()
            self.sim.step();self.scene.update(dt=self.physics_dt);self.metrics_manager.compute_substep()
        self.episode_length_buf+=1;self.common_step_counter+=1
        self.reset_buf=self.termination_manager.compute();self.reset_terminated=self.termination_manager.terminated
        self.reset_time_outs=self.termination_manager.time_outs
        self.reward_buf=self.reward_manager.compute(dt=self.step_dt);self.metrics_manager.compute()
        # Capture substep peaks before the reset mask clears failed worlds.
        record_peak_loads(self._disconnect_state.peak_utilization,self.interval_peak_loads,
                          self.interval_invalid_load_world_steps,self.cfg.break_force)
        self.interval_peak_other.copy_(torch.maximum(self.interval_peak_other,self._disconnect_state.max_other_load.max()))
        if getattr(self.cfg,'full_initial_closure_diagnostics',False) and bool(self.reset_buf.any()):
            ended=self.reset_buf;success=self.termination_manager.get_term('success')&ended
            count=ended.sum().clamp_min(1);s=self._disconnect_state
            info={'Metrics/ended_episodes':ended.sum(),
                  'Metrics/success_episodes':success.sum(),
                  'Metrics/success_fraction':success.sum().float()/count,
                  'Metrics/failure_fraction':(ended&~success).sum().float()/count,
                  'Metrics/wrong_break_fraction':(ended&s.wrong_break).sum().float()/count,
                  'Metrics/load_guard_fraction':(ended&s.safety_margin_violation).sum().float()/count,
                  'Metrics/headtail_collision_fraction':(ended&s.headtail_penetration).sum().float()/count,
                  'Metrics/headtail_penetration_batch_peak_mm':torch.where(ended,s.max_headtail_depth,0.).max()*1000.,
                  'Metrics/non_target_batch_peak_N':torch.where(ended,s.max_other_load,0.).max(),
                  'Metrics/episode_duration_s':(self.episode_length_buf*ended).sum()*self.step_dt/count}
            if bool(success.any()):
                info['Metrics/success_duration_s']=(self.episode_length_buf*success).sum()*self.step_dt/success.sum()
            episode_return=torch.zeros((),device=self.device)
            for name,total in self.reward_manager._episode_sums.items():
                value=(total*ended).sum()/count
                prefix='Penalty' if self.cfg.rewards[name].weight<0 else 'Reward'
                label=name.removeprefix('penalty_') if prefix=='Penalty' else name
                info[f'{prefix}/{label}']=value
                episode_return+=value
            info['Reward/episode_return']=episode_return
            self.extras['log']=info
        # This task has no interval/step events, commands, sensors, curricula or
        # history terms. Reset graph consumes the fixed-shape per-world mask.
        self._gpu_mask_reset.reset(self.reset_buf)
        self.scene.write_data_to_sim();self.sim.forward();self.command_manager.compute(dt=self.step_dt)
        self.sim.sense();self.obs_buf=self.observation_manager.compute(update_history=True)
        return self.obs_buf,self.reward_buf,self.reset_terminated,self.reset_time_outs,self.extras
