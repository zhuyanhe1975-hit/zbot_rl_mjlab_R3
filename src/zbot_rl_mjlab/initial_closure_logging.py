"""Fixed-schema console logs for closure PPO, with explicit episode windows."""
from datetime import timedelta
from statistics import mean

import torch
from rsl_rl.utils.logger import Logger


EPISODE_FIELDS=(
    'Metrics/ended_episodes','Metrics/success_fraction','Metrics/failure_fraction',
    'Metrics/wrong_break_fraction','Metrics/load_guard_fraction',
    'Metrics/headtail_collision_fraction','Metrics/headtail_penetration_peak_mm',
    'Metrics/non_target_peak_N','Metrics/episode_duration_s','Metrics/success_duration_s',
    'Penalty/failure','Penalty/elapsed_time','Reward/time','Reward/pose_nearness','Reward/episode_return',
)


def aggregate_episode_logs(records):
    """Weight episode means by completed counts, not by control-step counts."""
    result={key:None for key in EPISODE_FIELDS}
    ended=sum(r.get('Metrics/ended_episodes',0) for r in records)
    succeeded=sum(r.get('Metrics/success_episodes',0) for r in records)
    result['Metrics/ended_episodes']=ended
    if not ended:return result
    for key in EPISODE_FIELDS:
        if key in ('Metrics/ended_episodes','Metrics/non_target_peak_N','Metrics/headtail_penetration_peak_mm'):continue
        weight='Metrics/success_episodes' if key=='Metrics/success_duration_s' else 'Metrics/ended_episodes'
        denominator=succeeded if weight=='Metrics/success_episodes' else ended
        if denominator:
            result[key]=sum(r.get(key,0)*r.get(weight,0) for r in records)/denominator
    result['Metrics/non_target_peak_N']=max(r['Metrics/non_target_batch_peak_N'] for r in records
                                           if r.get('Metrics/ended_episodes',0))
    result['Metrics/headtail_penetration_peak_mm']=max(r.get('Metrics/headtail_penetration_batch_peak_mm',0)
                                                      for r in records if r.get('Metrics/ended_episodes',0))
    return result


class InitialClosureLogger(Logger):
    def __init__(self,*args,step_dt,**kwargs):
        super().__init__(*args,**kwargs)
        self.step_dt=step_dt

    def log(self,it,start_it,total_it,collect_time,learn_time,loss_dict,learning_rate,
            action_std,rnd_weight=None,**kwargs):
        if self.writer is None:return
        size=self.cfg['num_steps_per_env']*self.num_envs*self.gpu_world_size
        elapsed=collect_time+learn_time
        self.tot_timesteps+=size;self.tot_time+=elapsed
        # One device-to-host transfer for all episode diagnostics this iteration.
        rows=[r for r in self.ep_extras if r]
        records=[]
        if rows:
            keys=tuple(dict.fromkeys(key for r in rows for key in r))
            zero=torch.zeros((),device=self.device)
            values=torch.stack([torch.stack([torch.as_tensor(r.get(k,zero),device=self.device).float().mean()
                                            for k in keys]) for r in rows]).cpu().tolist()
            records=[dict(zip(keys,row)) for row in values]
        episodes=aggregate_episode_logs(records)
        fps=int(size/elapsed)
        std=float(action_std.mean())
        recent_return=mean(self.rewbuffer) if self.rewbuffer else None
        recent_length=mean(self.lenbuffer) if self.lenbuffer else None
        active_fraction=loss_dict.get('actor_active_fraction')
        scalars={**episodes,**{f'Loss/{k}':v for k,v in loss_dict.items() if k!='actor_active_fraction'},
                 'Loss/learning_rate':learning_rate,'Policy/mean_std':std,
                 'Policy/actor_active_fraction':active_fraction,
                 'Perf/total_fps':fps,'Perf/collection_time':collect_time,'Perf/learning_time':learn_time,
                 'Train/mean_reward':recent_return,'Train/mean_episode_length':recent_length}
        for key,value in scalars.items():
            if value is not None:self.writer.add_scalar(key,value,it)
        def row(label,value,fmt='.4f'):
            rendered='n/a' if value is None else format(value,fmt)
            return f'{label:>42}: {rendered}'
        lines=['#'*84,f'Learning iteration {it}/{total_it}'.center(84),
               row('Perf/total_steps',self.tot_timesteps,'.0f'),row('Perf/steps_per_second',fps,'.0f'),
               row('Perf/collection_s',collect_time,'.3f'),row('Perf/learning_s',learn_time,'.3f')]
        for key in ('value','surrogate','entropy'):
            lines.append(row(f'Loss/{key}',loss_dict.get(key)))
        lines.extend((row('Policy/mean_std',std),
                      row('Policy/actor_active_fraction',active_fraction),
                      row('Policy/learning_rate',learning_rate,'.6f'),
                      '-- Completed episodes in THIS iteration --'))
        lines.extend(row(key,episodes[key]) for key in EPISODE_FIELDS)
        lines.extend(('-- Most recent completed episodes (up to 100) --',
                      row('Recent100/episode_return',recent_return),
                      row('Recent100/episode_duration_s',recent_length*self.step_dt if recent_length is not None else None),
                      '-'*84,row('Perf/iteration_s',elapsed,'.3f'),
                      f'{"Time elapsed":>42}: {timedelta(seconds=int(self.tot_time))}',
                      f'{"ETA":>42}: {timedelta(seconds=int(self.tot_time/(it+1-start_it)*(total_it-it-1)))}'))
        print('\n'.join(lines),flush=True)
        self.ep_extras.clear()
