from dataclasses import asdict
from pathlib import Path
import json
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
import zbot_rl_mjlab
from zbot_rl_mjlab.experiments.snake import SnakeTeacherRunner
from zbot_rl_mjlab.tasks.snake import mdp
folder=Path('bests/snake_world_x_2026-10-19_iter1499')
path=folder/'model_1499.pt'
cfg=load_env_cfg('Mjlab-Zbot-6dof-Snake-Omni-Teacher',play=True)
cfg.scene.num_envs=8
cfg.auto_reset=False
cfg.seed=42
recipe=json.loads((folder/'metadata.json').read_text())['recipe']
cfg.actions['joint_pos'].target_limit=recipe['joint_target_limit']
cfg.actions['joint_pos'].max_target_speed=recipe['max_target_speed']
cfg.motion_penalty_min_scale=recipe['motion_penalty_min_scale']
cfg.motion_penalty_speed_gain=recipe['motion_penalty_speed_gain']
raw=ManagerBasedRlEnv(cfg,device='cuda:0')
try:
 agent=load_rl_cfg('Mjlab-Zbot-6dof-Snake-Omni-Teacher')
 wrapped=RslRlVecEnvWrapper(raw,clip_actions=agent.clip_actions)
 train_cfg=asdict(agent)
 SnakeTeacherRunner.prepare_playback_cfg(train_cfg,str(path))
 runner=SnakeTeacherRunner(wrapped,train_cfg,device='cuda:0')
 runner.load(str(path),load_cfg={'actor':True},strict=True,map_location='cuda:0')
 policy=runner.get_inference_policy(device='cuda:0')
 with torch.no_grad():
  raw.reset()
  obs=wrapped.get_observations()
  def position():
   mass=mdp.body_masses(raw)
   return (raw.scene['robot'].data.body_com_pos_w[:,:,:2]*mass[...,None]).sum(-2)/mass.sum(-1,keepdim=True)
  positions=position().clone()
  failed=torch.zeros(8,dtype=torch.bool,device='cuda:0')
  speeds=[];laterals=[];failure_step=torch.full((8,),-1,device='cuda:0',dtype=torch.long)
  for i in range(round(30/raw.step_dt)):
   obs,_,done,_=wrapped.step(policy(obs))
   terminated=raw.termination_manager.terminated
   newly=terminated & ~failed
   failure_step[newly]=i+1
   failed |= terminated
   speeds.append(mdp.world_com_velocity(raw)[:,0].clone())
   laterals.append(mdp.world_com_velocity(raw)[:,1].abs().clone())
  result={'strict_actor_load':True,'device':'cuda:0','num_envs':8,'seconds':30.,'initial_heading':'play默认世界+X，初始关节全零','stochastic':False,'auto_reset':False,'failures':int(failed.sum()),'first_failure_step':failure_step.tolist(),'mean_world_x_speed_m_s':torch.stack(speeds).mean().item(),'mean_abs_world_y_speed_m_s':torch.stack(laterals).mean().item(),'per_env_displacement_xy_m':(position()-positions).tolist(),'actor_dim':obs['actor'].shape[-1],'critic_dim':obs['critic'].shape[-1],'limits':'固定默认初态仿真回放；未验证全随机朝向、扰动或实机。'}
 folder.joinpath('verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print('VERIFICATION_RESULT '+json.dumps(result,ensure_ascii=False))
finally:raw.close()
