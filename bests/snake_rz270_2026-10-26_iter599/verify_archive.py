"""Bounded deterministic replay; run via the project's scripts/python.sh wrapper."""
from dataclasses import asdict, make_dataclass
from pathlib import Path
import hashlib
import json
import math
import time

import mjlab
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from zbot_rl_mjlab.rz270_snake_teacher import Rz270SnakeTeacherRunner
from zbot_rl_mjlab.tasks.snake import rz270_heading as task

root=Path(__file__).resolve().parent
TASK='Mjlab-Zbot-6dof-Snake-RZ270-Teacher'
checkpoint=root/'model_599.pt'
metadata=json.loads((root/'metadata.json').read_text())
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==metadata['source_sha256']
cfg=load_env_cfg(TASK,play=True)
Playback=make_dataclass('ArchivePlayback',[('env',type(cfg))])
cfg=tyro.cli(Playback,default=Playback(env=cfg),args=json.loads((root/'playback_args.json').read_text()),config=mjlab.TYRO_FLAGS).env
Rz270SnakeTeacherRunner.prepare_playback_env_cfg(cfg,checkpoint)
cfg.scene.num_envs=8
cases=[(0.,0.),(.05,0.),(.1,0.),(.2,0.),(.1,math.pi/2),(.1,-math.pi/2),(.1,math.pi),(0.,math.pi/2)]
env=ManagerBasedRlEnv(cfg,device='cuda:0')
started=time.monotonic()
try:
    wrapped=RslRlVecEnvWrapper(env,clip_actions=1.)
    runner=Rz270SnakeTeacherRunner(wrapped,asdict(load_rl_cfg(TASK)),device='cuda:0')
    runner.load(str(checkpoint),load_cfg={'actor':True},strict=True,map_location='cuda:0')
    policy=runner.get_inference_policy(device='cuda:0')
    term=env.command_manager.get_term('transform')
    speeds=torch.tensor([c[0] for c in cases],device=env.device)
    headings=torch.tensor([c[1] for c in cases],device=env.device)
    def fixed_goals(ids):
        term.target_heading[ids]=headings[ids]
        term.requested_velocity[ids]=0.
        term.requested_velocity[ids,0]=speeds[ids]
    term._sample_velocity=fixed_goals
    env.reset()
    observations=wrapped.get_observations()
    assert observations['actor'].shape==(8,63)
    dt=env.step_dt
    steps=round(20./dt)
    burn_in=round(5./dt)
    resets=torch.zeros(8,device=env.device,dtype=torch.int64)
    errors=torch.zeros(8,4,device=env.device)
    count=0
    with torch.inference_mode():
        for step in range(steps):
            actions=policy(observations)
            assert actions.shape==(8,6) and torch.isfinite(actions).all()
            observations,reward,done,_=wrapped.step(actions)
            assert torch.isfinite(reward).all() and all(torch.isfinite(value).all() for value in observations.values())
            resets+=done.long()
            if step>=burn_in:
                measured=torch.stack(((task.head_forward_velocity(env)-speeds).abs(),task.head_lateral_speed(env),task.heading_error(env).abs(),task.site_tilt(env)),-1)
                assert torch.isfinite(measured).all()
                errors+=measured
                count+=1
    means=(errors/count).cpu().tolist()
    result={
        'checkpoint_sha256_matches_source':True,
        'strict_actor_load':'passed', 'policy_actions_and_observations_finite':True,
        'actor_obs_dim':63,'action_dim':6,'num_envs':8,'device':'cuda:0',
        'control_dt_s':dt,'duration_s':steps*dt,'steps':steps,'burn_in_s':burn_in*dt,
        'policy':'deterministic mean',
        'protocol':'Fixed speed/heading commands supplied through the task command sampler, including after resets; original model, dynamics and reset pose retained. Not a formal success-rate protocol.',
        'cases':[{'forward_speed_m_s':speed,'heading_rad':heading,'resets':int(resets[i]),
                  'mean_abs_forward_error_m_s':means[i][0],'mean_abs_lateral_speed_m_s':means[i][1],
                  'mean_abs_heading_error_rad':means[i][2],'mean_head_tilt_rad':means[i][3]}
                 for i,(speed,heading) in enumerate(cases)],
        'wall_time_s':time.monotonic()-started,
        'performance_acceptance':'not asserted; archival approval is the user visual assessment, with bounded replay diagnostics recorded separately',
    }
    (root/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))
finally:
    env.close()
