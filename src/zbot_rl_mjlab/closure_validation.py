"""Independent checkpoint validation without touching training rollout state."""
import contextlib
import hashlib
import io
import json
import random
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from mjlab.rl import RslRlVecEnvWrapper

from .tasks.reconfiguration.full_batched_gpu_env import FullBatchedGpuEnv


class ClosureValidator:
    def __init__(self,training_runner,env_factory,agent_factory):
        self.training_runner=training_runner
        self.env_factory=env_factory;self.agent_factory=agent_factory
        self.env=None;self.runner=None;self.wrapped=None
        self.rows=[]
        report=Path(training_runner.logger.log_dir)/'deterministic_evaluation.json'
        if report.is_file():self.rows=json.loads(report.read_text()).get('results',[])

    def check(self,path):
        from scripts.eval_initial_closure_rsl import evaluate,select_successful_checkpoint
        start=time.perf_counter()
        py_state=random.getstate();np_state=np.random.get_state()
        device=torch.device(self.training_runner.device)
        try:
            with torch.random.fork_rng(devices=[device.index or 0]),torch.inference_mode(),contextlib.redirect_stdout(io.StringIO()):
                if self.env is None:
                    self.env=FullBatchedGpuEnv(self.env_factory(8),device=str(device))
                    self.wrapped=RslRlVecEnvWrapper(self.env)
                    self.runner=type(self.training_runner)(self.wrapped,asdict(self.agent_factory(1)),None,
                        device=str(device),reference_sha256=self.training_runner.reference_sha256,
                        evaluation_interval=0)
                row=evaluate(self.runner,self.wrapped,Path(path),round(self.env.max_episode_length_s/self.env.step_dt)+1)
        finally:
            random.setstate(py_state);np.random.set_state(np_state)
        self.rows=[old for old in self.rows if old['iteration']!=row['iteration']]+[row]
        usable=[r for r in self.rows if Path(r['checkpoint']).is_file()
                and hashlib.sha256(Path(r['checkpoint']).read_bytes()).hexdigest()==r['sha256']]
        selected=select_successful_checkpoint(usable)
        folder=Path(self.training_runner.logger.log_dir)
        if selected:
            temp=folder/'best.pt.tmp'
            shutil.copyfile(selected['checkpoint'],temp);temp.replace(folder/'best.pt')
        report={'policy':'Bounded Beta mean, no sampling','results':self.rows,'selected':selected,
                'limits':'Fixed initial state, 8 worlds; not randomized robustness evaluation.'}
        temp=folder/'deterministic_evaluation.json.tmp'
        temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(folder/'deterministic_evaluation.json')
        elapsed=time.perf_counter()-start
        self.training_runner.logger.tot_time+=elapsed
        writer=self.training_runner.logger.writer
        if writer is not None:
            writer.add_scalar('Validation/success_fraction',row['successes']/row['worlds'],row['iteration'])
            writer.add_scalar('Validation/check_seconds',elapsed,row['iteration'])
            if row['mean_success_time_s'] is not None:
                writer.add_scalar('Validation/success_duration_s',row['mean_success_time_s'],row['iteration'])
        print('VALIDATION',json.dumps({'iteration':row['iteration'],'successes':row['successes'],
            'worlds':row['worlds'],'success_time_s':row['mean_success_time_s'],'check_seconds':elapsed,
            'selected':selected['checkpoint'] if selected else None}),flush=True)
        return row

    def close(self):
        if self.env is not None:self.env.close()
        self.env=None;self.runner=None;self.wrapped=None
