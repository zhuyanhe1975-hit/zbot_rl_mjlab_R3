"""同一进程交替测量全向PPO采样的site frame eager/CUDA Graph后端。"""

import argparse
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.utils import check_nan
from rsl_rl.utils.logger import Logger

import zbot_rl_mjlab  # noqa: F401
from zbot_rl_mjlab.tasks.universal import mdp
from zbot_rl_mjlab.teachers import TeacherEnsemble

TASK = 'Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Deploy'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--num-envs', type=int, default=8192)
    parser.add_argument('--repeats', type=int, default=6)
    parser.add_argument('--output', type=Path, default=Path('outputs/omni_graph_benchmark.json'))
    args = parser.parse_args()
    if args.num_envs < 5 or args.repeats < 2 or not torch.cuda.is_available():
        parser.error('需要CUDA、至少5环境及2次重复')
    torch.set_num_threads(1)
    configure_torch_backends()
    cfg: Any = load_env_cfg(TASK)
    cfg.scene.num_envs, cfg.seed = args.num_envs, 42
    cfg.omni_frame_backend = 'eager'
    env: Any = ManagerBasedRlEnv(cfg, device='cuda:0')
    try:
        wrapper = RslRlVecEnvWrapper(env)
        config = asdict(load_rl_cfg(TASK))
        agents, loggers = {}, {}
        for backend in ('eager', 'cuda_graph'):
            torch.manual_seed(123)
            alg: Any = MjlabOnPolicyRunner(wrapper, config.copy(), device='cuda:0').alg
            alg.teachers = TeacherEnsemble(env, enabled_teachers=alg.enabled_teachers)
            agents[backend] = alg
            logger = Logger(None, config, cfg, args.num_envs, False, 1, 0, 'cuda:0')
            logger.writer = object()  # type: ignore[assignment]
            loggers[backend] = logger
        env._student_pretraining, env._student_rl_iteration = False, 0
        command = env.command_manager.get_term('transform')
        command.set_pretraining(False)
        results: dict[str, list[float]] = {'eager': [], 'cuda_graph': []}

        def collect(algorithm, logger, observation):
            algorithm.storage.clear()
            algorithm._clear_samples()
            logger.ep_extras.clear()
            for _ in range(config['num_steps_per_env']):
                actions = algorithm.act(observation)
                observation, rewards, dones, extras = wrapper.step(actions)
                check_nan(observation, rewards, dones)
                algorithm.process_env_step(observation, rewards, dones, extras)
                logger.process_env_step(rewards, dones, extras)
            return observation

        with torch.inference_mode():
            wrapper.reset()
            saved = [getattr(command, key).clone() for key in ('heading', 'initial_heading', 'heading_initialized')]
            expected = mdp.heading(env).clone()
            for key, value in zip(('heading', 'initial_heading', 'heading_initialized'), saved, strict=True):
                getattr(command, key).copy_(value)
            cfg.omni_frame_backend = 'cuda_graph'
            torch.testing.assert_close(mdp.heading(env), expected, atol=1e-6, rtol=1e-6)
            for repeat in range(args.repeats + 1):
                order = list(agents) if repeat % 2 == 0 else list(reversed(agents))
                for backend in order:
                    cfg.omni_frame_backend = backend
                    torch.manual_seed(42)
                    env.common_step_counter = 0
                    observation = wrapper.reset()[0]
                    observation = collect(agents[backend], loggers[backend], observation)
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    collect(agents[backend], loggers[backend], observation)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    if repeat:
                        results[backend].append(elapsed)
        medians = {key: statistics.median(value) for key, value in results.items()}
        report = {'task': TASK, 'num_envs': args.num_envs, 'steps': config['num_steps_per_env'],
                  'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                  'teacher_names': list(agents['eager'].teacher_names), 'teacher_warmup': False,
                  'physics_cuda_graph': env.sim.use_cuda_graph, 'frame_parity': 'passed',
                  'collection_seconds': results, 'median_seconds': medians,
                  'graph_time_reduction': 1 - medians['cuda_graph'] / medians['eager'],
                  'scope': 'PPO collection with transform imitation, no optimizer updates; matched seeded reset plus24settle steps; no compilation/capture in timed region'}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
