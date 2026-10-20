"""在同一仿真中交替测量教师查询及完整采样；不更新或保存训练策略。

./scripts/python.sh scripts/benchmark_collection.py --baseline-dir /tmp/zbot_collection_before
"""

import argparse
import importlib.util
import json
import os
import signal
import statistics
import sys
import time
from contextlib import contextmanager
from copy import deepcopy
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
from zbot_rl_mjlab import distillation, teachers

TASK = 'Mjlab-Zbot-6dof-Universal-Snake-Leg-Deploy'


def baseline_modules(directory):
    modules = {}
    for name in ('teachers', 'distillation'):
        identifier = f'zbot_rl_mjlab._collection_baseline_{name}'
        spec = importlib.util.spec_from_file_location(identifier, directory / f'{name}.py')
        if spec is None or spec.loader is None:
            raise ValueError(f'无法加载基线 {name}')
        module = importlib.util.module_from_spec(spec)
        sys.modules[identifier] = module
        spec.loader.exec_module(module)
        modules[name] = module
    modules['teachers']._ROOT = Path(__file__).resolve().parents[1]
    modules['teachers'].DEFAULT_CHECKPOINTS = teachers.DEFAULT_CHECKPOINTS.copy()
    return modules['teachers'], modules['distillation']


@contextmanager
def pause_training(pid):
    """仅显式指定时临时暂停训练进程，所有正常/异常退出均恢复。"""
    if pid is not None:
        command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ')
        if b'train.py ' not in command or pid == os.getpid():
            raise ValueError('pause-pid 必须是训练进程')
        os.kill(pid, signal.SIGSTOP)
    try:
        yield
    finally:
        if pid is not None:
            os.kill(pid, signal.SIGCONT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-dir', type=Path)
    parser.add_argument('--num-envs', type=int, default=8192)
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--query-repeats', type=int, default=24)
    parser.add_argument('--pause-pid', type=int)
    parser.add_argument('--output', type=Path, default=Path('outputs/collection_benchmark.json'))
    args = parser.parse_args()
    if args.num_envs < 5 or args.repeats < 2 or args.query_repeats < 1:
        parser.error('需要至少5个环境、2次采样重复和1次查询重复')
    if not torch.cuda.is_available():
        parser.error('此基准需要 CUDA')
    torch.set_num_threads(1)
    device = 'cuda:0'
    configure_torch_backends()
    variants = {'after': (teachers, distillation)}
    if args.baseline_dir:
        variants = {'before': baseline_modules(args.baseline_dir)} | variants
    cfg = load_env_cfg(TASK)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 42
    env: Any = ManagerBasedRlEnv(cfg, device=device)
    results: dict[str, Any] = {}
    try:
        wrapper = RslRlVecEnvWrapper(env)
        command = env.command_manager.get_term('transform')
        command.set_pretraining(True)
        env._student_pretraining = True
        env._student_rl_iteration = 0
        obs = wrapper.reset()[0]
        agents = {}
        loggers = {}
        config = asdict(load_rl_cfg(TASK))
        config['multi_gpu'] = None
        for label, (teacher_module, algorithm_module) in variants.items():
            agent_cfg = deepcopy(config)
            agent_cfg['algorithm']['class_name'] = f'{algorithm_module.__name__}:MultiTeacherPPO'
            torch.manual_seed(123)
            alg: Any = MjlabOnPolicyRunner(wrapper, agent_cfg, device=device).alg
            alg.teachers = teacher_module.TeacherEnsemble(env, checkpoints={'snake': teachers.SNAKE_CHECKPOINT})
            agents[label] = alg
            logger = Logger(None, agent_cfg, cfg, args.num_envs, False, 1, 0, device)
            logger.writer = object()  # type: ignore[assignment]
            loggers[label] = logger
            results[label] = {'collection_seconds': [], 'query_ms': []}

        def reset():
            torch.manual_seed(42)
            env.common_step_counter = 0
            return wrapper.reset()[0]

        def collect(alg, logger, observation):
            for _ in range(config['num_steps_per_env']):
                actions = alg.act(observation)
                observation, rewards, dones, extras = wrapper.step(actions)
                check_nan(observation, rewards, dones)
                alg.process_env_step(observation, rewards, dones, extras)
                logger.process_env_step(rewards, dones, extras)
            return observation

        parity_max_error = 0.
        with torch.inference_mode(), pause_training(args.pause_pid):
            # 先用IEEE FP32验证算法语义，再按训练默认TF32计时；子批GEMM的
            # TF32内核舍入误差另行记录，避免把精度变化混入语义判定。
            configure_torch_backends(allow_tf32=False)
            obs = reset()
            for _ in range(24):
                labels = {name: alg.teachers.query() for name, alg in agents.items()}
                if 'before' in labels:
                    old, new = labels['before'], labels['after']
                    torch.testing.assert_close(old.teacher_ids, new.teacher_ids)
                    torch.testing.assert_close(old.weights, new.weights)
                    parity_max_error = max(parity_max_error, (old.actions - new.actions).abs().max().item())
                    torch.testing.assert_close(old.actions, new.actions, atol=2e-5, rtol=2e-5)
                obs = wrapper.step(labels[next(iter(labels))].actions)[0]
            configure_torch_backends()
            tf32_max_error = None
            if 'before' in agents:
                old = agents['before'].teachers.query()
                new = agents['after'].teachers.query()
                torch.testing.assert_close(old.teacher_ids, new.teacher_ids)
                torch.testing.assert_close(old.weights, new.weights)
                tf32_max_error = (old.actions - new.actions).abs().max().item()
            for repeat in range(args.repeats):
                order = list(agents) if repeat % 2 == 0 else list(reversed(agents))
                for label in order:
                    alg = agents[label]
                    for _ in range(3):
                        alg.teachers.query()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    for _ in range(args.query_repeats):
                        alg.teachers.query()
                    torch.cuda.synchronize()
                    results[label]['query_ms'].append((time.perf_counter() - start) * 1000 / args.query_repeats)
            # 两种顺序交替，固定重置种子，排除首次内核加载和顺序偏差。
            for repeat in range(args.repeats + 1):
                order = list(agents) if repeat % 2 == 0 else list(reversed(agents))
                for label in order:
                    alg, logger = agents[label], loggers[label]
                    alg.storage.clear()
                    alg._clear_samples()
                    logger.ep_extras.clear()
                    obs = reset()
                    # 先完成起步/构型门控，测量随后连续采样，而非反复测回合初始停车。
                    obs = collect(alg, logger, obs)
                    alg.storage.clear()
                    alg._clear_samples()
                    logger.ep_extras.clear()
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    collect(alg, logger, obs)
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - start
                    if repeat:
                        results[label]['collection_seconds'].append(elapsed)
        for values in results.values():
            values['collection_median_seconds'] = statistics.median(values['collection_seconds'])
            values['query_median_ms'] = statistics.median(values['query_ms'])
            values['collection_steps_per_second'] = args.num_envs * config['num_steps_per_env'] / values['collection_median_seconds']
        report = {'task': TASK, 'num_envs': args.num_envs, 'steps_per_env': config['num_steps_per_env'], 'settle_steps': config['num_steps_per_env'], 'gpu': torch.cuda.get_device_name(), 'parity': 'IEEE FP32 passed', 'parity_max_action_error': parity_max_error, 'tf32_max_action_error': tf32_max_error, 'timing_precision': 'training default TF32', 'results': results}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
