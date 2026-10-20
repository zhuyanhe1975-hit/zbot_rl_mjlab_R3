"""Profile the actual collector and benchmark opt-in deterministic teacher graphs.

No PPO updates, checkpoint writes, or training-process interference. Profiler
instrumentation adds overhead: uninstrumented wall times are reported separately.
"""

import argparse
import json
import statistics
import subprocess
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
from tensordict import TensorDict

import zbot_rl_mjlab  # noqa: F401
from zbot_rl_mjlab.cuda_graph import FixedTensorGraph
from zbot_rl_mjlab.teachers import TeacherEnsemble


def measured_call(function, value, repeats):
    start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start = time.perf_counter()
    start_event.record()
    for _ in range(repeats):
        function(value)
    end_event.record()
    torch.cuda.synchronize()
    return {'wall_ms': (time.perf_counter() - start) * 1000 / repeats,
            'gpu_ms': start_event.elapsed_time(end_event) / repeats}


def instrument(obj, name, label):
    original = getattr(obj, name)

    def wrapped(*args, **kwargs):
        with torch.profiler.record_function(label):
            return original(*args, **kwargs)
    setattr(obj, name, wrapped)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', default='Mjlab-Zbot-6dof-Universal-Snake-Leg-Omni-Deploy')
    parser.add_argument('--num-envs', type=int, default=8192)
    parser.add_argument('--steps', type=int, default=24)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, default=Path('outputs/omni_collection_profile.json'))
    parser.add_argument('--trace', type=Path)
    args = parser.parse_args()
    if min(args.num_envs, args.steps, args.repeats) < 1 or not torch.cuda.is_available():
        parser.error('positive counts and CUDA required')
    torch.set_num_threads(1)
    configure_torch_backends()
    cfg = load_env_cfg(args.task)
    cfg.scene.num_envs, cfg.seed = args.num_envs, 42
    env: Any = ManagerBasedRlEnv(cfg, device='cuda:0')
    try:
        wrapper = RslRlVecEnvWrapper(env)
        config = asdict(load_rl_cfg(args.task))
        config['multi_gpu'] = None
        config['num_steps_per_env'] = args.steps
        config['algorithm']['enabled_teachers'] = ('transform',)
        config['algorithm']['teacher_checkpoints'] = None
        config['algorithm']['teacher_warmup_enabled'] = False
        alg: Any = MjlabOnPolicyRunner(wrapper, config, device='cuda:0').alg
        # Explicit selection also makes the legacy task a transform-only diagnostic.
        alg.teachers = TeacherEnsemble(env, enabled_teachers=('transform',))
        env._student_pretraining = False
        env._student_rl_iteration = 0
        command = env.command_manager.get_term('transform')
        command.set_pretraining(False)
        logger = Logger(None, config, cfg, args.num_envs, False, 1, 0, 'cuda:0')
        logger.writer = object()  # type: ignore[assignment]
        obs = wrapper.reset()[0]

        def collect(observation):
            alg.storage.clear()
            alg._clear_samples()
            logger.ep_extras.clear()
            for _ in range(args.steps):
                actions = alg.act(observation)
                observation, rewards, dones, extras = wrapper.step(actions)
                check_nan(observation, rewards, dones)
                alg.process_env_step(observation, rewards, dones, extras)
                logger.process_env_step(rewards, dones, extras)
            return observation

        with torch.inference_mode():
            for _ in range(2):
                obs = collect(obs)
            walls = []
            for _ in range(args.repeats):
                torch.cuda.synchronize()
                start = time.perf_counter()
                obs = collect(obs)
                torch.cuda.synchronize()
                walls.append(time.perf_counter() - start)
            instrument(alg.teachers, 'query', 'teacher_query')
            instrument(alg.actor, 'forward', 'student_actor')
            instrument(alg.critic, 'forward', 'student_critic')
            instrument(wrapper, 'step', 'env_step')
            instrument(alg, 'process_env_step', 'storage')
            instrument(logger, 'process_env_step', 'logger')
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.CUDA]) as profile:
                obs = collect(obs)
                torch.cuda.synchronize()
            averages = profile.key_averages()

            def operation(event):
                return {'name': event.key, 'calls': event.count,
                        'device_type': str(event.device_type),
                        'user_annotation': event.is_user_annotation,
                        'self_cpu_ms': event.self_cpu_time_total / 1000,
                        'total_cpu_ms': event.cpu_time_total / 1000,
                        'self_cuda_ms': event.self_device_time_total / 1000,
                        'total_cuda_ms': event.device_time_total / 1000}

            # Freeze already-frozen actual checkpoint model, including normalizer.
            model = alg.teachers.models['transform']
            value = alg.teachers.observation('transform').clone()

            def deterministic(x):
                return model(TensorDict({'actor': x}, batch_size=[x.shape[0]]), stochastic_output=False)

            graph = FixedTensorGraph(deterministic, value)
            torch.testing.assert_close(graph(value), deterministic(value), atol=2e-5, rtol=2e-5)
            subset = value[:max(1, args.num_envs // 8)]
            timing = {name: [] for name in ('full_eager', 'subset_eager', 'full_graph')}
            functions = {'full_eager': (deterministic, value), 'subset_eager': (deterministic, subset),
                         'full_graph': (graph, value)}
            for repeat in range(args.repeats):
                order = list(functions) if repeat % 2 == 0 else list(reversed(functions))
                for name in order:
                    function, inputs = functions[name]
                    timing[name].append(measured_call(function, inputs, 48))
        if args.trace:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            profile.export_chrome_trace(str(args.trace))
        report = {'task': args.task, 'teacher_selection': ['transform'], 'num_envs': args.num_envs,
                  'steps': args.steps, 'ppo_updates': 0, 'seed': 42,
                  'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
                  'simulation_cuda_graphs': {key: (getattr(env.sim, key, None) is not None)
                                             for key in ('step_graph', 'forward_graph', 'reset_graph', 'sense_graph')},
                  'simulation_use_cuda_graph': env.sim.use_cuda_graph,
                  'collection_seconds': walls, 'collection_median_seconds': statistics.median(walls),
                  'top_self_cpu': [operation(x) for x in sorted(averages, key=lambda x: x.self_cpu_time_total, reverse=True)[:20]],
                  'top_self_cuda': [operation(x) for x in sorted((e for e in averages if not e.is_user_annotation), key=lambda x: x.self_device_time_total, reverse=True)[:20]],
                  'segments': [operation(x) for x in averages if x.key in {'teacher_query', 'student_actor', 'student_critic', 'env_step', 'storage', 'logger'}],
                  'synchronization_ops': [operation(x) for x in averages if any(s in x.key.lower() for s in ('synchron', 'nonzero', '_local_scalar', 'item'))],
                  'mlp_microbenchmark': timing,
                  'scope': 'transform-only PPO collection; graph MLP microbenchmark only; no collector graph enabled',
                  'profiler_note': 'CPU and CUDA averages can repeat names; user annotations on CUDA are timeline spans, not kernel execution; never sum inclusive times',
                  'gpu_processes': subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'], capture_output=True, text=True, check=False).stdout.strip()}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps({'output': str(args.output), 'collection_seconds': walls,
                          'native_graph': report['simulation_use_cuda_graph'], 'mlp': timing}, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
