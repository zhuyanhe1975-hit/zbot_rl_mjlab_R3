"""MJLab 训练命令行，附加 Zbot 专用开关（见 ``zbot_rl_mjlab/cli.py``）。

    python train.py Mjlab-Zbot-6dof-Deploy-Walking --disturbance --domain-randomization \\
        --env.scene.num-envs 4096 --agent.max-iterations 500

其余参数由 mjlab 原样接收。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    # Preserve this task's fixed-shape GPU reset environment.
    if len(sys.argv)>1 and sys.argv[1]=='Mjlab-Zbot-6dof-Reconfigure-Initial-Closure':
        from scripts.train_refined_gpu_initial_closure import main as closure_main
        raise SystemExit(closure_main(sys.argv[2:]))
    import zbot_rl_mjlab  # noqa: F401, I001
    from zbot_rl_mjlab.cli import CliError, apply_cli
    from mjlab.scripts.train import main

    try:
        sys.argv[1:] = apply_cli()
    except CliError as error:
        sys.exit(f"error: {error}")
    main()
