"""评估输出默认精简，详细轨迹只在明确要求时落盘。"""

import json
import sys

import pytest

from scripts import transformation_eval


@pytest.mark.parametrize("save_trace", [False, True])
def test_cli_only_saves_trace_when_requested(monkeypatch, tmp_path, save_trace):
    output = tmp_path / "nested" / "evaluation.json"
    monkeypatch.setattr(
        transformation_eval,
        "evaluate",
        lambda *args: {"success_rate": 1.0, "trace_env_0": [{"wall_time_s": 0.02}]},
    )
    argv = ["transformation_eval.py", "--output", str(output)]
    if save_trace:
        argv.append("--save-trace")
    monkeypatch.setattr(sys, "argv", argv)
    transformation_eval.main()
    saved = json.loads(output.read_text())
    assert saved["success_rate"] == 1.0
    assert ("trace_env_0" in saved) is save_trace
    if save_trace:
        assert saved["trace_env_0"] == [{"wall_time_s": 0.02}]
