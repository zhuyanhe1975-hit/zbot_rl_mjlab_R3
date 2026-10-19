from mjlab.tasks.registry import load_env_cfg

import zbot_rl_mjlab  # noqa: F401
from zbot_rl_mjlab.tasks.walking.env_cfg import WALKING_TASK_ID


def test_height_scan_rays_start_at_the_soles():
    # 脚底站点位于双脚的地面接触平面上；foot_1 刚体原点
    # 不在该平面上（约高出脚底 54 mm），因此参考坐标系必须使用站点。
    cfg = load_env_cfg(WALKING_TASK_ID)
    sensor = next(s for s in cfg.scene.sensors if s.name == "foot_height_scan")
    assert [(f.type, f.name) for f in sensor.frame] == [
        ("site", "foot_0_sole"),
        ("site", "foot_1_sole"),
    ]
