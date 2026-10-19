"""多个任务共用的基座扰动行为与配置。"""

import torch
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply


class PartialBaseImpulse(envs_mdp.apply_body_impulse):
    """对部分环境施加随机冲量，其余环境不受扰动。"""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        fraction = float(cfg.params.get("disturbed_fraction", 0.75))
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("disturbed_fraction must be in [0, 1]")
        self._disturbed_mask = (
            torch.rand(self._num_envs, device=self._device) < fraction
        )

    def __call__(self, env, env_ids, *args, **kwargs):
        kwargs.pop("disturbed_fraction", None)
        super().__call__(env, env_ids, *args, **kwargs)
        quiet = (~self._disturbed_mask).nonzero(as_tuple=False).flatten()
        if quiet.numel():
            zeros = torch.zeros((len(quiet), self._num_bodies, 3), device=self._device)
            self._asset.write_external_wrench_to_sim(
                zeros, zeros, env_ids=quiet, body_ids=self._body_ids
            )

    def debug_vis(self, visualizer):
        """仅绘制实际受到扰动的刚体，箭头终点位于基座。"""
        viz = self._viz_cfg
        body_ids = self._body_ids
        if isinstance(body_ids, slice):
            body_ids = range(self._asset.num_bodies)[body_ids]
        for env_id in visualizer.get_env_indices(self._num_envs):
            if not self._active[env_id] or not self._disturbed_mask[env_id]:
                continue
            for body_id in body_ids:
                force = self._asset.data.body_external_wrench[env_id, body_id, :3]
                if force.norm().item() <= viz.min_force:
                    continue
                point = self._asset.data.body_com_pos_w[env_id, body_id]
                if self._body_point_offset is not None:
                    quat = self._asset.data.body_com_quat_w[env_id, body_id]
                    point = point + quat_apply(quat, self._body_point_offset)
                end = point.detach().cpu().numpy()
                start = end - force.detach().cpu().numpy() * viz.scale
                visualizer.add_arrow(start, end, color=viz.rgba, width=viz.width)


def add_base_disturbance(
    cfg,
    force_range=(-5.0, 5.0),
    torque_range=(-0.5, 0.5),
    duration_s=(0.08, 0.20),
    cooldown_s=(0.8, 2.0),
    disturbed_fraction=0.25,
):
    """对部分环境施加随机基座力／力矩冲量。"""
    cfg.events["base_disturbance"] = EventTermCfg(
        func=PartialBaseImpulse,
        mode="step",
        params={
            "force_range": force_range,
            "torque_range": torque_range,
            "duration_s": duration_s,
            "cooldown_s": cooldown_s,
            "asset_cfg": SceneEntityCfg("robot", body_names=("base",)),
            "body_point_offset": (0.0, 0.0, 0.0),
            "disturbed_fraction": disturbed_fraction,
        },
    )
    return cfg
