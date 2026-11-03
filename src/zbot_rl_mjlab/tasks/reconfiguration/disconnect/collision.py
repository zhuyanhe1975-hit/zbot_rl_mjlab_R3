"""Model-scoped EPA capacity adapter for the installed MuJoCo Warp version.

Only affects this experiment's convex collision scratch allocation. No shared
package files, meshes, collision masks, or solver tolerances are changed.
"""
from dataclasses import dataclass
from functools import wraps
from threading import RLock

from mjlab.sim import SimulationCfg

_LOCK=RLock()
_MARKER='_zbot_epa_horizon_capacity'


def install_epa_adapter():
    from mujoco_warp._src import collision_convex,collision_driver
    if getattr(collision_driver.convex_narrowphase,'_zbot_epa_adapter',False):
        return
    original=collision_driver.convex_narrowphase

    @wraps(original)
    def narrowphase(model,*args,**kwargs):
        # Allocation occurs during CUDA graph capture (or each eager step).
        # Restore the upstream default immediately, also when capture fails.
        with _LOCK:
            previous=collision_convex.MJ_MAX_EPAHORIZON
            capacity=getattr(model.opt,_MARKER,previous)
            collision_convex.MJ_MAX_EPAHORIZON=capacity
            try:
                return original(model,*args,**kwargs)
            finally:
                collision_convex.MJ_MAX_EPAHORIZON=previous

    narrowphase._zbot_epa_adapter=True
    collision_driver.convex_narrowphase=narrowphase


@dataclass(kw_only=True)
class DisconnectSimulationCfg(SimulationCfg):
    epa_horizon_capacity: int=64

    def apply_wp_opt(self,wp_opt):
        super().apply_wp_opt(wp_opt)
        if not isinstance(self.epa_horizon_capacity,int) or self.epa_horizon_capacity<24:
            raise ValueError('EPA horizon capacity must be an integer >=24')
        install_epa_adapter()
        setattr(wp_opt,_MARKER,self.epa_horizon_capacity)
