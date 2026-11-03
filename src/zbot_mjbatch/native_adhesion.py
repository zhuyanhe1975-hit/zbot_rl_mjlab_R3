"""Opt-in native geom-adhesion proxies on the independent six-module model.

This is a collision-based numerical approximation, not a magnetic-field model.
The hypothesized 5 N is adhesion capacity *per generated contact*, not a
measured per-magnet force or guaranteed total force. Proxy balls are flush with
the plate when radius equals embed depth, so active contacts also contribute
physical repulsion and may alter support. Do not combine this model with the
separate custom magnetic-force model, or treat it as validated robot physics.
"""
from __future__ import annotations

import math
import mujoco

from zbot_mjbatch.physical_zbot_geometry import add_physical_zbot_geometry, _site

def add_native_adhesion_proxies(spec, *, contact_force_n=5., interaction_gap_m=.003,
                                 proxy_radius_m=.001):
    """Add 4+4 isolated proxy balls per interface on previously created sites.

    Pair-specific high bits 1..6 only permit source-to-target contact belonging
    to the same interface. Bit zero remains reserved for the existing robot and
    ground. Geom adhesion is applied on the source side only; target adhesion
    is zero so MuJoCo's additive rule does not double the configured capacity.
    A 3 mm gap acts between surfaces, not magnet center points. At 1 mm embed
    and 1 mm sphere radius both proxy surfaces coincide with their flange faces.
    Actual magnet sizes, field law, and force are not measured.
    """
    if (not math.isfinite(contact_force_n) or contact_force_n <= 0 or
            not math.isfinite(interaction_gap_m) or interaction_gap_m < 0 or
            not math.isfinite(proxy_radius_m) or proxy_radius_m <= 0):
        raise ValueError('invalid adhesion capacity, range or proxy radius')
    metadata=[]
    for face in range(6):
        bit=1 << (face+1)
        pair={'source':[],'target':[], 'mask':bit}
        for side in ('source','target'):
            for j in range(4):
                site=_site(spec,f'physical_connector_{face}_magnet_{side}_{j}')
                name=f'physical_connector_{face}_adhesion_{side}_{j}'
                if any(geom.name==name for geom in spec.geoms):
                    raise ValueError(f'adhesion proxy already exists: {name}')
                # One directional filter only; both sides must not carry the
                # common base bit, otherwise they could stick to robot/terrain.
                site.parent.add_geom(name=name,type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    pos=site.pos.copy(),size=(proxy_radius_m,0.,0.),mass=0.,
                    contype=bit if side=='source' else 0,
                    conaffinity=0 if side=='source' else bit,
                    adhesion=contact_force_n if side=='source' else 0.,
                    gap=interaction_gap_m if side=='source' else 0.,
                    condim=1,friction=(0.,0.,0.),
                    rgba=(1.,.15,.05,.45) if side=='source' else (.05,.65,1.,.45))
                pair[side].append(name)
        metadata.append(pair)
    return metadata


def make_native_adhesion_zbot_model(*, contact_force_n=5., interaction_gap_m=.003,
                                    proxy_radius_m=.001, magnet_radius_m=.035,
                                    magnet_embed_depth_m=.001, magnet_phase_rad=math.pi/4,
                                    pin_shape='taper', pin_tip_radius_m=.0018, pin_sides=16):
    """Create standalone passive-adhesion model; no custom force or FSM."""
    from mjlab.scene import Scene
    from zbot_rl_mjlab.tasks.reconfiguration.reference_dynamic import reference_env_cfg
    cfg=reference_env_cfg().scene
    def decorate(spec):
        add_physical_zbot_geometry(spec, pin_shape=pin_shape,
            pin_tip_radius_m=pin_tip_radius_m, pin_sides=pin_sides,
            magnet_radius_m=magnet_radius_m, magnet_embed_depth_m=magnet_embed_depth_m,
            magnet_phase_rad=magnet_phase_rad)
        add_native_adhesion_proxies(spec, contact_force_n=contact_force_n,
            interaction_gap_m=interaction_gap_m, proxy_radius_m=proxy_radius_m)
    cfg.spec_fn=decorate
    model=Scene(cfg, device='cpu').compile()
    model.opt.timestep=.001
    return model
