"""Experiment20: base reference added only to this experiment's model spec."""

from ...robot import get_spec


def base_motion_spec():
    spec = get_spec()
    base = next(body for body in spec.bodies if body.name == 'base')
    # Base -Z already matches head forward at the straight pose; rotate local X/Y
    # so the existing +X-up gravity and tilt logic retains its initial meaning.
    base.add_site(name='base_motion', pos=(0., 0., 0.), quat=(0., 0., 0., 1.), size=(.005,)*3)
    return spec


