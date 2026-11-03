"""Benchmark identical full-disc models with polygonal hole resolutions."""
import argparse
import json
import math

from scripts.compare_socket_geometry import run
from zbot_mjbatch.full_square_connector import FullSquareConnector, make_full_square_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=2000)
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    if args.steps < 1 or args.repeats < 1:
        p.error('steps and repeats must be positive')
    for sides in (4, 8, 16):
        config = FullSquareConnector(hole_segments=sides)
        model = make_full_square_model(config)
        for scenario, (x, z, pressure) in {'free': (0., .003, 0.),
                                           'face_load': (0., 0., 1.),
                                           'pin_wall': (.0003, .0005, 0.)}.items():
            metrics = run(model, steps=args.steps, repeats=args.repeats,
                          x=x, z=z, pressure=pressure)
            shape = {'hole_sides': sides, 'outer_sides': config.outer_segments,
                     'outer_radial_deviation_mm': 1000 * config.disk_radius_m * (1 - math.cos(math.pi / config.outer_segments)),
                     'hole_vertex_overextension_mm': 1000 * config.hole_radius_m * (1 / math.cos(math.pi / sides) - 1),
                     'min_pin_center_clearance_mm': config.radial_clearance_m * 1000,
                     'max_pin_center_clearance_mm': config.radial_clearance_m / math.cos(math.pi / sides) * 1000}
            print(json.dumps({'shape': shape, 'scenario': scenario, 'steps': args.steps,
                              'repeats': args.repeats, **metrics}, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
