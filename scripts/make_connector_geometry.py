"""Generate an isolated visualizable disc/pin/socket MuJoCo XML prototype.

Example, illustrative only (not measured hardware dimensions):
  ./scripts/python.sh -m scripts.make_connector_geometry \
    --output /tmp/connector.xml
  ./scripts/python.sh -m mujoco.viewer --mjcf /tmp/connector.xml
"""
import argparse
import math
from pathlib import Path

from zbot_mjbatch.connector_geometry import ConnectorGeometry, make_standalone_spec


def main():
    parser = argparse.ArgumentParser(description='Generate isolated 100mm x 2mm disc and 2mm pin prototype; optional dimensions are in mm and not calibrated')
    parser.add_argument('--pin-radius-mm', type=float, default=2., help='assumed pin radius; default 2 mm')
    parser.add_argument('--bolt-radius-mm', type=float, default=35., help='assumed distance from disc centre to each pin centre; default 35 mm')
    parser.add_argument('--clearance-mm', type=float, default=.1, help='assumed radial clearance; default 0.1 mm')
    parser.add_argument('--initial-gap-mm', type=float, default=20., help='initial spacing between disc surfaces, default 20 mm for visibility')
    parser.add_argument('--output', type=Path, required=True, help='output MuJoCo XML')
    args = parser.parse_args()
    if not math.isfinite(args.initial_gap_mm) or args.initial_gap_mm < 0:
        parser.error('--initial-gap-mm must be finite and nonnegative')
    spec = make_standalone_spec(ConnectorGeometry(args.pin_radius_mm / 1000,
        args.bolt_radius_mm / 1000, args.clearance_mm / 1000))
    # Only the exported viewer scene starts separated. The library model keeps
    # its zero-gap pose, which the geometry tests use explicitly.
    spec.body('pin_plate').pos = (0., 0., args.initial_gap_mm / 1000.)
    spec.compile()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(spec.to_xml())
    print(f'Wrote {args.output}; disc 100 mm diameter, 2 mm thick; 2 mm high pins; '
          f'initial disc gap {args.initial_gap_mm} mm; socket holes are through-holes; '
          'all supplied pin dimensions are uncalibrated.')


if __name__ == '__main__':
    main()
