"""Write an isolated full-disc polygonal-hole connector for visual inspection."""
import argparse
import math
from pathlib import Path

from zbot_mjbatch.full_square_connector import FullSquareConnector, make_full_square_spec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hole-sides', type=int, choices=(4, 8, 16), default=8)
    p.add_argument('--initial-gap-mm', type=float, default=20.)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not math.isfinite(args.initial_gap_mm) or args.initial_gap_mm < 0:
        p.error('initial gap must be finite and nonnegative')
    spec = make_full_square_spec(FullSquareConnector(hole_segments=args.hole_sides))
    spec.body('pin_plate').pos = (0., 0., args.initial_gap_mm / 1000.)
    model = spec.compile()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(spec.to_xml())
    print(f'{args.output}: {args.hole_sides}-sided holes, {model.ngeom} geoms; '
          'independent illustrative model, no magnets or Zbot collision integration')


if __name__ == '__main__':
    main()
