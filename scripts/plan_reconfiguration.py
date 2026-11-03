"""Generate and collision-check joint-space closure plans for the guided experiment."""
import argparse
import json
from pathlib import Path

from zbot_rl_mjlab.tasks.reconfiguration.planning import DEFAULT_PLAN, generate_plan

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=DEFAULT_PLAN)
    parser.add_argument('--starts', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    document = generate_plan(args.output, args.starts, args.seed)
    print(json.dumps({'output':str(args.output),'solutions':len(document['solutions']),
                      'gaps_m':[s['gap_m'] for s in document['solutions']]},indent=2))
