"""Check physical motion reaches both camera images. Moves and resets the live lab.

Run outside a participant session using the notebook Python environment.
A successful HTTP response alone cannot detect a stale render scene.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vinci_lab.client import Lab


def assess(before, still, moved):
    metrics = {}
    for key in ('rgb_front', 'rgb_wrist'):
        a, b, c = (np.asarray(o[key], dtype=float) for o in (before, still, moved))
        noise = float(np.abs(a - b).mean())
        motion = float(np.abs(b - c).mean())
        threshold = max(2.5, 4 * noise)
        metrics[key] = {'idle_difference': noise, 'motion_difference': motion,
                        'threshold': threshold, 'passed': motion > threshold}
    joint_motion = float(np.max(np.abs(np.asarray(moved['joint_pos'][:7]) - np.asarray(still['joint_pos'][:7]))))
    return {'passed': joint_motion > 0.15 and all(v['passed'] for v in metrics.values()),
            'joint_motion_rad': joint_motion, 'cameras': metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--out')
    args = parser.parse_args()
    lab = Lab(args.url)
    try:
        lab.reset()
        before = lab.observe()
        still = lab.observe()
        target = list(still['joint_pos'][:7])
        target[0] += 0.8  # free-space sweep from home; well within joint limits
        target[1] -= 0.3  # lift away from the cabinet; make front-view motion clearly visible
        lab.move_joints(target)
        moved = lab.observe()
        result = assess(before, still, moved)
    finally:
        lab.reset()
    print(json.dumps(result), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
    if not result['passed']:
        print('Camera motion check failed. Do not validate visual policies from these images; inspect/restart the Isaac service and repeat this check.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
