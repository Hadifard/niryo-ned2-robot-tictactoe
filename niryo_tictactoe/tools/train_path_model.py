"""
File: niryo_tictactoe/tools/train_path_model.py
Description: Trains the learned-path model (common/path_learning.py) and saves it to models/path_model.json.
The robot "practises" 150 simulated transfers over a random obstacle: for each one it tries 29 lift heights,
keeps the shortest collision-free one (trial and error) and finally fits a ridge-regularised polynomial
    lift = f(distance, obstacle height)
by least squares. The script prints the training result and an honest check on 300 UNSEEN transfers.
No robot needed (a few seconds on a laptop).   Usage:  python main.py train-path [--trials 150]
Author: Hadi Sarhangi Fard
LinkedIn: https://www.linkedin.com/in/hadi-sarhangi-fard-mech-eng/
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from common.path_learning import PathLearner, min_clearance, path_points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=150, help="number of practice transfers")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="models/path_model.json")
    args = ap.parse_args()

    print(f"training on {args.trials} simulated transfers (trial and error + ridge regression) ...")
    learner = PathLearner().train(trials=args.trials, seed=args.seed)
    print("learned weights  [1, dist, height, dist*height, dist^2, height^2]:")
    print("  ", np.round(learner.w, 4))

    rng, errs, ok = np.random.default_rng(args.seed + 999), [], 0
    for _ in range(300):
        a, b, obstacles, top = learner._random_transfer(rng)
        best = learner.best_lift(a, b, obstacles)
        dist = float(np.linalg.norm(b[:2] - a[:2]))
        errs.append(abs(learner.predict_lift(dist, top) - best))
        pts, _ = learner.plan(a, b, obstacles)
        ok += min_clearance(pts, obstacles, learner.monitor) >= learner.monitor.margin - 1e-9
    print(f"unseen transfers: mean lift error {np.mean(errs) * 1e3:.1f} mm, collision-free plans {ok / 3:.1f} %")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    learner.save(args.out)
    print(f"saved {args.out}   (load later with PathLearner.load('{args.out}'))")


if __name__ == "__main__":
    main()
