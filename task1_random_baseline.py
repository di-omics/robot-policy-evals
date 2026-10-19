"""Task 1: measure a random-policy baseline on the reporting seeds.

A random policy picks each action uniformly from the action range and ignores
what it observes. A learned policy has to beat this to have learned anything.
This is PREREG.md analysis A1.

Before running: write the radius rule in PREREG.md section 4, fill in the three
TODOs here, and write your prediction in NOTEBOOK.md.

Run it:
    uv run task1_random_baseline.py
"""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np

# Reporting seeds (PREREG.md section 2). Each seed fixes a target position and
# starting arm pose. Training seeds are 100000 and up, so no training episode
# is ever seeded with one of these.
EVAL_SEEDS = list(range(1000, 1100))

# TODO 1: SUCCESS_RADIUS_M is not chosen here. It is set by the rule you write
# in PREREG.md section 4, before any numbers exist. Write the rule, then paste
# the radius it gives and the date.
# Useful numbers: the target sphere has radius 0.009 m and the fingertip sphere
# 0.010 m, so they touch when the distance is below 0.019 m. The arm reaches
# 0.21 m from its base, and targets appear within 0.20 m of the base.
SUCCESS_RADIUS_M = None  # set on ______ from PREREG.md section 4

if SUCCESS_RADIUS_M is None:
    raise SystemExit("TODO 1: write the rule in PREREG.md section 4, then set SUCCESS_RADIUS_M.")

env = gym.make("Reacher-v5")
final_distances, returns = [], []
for seed in EVAL_SEEDS:
    obs, info = env.reset(seed=seed)
    env.action_space.seed(seed)
    episode_return, done = 0.0, False
    while not done:
        # TODO 2: three lines.
        #   - pick a random action (hint: env.action_space has a method for this)
        #   - step the environment with it, keeping obs and reward
        #   - set done when the episode is terminated or truncated
        raise NotImplementedError("TODO 2: fill in the episode loop")
        episode_return += reward
    # obs[8:10] is fingertip minus target, so its length is the final distance.
    final_distances.append(float(np.linalg.norm(obs[8:10])))
    returns.append(float(episode_return))
env.close()

d = np.array(final_distances)
successes = int((d < SUCCESS_RADIUS_M).sum())
print(f"Random policy on {len(EVAL_SEEDS)} reporting seeds")
print(f"  final distance: mean {d.mean():.3f} m (standard error {d.std(ddof=1) / np.sqrt(len(d)):.3f}), "
      f"median {np.median(d):.3f} m")
print(f"  success (final distance < {SUCCESS_RADIUS_M} m): {successes}/{len(d)}")
print(f"  episode return: mean {np.mean(returns):+.2f}")

# TODO 3: write wilson_interval(successes, n, z=1.96) in a new stats.py, from
# the formula (look it up; it is a few lines). Test it on a planted case first:
# 0 of 10 must give an interval that starts at 0, not below. Then import it
# here and print the 95% interval for the success rate. That interval is
# PREREG.md A1's primary number.

# One record per episode, each carrying its seed, so that the seed-range test
# in PREREG.md section 2 can be written against this file.
Path("results").mkdir(exist_ok=True)
record = {
    "policy": "random",
    "env": "Reacher-v5",
    "checkpoint": None,
    "mode": "stochastic",
    "success_radius_m": SUCCESS_RADIUS_M,
    "episodes": [
        {"seed": s, "final_distance_m": dist, "return": ret, "success": bool(dist < SUCCESS_RADIUS_M)}
        for s, dist, ret in zip(EVAL_SEEDS, final_distances, returns)
    ],
}
Path("results/random_baseline.json").write_text(json.dumps(record, indent=2) + "\n")
print("Saved results/random_baseline.json")
