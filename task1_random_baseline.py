"""Task 1: measure a random-policy baseline on held-out evaluation seeds.

A random policy picks each action uniformly from the action range and ignores
what it observes. A learned policy has to beat this to have learned anything.

Before running: fill in the two TODOs, and write your success criterion and
your prediction in NOTEBOOK.md.

Run it:
    uv run task1_random_baseline.py
"""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np

# Held-out evaluation seeds. Each seed fixes a target position and starting arm
# pose. Training will never use these seeds, so evaluation measures how a
# policy does on situations it has not trained on.
EVAL_SEEDS = list(range(1000, 1100))

# TODO 1: decide what counts as success, before you see any results.
# Useful numbers: the target sphere has radius 0.009 m and the fingertip sphere
# 0.010 m, so they touch when the distance is below 0.019 m. The arm reaches
# 0.21 m from its base, and targets appear within 0.20 m of the base.
SUCCESS_RADIUS_M = None

if SUCCESS_RADIUS_M is None:
    raise SystemExit("TODO 1: set SUCCESS_RADIUS_M, and write down why in NOTEBOOK.md.")

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
print(f"Random policy on {len(EVAL_SEEDS)} held-out seeds")
print(f"  final distance: mean {d.mean():.3f} m (standard error {d.std(ddof=1) / np.sqrt(len(d)):.3f}), "
      f"median {np.median(d):.3f} m")
print(f"  success (final distance < {SUCCESS_RADIUS_M} m): {successes}/{len(d)}")
print(f"  episode return: mean {np.mean(returns):+.2f}")

Path("results").mkdir(exist_ok=True)
record = {
    "policy": "random",
    "env": "Reacher-v5",
    "eval_seeds": [EVAL_SEEDS[0], EVAL_SEEDS[-1]],
    "success_radius_m": SUCCESS_RADIUS_M,
    "final_distances": final_distances,
    "returns": returns,
}
Path("results/random_baseline.json").write_text(json.dumps(record, indent=2) + "\n")
print("Saved results/random_baseline.json")
