"""Step 1: run the Reacher arm and see what a policy observes, controls and is scored on.

Prints the observation, action and reward for one step, plays a few episodes of
random actions, and saves them as media/random_policy.gif.

Run it:
    uv run explore_env.py
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

OBSERVATION_LABELS = [
    "cos(shoulder angle)",
    "cos(elbow angle)",
    "sin(shoulder angle)",
    "sin(elbow angle)",
    "target x (m)",
    "target y (m)",
    "shoulder angular velocity (rad/s)",
    "elbow angular velocity (rad/s)",
    "fingertip x - target x (m)",
    "fingertip y - target y (m)",
]
ACTION_LABELS = ["shoulder torque", "elbow torque"]

# Camera straight above the table, close in. In the picture the pink dot is the
# arm's base, the green dot is the fingertip and the red dot is the target.
TOP_DOWN = {"distance": 0.75, "elevation": -90.0, "azimuth": 90.0, "lookat": np.zeros(3)}

env = gym.make("Reacher-v5", render_mode="rgb_array", default_camera_config=TOP_DOWN)
obs, info = env.reset(seed=0)
env.action_space.seed(0)

steps = env.spec.max_episode_steps
print("Reacher-v5: a two-joint arm on a table has to touch a red target.")
print(f"One step is {env.unwrapped.dt} s of simulated time; an episode is {steps} steps.\n")

print(f"OBSERVATION: what the policy sees each step ({env.observation_space.shape[0]} numbers)")
for i, (label, value) in enumerate(zip(OBSERVATION_LABELS, obs)):
    print(f"  {i:>2}  {label:<36} {value:+.4f}")

low, high = env.action_space.low, env.action_space.high
print(f"\nACTION: what the policy controls ({env.action_space.shape[0]} numbers, each in [{low[0]:g}, {high[0]:g}])")
for i, label in enumerate(ACTION_LABELS):
    print(f"  {i:>2}  {label}")

# One step with a fixed action, and the reward broken into its two parts.
action = np.array([0.5, -0.5], dtype=np.float32)
distance_before = np.linalg.norm(obs[8:10])
obs, reward, terminated, truncated, info = env.step(action)
distance_after = np.linalg.norm(obs[8:10])
print(f"\nREWARD after one step with action {action.tolist()}:")
print(f"  distance to target: {distance_before:.4f} m before the step, {distance_after:.4f} m after")
print(f"  reward_dist = -(distance after the step)    = {info['reward_dist']:+.4f}")
print(f"  reward_ctrl = -(weight x sum of action^2)   = {info['reward_ctrl']:+.4f}")
print(f"  reward      = reward_dist + reward_ctrl     = {reward:+.4f}")
weight = -info["reward_ctrl"] / np.square(action).sum()
print(f"  control-cost weight, measured: {weight:g} (the Gymnasium docs say 0.1; the code uses 1)")

# A few full episodes of random actions, recorded as a GIF.
frames = []
print("\nRANDOM POLICY: actions drawn uniformly from the action range, ignoring observations")
for seed in range(3):
    obs, info = env.reset(seed=seed)
    episode_return, done = 0.0, False
    while not done:
        frames.append(Image.fromarray(env.render()).resize((240, 240)))
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        episode_return += reward
        done = terminated or truncated
    final_distance = np.linalg.norm(obs[8:10])
    print(f"  seed {seed}: return {episode_return:+.2f}, final distance {final_distance:.3f} m")
env.close()

Path("media").mkdir(exist_ok=True)
# 40 ms per frame is half real speed (one step is 20 ms).
frames[0].save("media/random_policy.gif", save_all=True, append_images=frames[1:], duration=40, loop=0)
print("\nSaved media/random_policy.gif (3 episodes, half speed)")
