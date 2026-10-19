# robot-policy-evals

Is the gain real? A known-truth audit of how robot-policy checkpoints get picked and
reported. A common routine evaluates every checkpoint on 50 episodes, keeps the best, and
reports its score on those same episodes, which biases the reported number upward. This
project measures how far, first on Reacher-v5 (a pilot that proves the evaluation pipeline
against planted faults), then on LeRobot's Diffusion Policy on PushT against thousands of
held-out rollouts per checkpoint.

Why it matters: in LeRobot 0.6.1 the training-time evaluation and the standalone
`lerobot-eval` both default to seed 1000 and 50 episodes, with no option for a separate
selection seed range, so a checkpoint picked from the training curve and then reported has
been scored on the start states it was picked on. A measured inflation is the case for a
small upstream change; a null result says the default is fine.

Status: started 2026-10-18. Plan committed, nothing run yet beyond the environment checks.
See [PLAN.md](PLAN.md) for the question, design, timeline and gates, [PREREG.md](PREREG.md)
for the pilot's pre-registration (draft, not yet frozen), and [NOTEBOOK.md](NOTEBOOK.md) for
the running lab notebook.

![random policy on Reacher-v5](media/random_policy.gif)

*A random policy on Reacher-v5, top-down view at half speed. Pink is the arm's base,
green the fingertip, red the target.*

## Where this starts

1. Run the simulation and understand observations, actions and rewards. (`explore_env.py`)
2. Measure a random-policy baseline on the reporting seeds. (`task1_random_baseline.py`)
3. Train PPO on three seeds, saving checkpoints; walk through one update by hand.
4. Write the statistics module from the formulas and prove it on planted answers.
5. Plant faults in the harness and prove the tests catch them.
6. Audit checkpoint selection on Reacher against oracle truth, then move to PushT.

Weeks and gates are in PLAN.md. It starts with state observations; a real arm comes only
after the simulation study ships.

## The environment: Gymnasium Reacher-v5 (MuJoCo)

A two-joint arm on a table has to touch a target. One step is 0.02 s of simulated time
and an episode lasts 50 steps (1 s). The arm reaches 0.21 m from its base; each episode
places the target at random within 0.20 m of the base.

**Observation (10 numbers).** Cosine and sine of the two joint angles, target x and y,
the two joint angular velocities, and the fingertip-minus-target vector (x, y).

**Action (2 numbers in [-1, 1]).** Torque on the shoulder joint and on the elbow joint.

**Reward per step.** `-(distance from fingertip to target) - (sum of squared actions)`.
The distance is measured after the step. The control-cost weight is 1: the Gymnasium
1.4.0 docs say 0.1, but the code uses 1, and `explore_env.py` measures it directly.

## Run it

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv run explore_env.py
```

## Notes

The lab notebook (`NOTEBOOK.md`) records each prediction before the run that tests it.
The learning-rate warm-up that preceded this repo is in
[ml-foundations](https://github.com/di-omics/ml-foundations).
