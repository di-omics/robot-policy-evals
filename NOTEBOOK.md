# Lab notebook

Entries are in order. Every prediction is written down before the run it predicts.

## 2026-10-18: Reacher-v5, first look

**Command.** `uv run explore_env.py`

**Facts checked in code, not docs.** The reward is computed after the physics step. The
control-cost weight is 1, not the 0.1 the Gymnasium 1.4.0 docs state.

**Q1.** The policy gets cos and sin of each joint angle instead of the angle itself. Why?
(Think about what happens to an angle as the shoulder spins past a full turn.)

A:

**Q2.** Observations 8 and 9 (fingertip minus target) can be calculated from the other
eight. Why might the environment include them anyway?

A:

**Q3.** The reward has two parts. What behavior does the control term discourage? What
could go wrong if its weight were 100 times larger?

A:

**Q4.** Watch `media/random_policy.gif`. What does the arm do, and why would a random
policy rarely end up touching the target?

A:

## Task 1: random-policy baseline (`task1_random_baseline.py`), PREREG A1

Write these before running:

- Success radius, copied from PREREG.md section 4 after the rule was written there (date):
- Prediction (mean final distance, how many of 100 reporting seeds succeed, and the
  width of the Wilson interval you expect):

Then:

- Result:
- Interpretation:
- What remains uncertain:
