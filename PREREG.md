# Pre-registration v0: Reacher-v5 pilot

Status: DRAFT. Not frozen. Lines marked `DI:` are decisions Di makes and writes in; nothing
is run until they are filled and this file is committed and tagged `prereg-v0`. After the
tag, this file does not change; every later change is logged in DEVIATIONS.md with a date
and a reason.

Why a pre-registration: the pilot's point is to catch the ways an evaluation fools its
author. Writing the rules and the predictions down first is the only protection against
choosing them after seeing the results.

PREREG v1 (PushT) is a separate file, frozen in week 5 before any Diffusion Policy training.

## 1. Environment, pinned

- Gymnasium 1.4.0, `Reacher-v5`, MuJoCo 3.16.0, Stable-Baselines3 2.9.0, as locked in
  `uv.lock`. Python 3.12.
- One step is 0.02 s; an episode is 50 steps.
- `reward_control_weight`: the code default is 1; the docstring says 0.1.
  DI: pass it explicitly as ______ because ______.
  (This only affects training rewards and the return metric; success is distance-based.)

## 2. Seed ranges, disjoint, never mixed

| range | seeds | used for |
|---|---|---|
| pilot | 0-99 | choosing the success radius from the two controls; never reported |
| training | SB3 seeds 100000, 100001, 100002 | the three PPO runs; SB3 gives environment rank i the seed plus i, so keep n_envs below 100; only the first reset of each environment is seed-indexed, later start states come from the environment's RNG stream |
| reporting | 1000-1499 | what the common protocol reports on; 1000-1099 is task1's set and 1000-1049 is LeRobot's default |
| selection | 2000-2999 | the separate range a corrected protocol picks on |
| oracle | 10000-14999 | truth: 5,000 episodes per checkpoint |

A test asserts that the per-episode seeds logged for pilot, reporting, selection and oracle
do not intersect each other or any training seed plus environment rank. Training episodes
after the first reset are not seed-indexed and the test does not cover them; this is
accepted because the pilot range only sets the radius from the two controls, and training
never uses the radius.

## 3. Controls

- Negative control: the random policy (actions drawn uniformly from the action range).
- Positive control: an analytic controller Di writes, two-link inverse kinematics (law of
  cosines) to the target angles plus a PD loop on the joints. It has no learned parts. If it
  cannot reach the target on most pilot seeds, suspect the controller's gains first, then
  the harness; only once the controller is tuned does a failure count against the task or
  the harness.

Both controls run on the pilot seeds before any learned policy exists, and both run on the
reporting seeds alongside every learned policy in every table.

## 4. Success criterion

Success is binary: final fingertip-to-target distance below `SUCCESS_RADIUS_M` at the last
step of the episode.

The radius is set by a rule written here before any numbers exist. Useful numbers: the
fingertip and target spheres touch below 0.019 m; the arm reaches 0.21 m; targets lie
within 0.20 m of the base.

DI: the rule is ______.

A valid rule either fixes the radius from the geometry alone, or computes it from the two
controls' pilot-seed distance distributions. Either way it is written here first, it does
not mention the learned policy, and the two controls' pilot runs must confirm that it
separates them (the positive control mostly succeeds, the random control mostly fails). If
they do not, the rule is revised in DEVIATIONS.md before any learned policy is evaluated.

DI: the radius that came out of the rule is ______ m, set on ______ (date); pilot check:
positive control ______ of 100, random control ______ of 100.

## 5. Metrics

- Primary: success rate (Wilson 95% interval).
- Secondary: final distance (median and seed-clustered bootstrap 95% interval), steps until
  first entry inside the radius, control effort (sum of squared actions), episode return.
- DI: alpha for every test is ______ (the convention is 0.05; choose and say why), and the
  pilot's multiple-comparison rule is ______ (for example Holm across the pre-specified
  secondary analyses).

## 6. Training plan

- Algorithm: PPO from Stable-Baselines3, starting from RL Zoo's published Reacher settings
  (step count and hyperparameters to be read from the RL Zoo config and recorded here
  before training: ______), on CPU.
- 3 training seeds (100000, 100001, 100002). SAC only if time allows.
- A checkpoint every 10% of training (10 per run, K up to 10 candidates per seed).
- Logged per episode, for every evaluation: seed, success, final distance, return, control
  effort, steps to first success, checkpoint id, policy mode (deterministic or stochastic).

## 7. Pre-specified analyses

Every analysis below is run exactly as written. Anything else is exploratory and labeled so.

- A1. Random baseline on reporting seeds 1000-1099: success rate with a Wilson 95% interval;
  final distance with a bootstrap 95% interval. (task1_random_baseline.py writes the
  per-episode log and prints the Wilson interval from Di's stats.py.)
- A2. Oracle truth: every checkpoint of every seed on oracle seeds 10000-14999, with
  Wilson intervals. Each checkpoint's oracle run uses the same seed list, so oracle logs are
  seed-aligned across checkpoints.
- A3. Protocol A vs Protocol B (PLAN.md, Design): reporting bias and selection regret for
  n = 10, 50, 500 and K = 2..10, candidate sets = the last K checkpoints of a run (primary)
  and all K-subsets (secondary). Measured as the single literal run on seeds 1000-1049 and
  as a distribution over 1,000 seed sets drawn with replacement from the oracle logs; the
  same drawn set scores all K checkpoints for A, and B draws two independent sets. Truth for
  the picked checkpoint is its full oracle mean. Primary estimand: mean reporting bias at
  n = 50, K = 5, per training seed with its resampling interval.
- A4. Rank-flip probability: for every pair of checkpoints, the gap and its standard error
  are estimated from the seed-aligned oracle logs (paired: the SE of the per-seed success
  difference). Pairs with |gap| below 2 SE are excluded. For the rest, the fraction of
  n-episode seed-aligned resamples that order them wrongly, for n = 10, 50, 500, binned by
  estimated gap and labeled as estimated, not true, gap. A tie counts as half a flip. The
  empirical curve is compared with the analytic prediction computed from the same estimated
  gaps, so both carry the same conditioning.
- A5. A/A calibration: a checkpoint compared with itself on two disjoint n-episode draws,
  1,000 times, for every test used in the pilot. Pass rule: observed false-difference rate
  at most alpha + 2 * sqrt(alpha * (1 - alpha) / 1000) (0.064 at alpha = 0.05); a rate
  below 0.036 is reported as conservative. This check can detect a true rate above about
  0.07; it cannot certify a rate exactly at alpha.
- A6 (stretch, weeks 9-10). Paired e-process: the wealth is checked after every pair up to
  the full horizon n, and a replication counts as a false decision if it ever reaches
  1/alpha; observed false-decision rate on A/A pairs and trials-to-decision on true-gap
  pairs, against the fixed-n McNemar test. If A6 is cut, the report says so.

## 8. Predictions, written before running

Each prediction is a number or a range with a one-line reason. Vague predictions do not
count.

- DI: A1 random baseline success rate: ______ because ______.
- DI: A2 best PPO checkpoint true success rate, per seed: ______.
- DI: A3 reporting bias under Protocol A at n = 50, K = 5: ______ points. Under
  Protocol B: ______ points. Selection regret under each: ______.
- DI: A4 rank-flip probability at n = 50 for an estimated gap of 10 points: ______.
- DI: A5 false-difference rate for each test used: ______ because ______.
- DI: A6, if run: trials-to-decision for the e-process will be ______ of McNemar's fixed n
  because ______.

## 9. Planted harness bugs

Each fault below is implemented as a deliberately broken policy or harness setting, with a
pytest that asserts which check catches it. The pilot passes only if every planted bug is
caught. Five are needed for the pilot; the rest are optional.

1. VecNormalize statistics not loaded at evaluation.
2. Evaluation seeds overlapping training or pilot seeds.
3. Success scored at the wrong timestep.
4. Stochastic actions where deterministic ones were meant.
5. A sign-flipped joint.
6. NaN actions.
7. A policy that parks just outside the success radius.
8. Truncated episodes counted as successes.

DI: choose the five for the pilot, and add or remove faults here with a reason, before
freezing: ______.

## 10. Gates

- End of week 2: PPO beats the random control with non-overlapping 95% intervals on at
  least 2 of 3 seeds, or scope freezes until fundamentals are fixed.
- End of week 3: every test used passes the A5 rule and every planted bug is caught, or no
  GPU spend.

## 11. Freeze

DI: frozen on ______ (date) at commit ______, tag `prereg-v0`. Changes after this line go
to DEVIATIONS.md.
