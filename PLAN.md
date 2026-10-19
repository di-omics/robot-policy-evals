# Is the gain real? A known-truth audit of how robot-policy checkpoints get picked and reported

Status: plan committed 2026-10-18. Nothing below has been run yet except the checks in
"Facts checked on 2026-10-18". PREREG.md holds the pre-registration; DEVIATIONS.md records
every change made after a pre-registration is frozen.

## The question

When a robot-learning result is reported as "success rate X% on PushT", the number can
come from this routine: train a policy, evaluate every checkpoint on 50 episodes, keep the
best one, report its 50-episode score. If the same 50 start states are used both to pick
the checkpoint and to report it, the reported number is biased upward: the best of K noisy
scores is high partly because picking the maximum also picks the luck (the winner's curse).
How large is that bias on a real imitation-learning benchmark, and what does it do to
comparisons between policies?

Primary question. For LeRobot's Diffusion Policy on PushT, trained with the published
recipe, how much does reported success overstate true held-out success when the checkpoint
is chosen on the same start-state seeds it is reported on, compared with choosing it on a
separate seed range? How does the gap change with episodes per checkpoint (n = 10, 50, 500)
and with the number of candidate checkpoints K? "True" means an estimate from at least
1,000-2,000 held-out episodes per checkpoint, reported with its own interval.

Secondary questions, each with one pre-registered primary analysis:

- S1. How often do common protocols rank two checkpoints in the wrong order, as a function
  of their estimated true gap and of n?
- S2. The true success of each training seed's best checkpoint, as 5 numbers (Reacher: 3)
  with oracle intervals, and the across-seed standard deviation with its chi-square
  interval. The scoped claim is whether that spread is clearly above, clearly below, or
  indistinguishable from the standard error of a single 50-episode evaluation (6.7 points at
  p = 0.65). The behavior-cloning policy's seed spread is reported as its own result, not as
  a stand-in for Diffusion Policy.
- S3. For a policy that samples its actions stochastically, how much variance does pairing
  episodes on start state actually remove, with and without a shared action-noise RNG? Does
  a paired anytime-valid test keep its promised false-decision rate when checked against
  oracle truth? (A betting e-process is a running test you may check after every episode
  and stop whenever you like, with the false-alarm rate still held at alpha.) S3 is a
  stretch goal; if it is cut, the report says so.

A null result (little inflation, rare rank flips) is a result: "the common protocol holds
on PushT" gets published the same way.

## Why this question

- The seed overlap is LeRobot's default. In LeRobot (installed 0.6.1, and main at 146d847
  on 2026-10-16) the periodic evaluation inside training and the standalone evaluation
  script both seed episodes from `cfg.seed`. Both configs default `seed` to 1000 and
  `n_episodes` to 50, and seeds count up by one. There is no option for a separate
  selection seed range. LeRobot does not pick a checkpoint for anyone; but a user who leaves
  both seeds at their defaults, picks a checkpoint from the in-training evaluation curve,
  and reports `lerobot-eval`'s number on it has scored it on the exact start states it was
  picked on. The coupling is a default, not a certainty: a user who sets a training seed
  moves the in-training evaluation to that seed range while `lerobot-eval` stays at 1000
  (the public `lerobot/diffusion_pusht` checkpoint was trained with seed 100000, so its own
  two evaluations were disjoint). The audit measures the default. How common the practice
  is has not been measured here. A measured inflation is the case for a small upstream
  change (a separate selection seed range); a null result says the default is fine.
- The closest prior work leaves exactly this gap. Yu 2026 (arXiv 2608.07555) reports +9.8
  points of best-of-K optimism at K = 27 on LIBERO-10 with SmolVLA, but from a Monte-Carlo
  replay calibrated to one campaign's noise, on a single training lineage, with selection
  and holdout split by task rather than by start seed, and nothing released. The
  operator-variance-audit repo (2026) and Barannikov 2024 (see Prior work) measure
  evaluation noise on robomimic and PushT but not selection inflation. No PushT study of the
  winner's curse was found.
- It is the method this portfolio already uses, pointed at learned policies: predictions
  written before the run, positive and negative controls, planted faults that the checker
  must catch, and claims scoped to what the evidence supports.

## Design

### Disjoint seed ranges, never mixed

| range | purpose | Reacher-v5 | PushT |
|---|---|---|---|
| pilot | set the success rule from controls only; never reported | 0-99 | not needed (PushT defines success) |
| training | seeds for training runs; only the first reset of each training environment is seed-indexed (SB3 gives environment rank i the seed plus i), later start states come from the environment's RNG stream | SB3 seeds 100000, 100001, 100002 | LeRobot seeds 100000 + k |
| in-training eval | LeRobot's periodic evaluation, seeded from the training seed | not used | disabled (`eval_freq` 0); every evaluation runs through this project's driver on the ranges below |
| reporting | what the common protocol reports on; LeRobot's default is 1000-1049 | 1000-1499 | 1000-1499 |
| selection | the separate range a corrected protocol would pick on | 2000-2999 | 2000-2999 |
| oracle | truth: thousands of episodes per checkpoint | 10000-14999 | 10000-11999 |

Every per-episode record carries its seed. A test asserts that the seeds logged for pilot,
reporting, selection and oracle do not intersect each other or any training seed plus
environment rank. Training episodes after the first reset are not seed-indexed, so the test
cannot cover them; this is accepted because the pilot range only sets the radius from the
two controls, and training never uses the radius.

### Two selection protocols, emulated against oracle truth

- Protocol A (shared seeds, the default): evaluate each of K candidate checkpoints on n
  reporting seeds, pick the best, report its score on those same seeds.
- Protocol B (split seeds): evaluate each candidate on n selection seeds, pick the best,
  report its score on n reporting seeds.

Two quantities per selection event: reporting bias = reported score minus the picked
checkpoint's oracle truth; selection regret = oracle truth of the best candidate minus
oracle truth of the picked one. Protocol B's reporting bias is zero in expectation by
construction, so the A-versus-B comparison shows the reporting bias; the regret shows what
each protocol's pick costs. Protocol B is also compared at matched budget: A at 2n against B
at n selection plus n reporting.

Candidate sets: for each K, the last K checkpoints of a run (the early-stopping case) as
primary; all K-subsets as secondary. The primary estimand is the mean reporting bias over
resamples at n = 50 and K = 5, reported per training seed with its resampling interval, and
across seeds as the mean and range.

Two measurements of each quantity: the single literal run on the default seeds (what
LeRobot would print), and the distribution over 1,000 resampled seed sets drawn with
replacement from the oracle logs, for n = 10, 50, 500 and K from 2 up to the number of
checkpoints saved. The same drawn seed set scores all K checkpoints (Protocol A); Protocol B
draws two independent sets, one for selection and one for reporting. Truth for the picked
checkpoint is its full oracle mean. If a without-replacement variant is ever reported, truth
is the mean over the oracle episodes not in the subset, because otherwise reported minus
truth is shrunk by (1 - n/N), which is 0.75 at n = 500 on a 2,000-episode oracle. Caveat,
stated in the report: the resampled distribution is conditional on the realized oracle
rollouts, which are one draw of the policy's action noise per seed.

### S3 replicate sub-study (stretch)

On two training seeds: two checkpoint pairs per seed (one near-equal pair, one with a gap
above 10 points), 200 oracle seeds x 5 repeated rollouts per checkpoint under each RNG
condition (action-noise generator seeded from the start seed and shared across the pair;
and seeded independently). Variance components between seed and within seed come from the
repeats; the pairing gain is 1 - Var(paired difference) / Var(unpaired difference) under
each condition. About 16,000 episodes. The main oracle runs once, with the noise generator
seeded from the start seed, so every oracle episode is reproducible.

### Metrics

- Primary: binary success at episode end (Reacher: final fingertip-target distance below the
  pre-registered radius; PushT: the environment's own success flag).
- Secondary: final distance (Reacher), PushT's continuous max-coverage score, episode
  return, control effort, steps until first success.
- Intervals: Wilson for proportions (a binomial interval that behaves near 0% and 100%,
  unlike mean +/- 1.96 SE); percentile bootstrap and seed-clustered bootstrap for means;
  exact McNemar for paired binary comparisons; a betting e-process for the sequential paired
  test. Each is written from its formula, tested on synthetic data with a planted answer,
  then cross-checked against rliable and scipy before use.

### A/A calibration rule

A checkpoint is compared with itself on two disjoint n-episode draws, 1,000 times, for
every test used. Pass rule: the observed false-difference rate is at most
alpha + 2 * sqrt(alpha * (1 - alpha) / 1000), which is 0.064 at alpha = 0.05. A rate below
0.036 is reported as conservative, not as a pass or a fail. For the e-process, the wealth is
checked after every pair up to the full horizon n, and a replication counts as a false
decision if it ever reaches 1/alpha. With 1,000 replications this check can detect a true
rate above about 0.07; it cannot certify a rate exactly at alpha.

### Planted harness bugs

Each is a deliberate fault with a pytest that asserts which check catches it: VecNormalize
statistics not loaded at evaluation (VecNormalize is SB3's running observation and reward
normalizer; its statistics must be saved with the model and loaded at evaluation, or the
policy sees inputs on the wrong scale); evaluation seeds overlapping training or pilot seeds;
success scored at the wrong timestep; stochastic actions where deterministic ones were
meant; a sign-flipped joint; NaN actions; a policy that parks just outside the success
radius; truncated episodes counted as successes. Five in the pilot is enough; no tooling
around them beyond the tests.

## Timeline (12 weeks at 6-8 hours per week)

| weeks | work | done when |
|---|---|---|
| 1 | PREREG v0 frozen (radius rule, seed ranges, metrics, alpha). The random control and the IK+PD positive control on pilot seeds; radius set by the rule. Random baseline on reporting seeds 1000-1099 with a Wilson interval written from the formula. SB3 steps-per-second on the M5 committed. | first committed, pre-registered result with an interval |
| 2 | PPO on 3 training seeds, checkpoint every 10% of training. Per-episode logger. Wilson and bootstrap intervals with planted-answer tests. First planted-bug policies. | PPO beats the random control with non-overlapping 95% intervals on at least 2 of 3 seeds |
| 3 | Reacher pilot audit: oracle per checkpoint, Protocol A vs B reporting bias and regret, rank flips against n, A/A calibration within the rule above, five planted bugs caught. One-page pilot note in the README. | go/no-go for GPU spend |
| 4 | The PPO-update walkthrough notebook (GAE, probability ratio, clipped surrogate, value loss) matched to SB3's logged values. LeRobot seed handling re-confirmed on the pinned commit and recorded with file and line numbers. PushT running locally. GPU account funded with a hard cap. | written up |
| 5 | Public `lerobot/diffusion_pusht` evaluated on 500 reporting-range episodes on a rented GPU. Episodes per GPU-hour measured. Pinned recipe verified by a short gate run. PREREG v1 for PushT frozen, with a numeric prediction per hypothesis. | our 500-episode Wilson interval overlaps the card's own, 61.1-69.4% (327 of 500) |
| 6-7 | 5 Diffusion Policy training seeds (8 if budget allows), checkpoint every 25k steps to 200k, in-training evaluation disabled. A small PyTorch behavior-cloning policy written by hand as the cheap, deliberately weak comparison arm, with many seeds. Oracle evaluation on the disjoint ranges, binary and coverage metrics. | per-episode logs versioned; cost per 1,000 episodes logged |
| 8 | The audit against oracle truth: reporting bias and regret against K and n for both protocols, compared with the analytic order-statistics prediction and with Yu 2026; rank-flip probability against estimated gap and n; S2; exact McNemar written and calibrated. Failure taxonomy from rollout videos. | a standalone study exists even if nothing else ships |
| 9-10 | Robustness: more seeds if budget allows, sensitivity to oracle size, the null-results section, the BC comparison. Stretch: the betting e-process and the S3 replicate sub-study. If the shared-seed default still holds on the pinned commit and inflation is material: a LeRobot issue with the measured number, then a small PR adding a separate selection seed range (following whatever LeRobot's CONTRIBUTING asks at that time, checked then, and with AI assistance disclosed). | drafts ready for Di's go |
| 11-12 | A 6-8 page technical report, figures first, with a claims-scope table. A README whose first line states the specific finding. The oracle-rollout dataset on the Hugging Face Hub with a datasheet. One command that regenerates every number. The "why this and not that" log. | shipped |

Phase 2, only after week 12 and only if the simulation study shipped: a personal SO-101
follower arm running the same harness, as a protocol demonstration, labeled as such.

The original 7-step plan's observation-noise study is dropped from the main line; robustness
to observation noise is covered by existing work and is not this project's question.

## Gates and kill criteria

- End of week 2: if PPO does not beat the random control with non-overlapping 95% intervals
  on at least 2 of 3 seeds after one debugging round, stop expanding scope and fix
  fundamentals first.
- End of week 3: if any test fails the A/A calibration rule, or any planted bug goes
  undetected, no GPU money is spent until it is fixed.
- Week 5: if our 500-episode Wilson interval does not overlap 61.1-69.4% after one extra
  week of debugging, drop training; audit public checkpoints plus the hand-written BC policy
  only, and say so.
- Week 5 recipe gate: if the pinned recipe's short run misses the pre-set level, do not
  launch the seeds.
- Compute: stop any single run over 15 GPU-hours or $25; stop all GPU spend at $200.
- Truth stability: if two independent oracle runs of one checkpoint disagree beyond their
  intervals, that is evaluation nondeterminism. Fix it within one week or publish the Reacher
  study plus the nondeterminism finding.
- Week 8 understanding gate: if Di cannot explain the PPO clip, selection inflation, and
  why Protocol B's reporting bias is zero without notes, freeze new scope for two weeks.
- Release gate: nothing ships unless every number regenerates from the released data with
  one command.

## Who writes what

Di writes, and must be able to explain every line of: PREREG.md, DEVIATIONS.md and the
radius rule; the episode runner and per-episode logger (not SB3's `evaluate_policy`); the
IK+PD positive control; the PPO-update walkthrough; the statistics module and its
planted-answer tests; the planted-bug policies; the PushT evaluation driver that enforces
the seed ranges; the behavior-cloning policy and its training loop; the resampling audit
and analysis notebook, including the order-statistics prediction for the primary
hypothesis; the conclusions and the report.

Order, if time runs short: (1) radius rule, task1 with its Wilson interval, IK+PD;
(2) episode runner and logger; (3) Wilson and bootstrap with planted-answer tests; (4) PPO
on 3 seeds through SB3; (5) five planted bugs; (6) the resampling audit. Below the line and
not needed for the week-3 gate: the PPO-update walkthrough (week 4), exact McNemar (week 8),
the betting e-process and S3 (weeks 9-10, stretch; if cut, A6 and S3 are dropped and the
report says so).

Claude Code tutors, quizzes and reviews diffs. It writes only glue, and only after Di's
core version exists: the pytest scaffold for each planted bug after Di writes the bug
policy, plotting and figure styling, cloud launcher scripts, LeRobot version pinning, CI
configuration, test-case suggestions, dataset-card boilerplate. Di reads every scaffolded
line before committing. Diffusion Policy itself is trained through LeRobot's CLI on the
pinned recipe; Di reads `DiffusionConfig` and `select_action` closely enough to explain
action chunks, the observation horizon and denoising, and does not rewrite them.

Rules: Claude never writes a core module first. AI assistance is disclosed in the README and
in any LeRobot issue or PR. Di keeps a running "why this and not that" log. Every public
action (issue, PR, post, dataset upload) needs Di's explicit go.

## Compute and cost

Mac (Apple M5, 24 GB), $0: all Reacher work (SB3 recommends CPU for MLP policies), PushT
development and small evaluations, the BC policy (after a CPU-vs-MPS parity check), all
statistics and plots.

Rented NVIDIA GPU, for everything Diffusion Policy. The arithmetic:

- Training: 5 seeds x 200k steps. An old model-card sentence said about 7 hours per run on
  an RTX 3090; it was later removed, so treat it as a rough anchor: about 35 GPU-hours,
  $12-53 at the $0.34-1.50 per hour quoted for RunPod and Hugging Face Jobs (to be
  re-checked at week 3).
- Evaluation: 5 seeds x 8 checkpoints x (2,000 oracle + 1,000 selection + 500 reporting)
  = 140,000 episodes, plus about 16,000 if S3 runs. On the M5 one episode took 107 s (see
  the facts below), which would be 4,160 hours; the $200 cap at about $1 per GPU-hour
  requires under about 5 s per episode from batched evaluation on the rented GPU, measured
  in week 5 before any seed is launched. Pre-committed fallback: the last 6 checkpoints x
  1,000 oracle episodes, 30,000 in all.
- Spend is a guess until week 5 measures episodes per GPU-hour. Hard cap $200; stop any
  single run over 15 GPU-hours or $25. Every job gets an explicit timeout.

No GPU purchase. No hardware purchase before week 12.

## Prior work and positioning

This is not a new statistical method. It is a known-truth measurement on a community
imitation-learning benchmark, with a released dataset and a possible upstream fix. Credit
plainly:

- Yu 2026, "You Don't Need To Stay in The Loop: An Agentic Robotics Loop for Robot-Policy
  Improvement", arXiv 2608.07555. Quantified best-of-K optimism (+9.8 points at K = 27) and
  compared e-processes with McNemar under peeking, by Monte-Carlo replay, on LIBERO-10 with
  SmolVLA. This project replicates the winner's curse on a different policy class and
  benchmark against an analytic prediction, and adds high-n per-checkpoint truth across
  training seeds, a seed-range split, paired testing with a stochastic policy, and a
  released dataset.
- operator-variance-audit, github.com/imjbassi/operator-variance-audit (GitHub user
  imjbassi; author name and year to be taken from its Zenodo record). A pre-registered
  robomimic audit (138 BC-RNN runs; 46 checkpoints x 500 rollouts; 50- and 500-rollout
  success correlate at r = 0.56). Final checkpoint only; no selection study. Repo plus
  Zenodo, not a paper.
- Barannikov, Hugging Face blog, Dec 2024, "Is using a validation set useful for end-to-end
  learning in robotics?". Diffusion Policy on PushT, 3 seeds, 50 vs 500 episodes, finding 50
  too noisy; no intervals, no selection study.
- rliable (Agarwal et al. 2021) for interval-based reporting; Colas et al. 2019 for
  empirical false-positive and power measurement of RL comparisons; Kress-Gazit et al. 2024
  for the policy-evaluation reporting checklist. These three were not re-checked this
  session; verify the citations before the report.

The model card's published number moved from 63.8% to 65.4% between card versions (about
0.75 binomial standard errors at n = 500). Re-measuring it is measuring a common practice,
not criticism of the authors, and the report says so.

## Risks

- Beginner overload at 6-8 hours per week. Mitigations: the priority order above; the real
  arm is outside the 12 weeks; Reacher is capped at 3 weeks; Diffusion Policy trains through
  the LeRobot CLI; week 8 is a complete stopping point.
- Recipe drift. Current LeRobot defaults no longer match the published checkpoint (see
  facts). Mitigation: pin the recipe to the checkpoint's `config.json` and a fixed LeRobot
  commit, and pass a short-run gate before launching seeds.
- Checkpoint format drift. The public checkpoint needs a migration step in current LeRobot,
  and the migration script fails on its config (see facts). Mitigation: the workaround is
  recorded; pin the LeRobot version that works.
- Oracle cost is unmeasured on a GPU. Mitigations: measure in week 5; the pre-committed
  fallback; never reduce denoising steps without labeling the result as a different policy.
- Few independent selection events: checkpoints within one run are correlated, so 5 seeds
  give only 5 selection events. Mitigations: report per seed; predict the primary
  hypothesis from the measured checkpoint correlation; add seeds if budget allows.
- The oracle is itself an estimate (about +/-2 points at n = 2,000 for one checkpoint, more
  for a difference). Mitigation: gaps are estimated with their paired standard error from
  the seed-aligned logs, and pairs with |gap| below 2 SE are excluded from rank-flip claims.
- Scoop risk in a crowded genre (the operator-variance-audit repo and Yu 2026). Mitigations:
  post the interim Reacher note at week 3 and the PushT result at week 8; release the
  dataset so the work stays citable.
- MPS numerics: keep Diffusion Policy on CUDA; run a CPU-vs-MPS parity check before any
  training on the Mac.
- Separation from employment: personal laptop and accounts only; no employer hardware, data
  or code; check the employment agreement's side-project terms before publishing.

## Facts checked on 2026-10-18

LeRobot seed handling (installed lerobot 0.6.1; the same lines exist on main at 146d847,
2026-10-16):

- `lerobot/scripts/lerobot_train.py` line 730: the periodic in-training evaluation passes
  `start_seed=cfg.seed`.
- `lerobot/scripts/lerobot_eval.py` line 801: the standalone evaluation passes
  `start_seed=cfg.seed`; lines 531-532 seed episodes consecutively from `start_seed`.
- `lerobot/configs/train.py` line 96: `seed: int | None = 1000`, with the comment that it is
  used for training "AND for the evaluation environments".
- `lerobot/configs/eval.py` line 39: `seed: int | None = 1000`.
- `lerobot/configs/default.py` line 95: `n_episodes: int = 50`.
- No option exists for a separate selection seed range.

Public checkpoint `lerobot/diffusion_pusht`:

- Card: 65.4% success over 500 episodes, which is 327 of 500; its Wilson 95% interval is
  61.1-69.4%. Earlier card versions in 2024 said 63.8% and carried the "about 7 hours on an
  RTX 3090" sentence, since removed.
- `config.json`: horizon 16, n_action_steps 8, n_obs_steps 2, crop_shape [84, 84],
  num_train_timesteps 100 (DDPM). `train_config.json`: 200,000 steps, batch 64, seed 100000,
  dataset `lerobot/pusht`; the checkpoint is from step 175k, trained with LeRobot at 3c0a209.
- Current LeRobot main defaults differ: crop_shape None (PR #3046, Feb 2026) and
  horizon 64 / n_action_steps 32 (PR #3202, May 2026). The card's published training command
  passes no overrides; at 3c0a209 those values were the defaults. Re-running it on current
  main without overrides would not reproduce the recipe.

This Mac (Apple M5, 24 GB, macOS):

- `gym-pusht` 0.1.8 needs Python pinned to 3.12 (uv otherwise picks 3.14, for which pygame
  has no wheel). Random policy: 529 steps/s including rendering; obs `pixels` 96x96x3 plus
  `agent_pos`; 300 steps per episode.
- `lerobot[pusht,diffusion]` 0.6.1 installs with torch 2.11.0 and MPS available.
- `lerobot-eval` needs `--eval.use_async_envs=false` here (cause not diagnosed).
- The public checkpoint predates LeRobot's processor format. The bundled
  `migrate_policy_normalization.py` writes the processor files, then fails on the config
  ("Couldn't encode 84") and leaves an empty `config.json`. Copying the original
  `config.json` and `model.safetensors` from the Hub snapshot next to the generated
  processor files makes it load.
- With that patch, `lerobot-eval` on 4 episodes (seeds 1000-1003) on MPS, batch 4: 4 of 4
  successes, 107 s per episode. The Mac is fine for development and not for the oracle.

Gymnasium 1.4.0 Reacher-v5: the control-cost weight is 1 in the code and 0.1 in the
docstring; the reward is computed after the physics step.

## Unverified, to be measured or checked

- SB3 and MuJoCo throughput on the M5 (week 1).
- RL Zoo's published PPO settings for Reacher, including the step count (week 2).
- Diffusion Policy training time on current LeRobot per 200k steps (week 5).
- Oracle evaluation cost per 1,000 PushT episodes on a rented GPU (week 5).
- Whether `gym-pusht` has a state or keypoint observation variant suitable for a cheap BC
  policy, and whether it behaves identically on macOS and Linux.
- LeRobot's contribution and AI-disclosure requirements at the time of any PR.
- The three reporting-standard citations above.
- Cloud GPU prices (re-check at week 3).

## Deliverables

1. `robot-policy-evals`: the harness, statistics module, planted-bug tests, pre-registration
   history, per-episode logs, and one command that regenerates every figure.
2. The oracle-rollout dataset on the Hugging Face Hub with a datasheet.
3. A 6-8 page technical report with a claims-scope table.
4. If warranted: a LeRobot issue and PR for a separate selection seed range.
5. The "why this and not that" log.
