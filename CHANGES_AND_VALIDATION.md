# Modification and validation record

Input archive: `/mnt/data/baselines.zip` (sha256 `6967cf67f62051bb84e83a849546af5dd48508968492d6499810a8152f77891b`).

## Original PPO1 files modified
- `baselines/ppo1/pposgd_simple.py`: four surrogate choices (unclipped, clipped, fixed KL, adaptive KL), beta updated after measured policy KL, additional clip fraction/beta logs, fixed actual sampled timestep counter, numeric advantage normalization guard, TF v1 compatibility.
- `baselines/ppo1/mlp_policy.py`: TF v1 compatibility imports.
- `baselines/common/tf_util.py`: TF v1 compatibility imports.
- `baselines/common/distributions.py`: TF v1 compatibility imports.
- `baselines/common/misc_util.py`: TF v1 seed compatibility.
- `baselines/common/mpi_adam.py`, `baselines/common/mpi_running_mean_std.py`, `baselines/common/mpi_moments.py`: use single-worker MPI compatibility helper.
- `baselines/a2c/utils.py`: TF v1 compatibility imports (PPO1's distributions module imports its `fc` helper).

## New files
- `experiments/`: runners, MuJoCo interface, random baseline, score normalization, reports, preflight.
- `tests/`: 5 offline unit tests (equations and score aggregation).
- `Dockerfile.table1` and `requirements-table1.txt`: Linux x86-64 modern runtime specification.
- `RUN_ME_FIRST.md`: user-facing instructions.

## Verified here
- Original user archive unpacked and original PPO1 code inspected.
- Source compiled with `python -m compileall` (static Python syntax).
- All 5 focused unit tests passed (`pytest -q tests/test_objective_math.py tests/test_reporting.py`).
- Full workload enumerated as 231 jobs with `batch --mode full --dry-run`.

## NOT verified here
- No TensorFlow/MuJoCo dependency could be installed in this sandbox due to missing package network access.
- Docker image was NOT built. No actual PPO1 rollout/training completed.
- No results, learning curves, or numerical reproductions are included. They will be generated only when the user runs the project.
- Exact 2017 MuJoCo-v1 simulator equivalence is not achieved by the modern Gymnasium-v4 runtime. Scores may differ substantially.
