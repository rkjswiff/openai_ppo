# PPO 2017 — Table 1 reproduction (based on your uploaded OpenAI Baselines)

**What it does:** Runs the 11 surrogate-objective configurations from Section 6.1 of [Schulman et al., 2017](https://arxiv.org/pdf/1707.06347), across seven MuJoCo tasks and three random seeds (231 runs). It records raw training episode returns, computes averages over the last 100 completed training episodes, estimates a random-action baseline, and generates CSV/PNG evidence for a presentation.

**Important:** This is a reproduction **implementation**, not a completed reproduction. No neural-network training was run in the development environment; only static/logic tests were run. Running the commands below will produce *your own* results. **Do not describe the original paper's 0.82 as the result of executing this code.**

## 1. Which environment should I use?

Your uploaded Baselines is an older TensorFlow-1 implementation. The **recommended runnable option** below is a modern compatibility port using **TensorFlow 2.15 in TF1 graph mode + Gymnasium MuJoCo-v4**. It preserves the PPO1 architecture, losses and optimization settings, but **does not reproduce the historical `*-v1` MuJoCo physics/observation/reward definitions exactly**.

For strict fidelity to the paper you need historical MuJoCo `*-v1`, the matching Gym/MuJoCo versions, the original random reference and preferably the precise historical code revision. See **Section 10** for the optional legacy path and limitations. Do not expect numerically identical scores from the modern environment.

### Recommended: Docker on Linux / macOS / Windows

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (or Docker Engine on Linux). Unzip the downloaded archive and open a terminal **inside the extracted folder**, the folder containing `Dockerfile.table1` and `RUN_ME_FIRST.md`.

```bash
# Mac M1/M2/M3: run Linux x86-64 in emulation (CPU only; slower).
docker build --platform linux/amd64 -t ppo-table1 -f Dockerfile.table1 .

# Create persistent results on your host
mkdir -p results

docker run --rm -it --platform linux/amd64 \
  -v "$(pwd)/results:/work/results" \
  ppo-table1 bash
```

All commands below are executed **inside the container** unless noted otherwise. **Keep the `-v` volume mount** or the results will disappear when the container exits.

### Alternative: Ubuntu Linux, without Docker

Use **Python 3.10 (x86-64)**. The following is *not* the native-Mac-M2 path.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-table1.txt
```

Do **not** run `pip install -e .` from the old `setup.py`: its historical gym and TensorFlow requirements conflict with the modern compatibility environment. Running from the extracted project directory is sufficient.

## 2. Verify the setup before training

```bash
python -m pytest -q tests/test_objective_math.py tests/test_reporting.py
python -m experiments.preflight --backend modern
```

The first command checks 11 configurations, clipping/penalty arithmetic, and aggregation behavior. The second imports TensorFlow and MuJoCo, then **actually resets and steps all 7 environments**. Stop here if it fails; don't start a 231-run batch with a broken environment.

## 3. Test one actual training run (quick smoke test)

```bash
python -m experiments.train \
  --env Hopper-v1 \
  --objective clip_0.2 \
  --seed 0 \
  --steps 8192
```

Despite the `Hopper-v1` **paper label**, the modern adapter uses `Hopper-v4`. Check the run output under:

```
results/modern/Hopper-v1/clip_0.2/seed_0/
  config.json             exact configuration and environment actually used
  progress.csv            PPO training metrics, KL, clip fraction, beta
  episodes.monitor.csv    actual episode returns (one per completed episode)
  result.json             mean of up to the last 100 completed episodes
  log.txt                 text training logs
```

A short smoke run often produces **fewer than 100 complete episodes**. It is **not** sufficient for comparing the original paper's Table 1 numbers.

## 4. Optional: Run a short four-objective smoke batch

Use this in a **fresh results directory** because the batch runner protects against mixing time budgets for identical job names.

```bash
python -m experiments.batch --mode smoke --steps 8192 --out smoke_results
python -m experiments.random_baseline \
  --envs Hopper-v1 --episodes 20 --out smoke_results
python -m experiments.summarize \
  --allow-partial --out smoke_results
```

This creates a **partial, non-comparable** report. It is meant only for validating that code, logging, and plotting work, not for estimating the published normalized scores.

## 5. Full Experiment 1 / Table 1 — all 11 settings × 7 tasks × 3 seeds

To check the work list **without running it**:

```bash
python -m experiments.batch --mode full --dry-run
```

To execute all 231 training jobs sequentially, each for 1 million timesteps:

```bash
python -m experiments.batch --mode full --steps 1000000
```

**Compute warning:** This is ~231,000,000 MuJoCo timesteps, plus 10 policy optimization epochs per 2048-step rollout. On an Apple Silicon laptop running emulated x86 Docker, this could take a very long time (potentially days or more). Use a native x86-64 Linux CPU/GPU machine or cloud VM if available. The script runs sequentially and **skips completed runs** on restart. A GPU is optional; the code can use CPU. Start with `--limit 1` to benchmark runtime.

Available objective IDs:

| CLI objective | PPO objective / setting | Published normalized score |
|---|---|---:|
| `unclipped` | No clipping, no penalty | -0.39 |
| `clip_0.1` | Clipping ε = 0.1 | 0.76 |
| `clip_0.2` | Clipping ε = 0.2 | 0.82 |
| `clip_0.3` | Clipping ε = 0.3 | 0.70 |
| `adaptive_0.003` | Adaptive KL, target 0.003 | 0.68 |
| `adaptive_0.01` | Adaptive KL, target 0.01 | 0.74 |
| `adaptive_0.03` | Adaptive KL, target 0.03 | 0.71 |
| `fixed_0.3` | Fixed KL, β = 0.3 | 0.62 |
| `fixed_1` | Fixed KL, β = 1 | 0.71 |
| `fixed_3` | Fixed KL, β = 3 | 0.72 |
| `fixed_10` | Fixed KL, β = 10 | 0.69 |

The parameters held fixed from Table 3: 2048 timesteps per rollout, 10 epochs, Adam learning rate 0.0003, minibatch size 64, γ = 0.99, GAE λ = 0.95; separate 64×64 tanh policy/value networks; no entropy bonus.

## 6. Estimate the random-action reward for normalization

After training (or in a separate terminal), run:

```bash
python -m experiments.random_baseline --episodes 100
```

This evaluates 100 episodes of `env.action_space.sample()` per environment and writes `results/modern/random_baselines.csv`. This is an **operational approximation** of the paper's random-policy reference, not a recovered 2017 reference constant. Scores are shifted by this estimate and divided by the maximum observed reward for the same environment across the 33 trained runs (11 settings × 3 seeds).

## 7. Generate the comparison table, graphs and raw data

```bash
python -m experiments.summarize
```

When all 231 runs have succeeded, each has 100+ episodes, all use a million timesteps, and random baselines exist, the script generates:

```
results/modern/report/
  table1_comparison.csv     paper figures beside your actual aggregate
  run_scores.csv            per-env, per-seed, raw and normalized scores
  report_metadata.json      protocol completeness / limitations
  Hopper_v1_learning_curve.png  (and one PNG per other task)
```

The summarizer **rejects partial runs by default**, preventing an accidental false claim of reproducing the entire benchmark. For a four-variant or single-environment report use `--allow-partial`, but clearly mark the scores as partial and not comparable with the paper's normalized scores.

## 8. How normalization works

For each environment E, and every setting/seed pair i:

```
R_i = mean reward of the LAST 100 completed training episodes
R_random = mean return of 100 episodes with randomly sampled actions
R_best = max(R_i) over all 11 settings × 3 seeds, within environment E
normalized_i = (R_i - R_random) / (R_best - R_random)
```

Finally, average all 21 normalized values for one setting (7 environments × 3 seeds). The denominator is computed using **this reproduction's runs**, not the published paper's raw run values. You cannot simply take the raw Hopper score and compare it to 0.82.

The paper does not publish all individual episode logs/random-policy reference values; exactly matching its 2017 normalized scores cannot be guaranteed with the information provided.

## 9. Evidence to show your BITS DRL assignment evaluator

- `config.json`: screenshots of environment mapping, parameters, and exact objective ID.
- `baselines/ppo1/pposgd_simple.py`: show the clipped, unclipped, and KL loss branches and the after-update beta adaptation.
- `episodes.monitor.csv` and `progress.csv`: actual training evidence and KL/beta logs.
- `results/modern/report/run_scores.csv`, `table1_comparison.csv`, graphs: your computed results.
- Include names/roles of contributors and the [paper](https://arxiv.org/pdf/1707.06347) and [Baselines original repository](https://github.com/openai/baselines). Be transparent that MuJoCo-v4 is different from historical MuJoCo-v1 if using the modern Docker method.

**Do not copy original paper numbers into the 'our result' column.** These are intentionally blank until real training completes.

## 10. Optional historical MuJoCo-v1 path (advanced)

Pass `--backend legacy` to the training/batch/random/summarize commands to instantiate the **actual `*-v1` Gym environment names**, rather than mapping them to Gymnasium-v4. This requires installing a historical compatible **Python 3.7 + TensorFlow 1.15 + Gym 0.15.x + mujoco-py 2.0 + MuJoCo 2.0** stack on **Linux x86-64**, including its native simulator binaries, library paths, and licensing configuration (depending on your chosen distribution). That environment is **not included or validated** in this zip; the user must provision it separately. The old Baselines package's default setup is not a reliable one-command install on a modern Mac.

Legacy path command after setting up historical dependencies:

```bash
python -m experiments.train --backend legacy --env Hopper-v1 \
  --objective clip_0.2 --seed 0 --steps 1000000
```

Even the historical path is **not a mathematically exact reproduction guarantee**: our runner additionally fixes a timestep-count discrepancy in the uploaded PPO1 code and makes slight numerical stability changes. Pin dependency/library versions, capture the environment config and compare your results critically.

## 11. Troubleshooting

- `No module named tensorflow` / `gymnasium` / `mujoco`: install `requirements-table1.txt` inside the proper Python environment, or use Docker.
- `No module named baselines`: run Python in the extracted package root (`/work` inside Docker).
- `Hopper-v1 not found`: you are using native historical Gym without MuJoCo-v1 installed. Use **modern** backend or install the old simulator.
- `ModuleNotFoundError: gymnasium`: expected for modern backend if requirements weren't installed; backend legacy doesn't require Gymnasium.
- Installation fails on Apple Silicon: use the recommended Linux/x86-64 Docker image with `--platform linux/amd64`, or a cloud Linux x86-64 VM.
- `Not a full paper protocol`: run all jobs for 1M steps, collect at least 100 episodes per run, and create random baseline CSV; otherwise pass `--allow-partial` for clearly labeled preliminary results.
- `Existing result settings differ`: use a different `--out` directory for short-vs-long runs instead of overwriting evidence.

## 12. What was modified from your attachment?

- Added switchable unclipped / clipping / fixed KL / adaptive KL objective to `baselines/ppo1/pposgd_simple.py`.
- Logged KL divergence, clip fraction, beta used/next; updated beta using the paper's rule after every policy update.
- Corrected the trainer's budget counter to measure actual sampled environment timesteps (rather than the sum of lengths of completed episodes).
- Ported the PPO1 dependency chain to TensorFlow 2.15 TF1 compatibility imports.
- Added one-worker MPI compatibility, Gymnasium-to-Gym MuJoCo-v4 adapter, controlled experiment runners, raw-reward aggregation, random-baseline evaluation, CSV/PNG export, smoke tests, and Docker environment files.
- **All non-PPO algorithms in the uploaded Baselines repository are left unchanged and are not tested by this project.**
