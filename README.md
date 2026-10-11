# PPO Experiment 1 — Reproduction Guide

This repository is a **partial reproduction of Experiment 1 (Table 1)** from *Proximal Policy Optimization Algorithms* (Schulman et al., 2017). It builds on the existing [OpenAI Baselines PPO1 implementation](https://github.com/openai/baselines/tree/master/baselines/ppo1) and compares **11 policy-loss objectives** using the same PPO training machinery.

Our assignment experiment uses **3 environments × 11 objectives × 3 random seeds = 99 training runs**, with **500,000 requested environment timesteps per run**.

## 1. Setup

Run the commands below **from the repository root** (the directory containing `baselines/` and `experiments/`).

Python **3.10** is recommended. The pinned dependencies are in `requirements-table1.txt`. A Linux x86-64 environment is the documented target; the project has also been run on an Apple Silicon Mac with a compatible local environment.

```bash
conda create -n ppo-repro python=3.10 -y
conda activate ppo-repro
python -m pip install -r requirements-table1.txt
```

MuJoCo must be supported by the machine. On Apple Silicon, TensorFlow and MuJoCo installation may require platform-specific adjustments to the pinned dependencies.

Verify the installation and environment setup:

```bash
python -m experiments.preflight
```

## 2. Run the experiments

### A. Quick smoke test

```bash
python -m experiments.batch --mode smoke --steps 8192 --dry-run
python -m experiments.batch --mode smoke --steps 8192 --out results_smoke
```

`smoke` runs 4 configurations on Hopper (unclipped and three clipping thresholds), with seed 0. The separate `results_smoke` directory prevents conflicts with the main experiment.

### B. Full assignment experiment — 99 runs

The environments are defined in `experiments/experiment.py`:

- `Hopper-v1`
- `HalfCheetah-v1`
- `InvertedDoublePendulum-v1`

Run:

```bash
python -m experiments.batch --mode full --steps 500000 --dry-run
python -m experiments.batch --mode full --steps 500000
```

The dry run should show **99 jobs** for the current configuration. Each run trains one environment–objective–seed combination.

**Important:** `--mode full` uses the environments currently listed in `experiments/experiment.py`; it does not automatically select all seven environments in the paper. The batch runner skips an existing completed run when its timestep budget and backend match. If an existing result has different settings, it stops with an error. Use a different `--out` directory to keep multiple budgets.

### C. Run one configuration

```bash
python -m experiments.train \
  --env Hopper-v1 \
  --objective clip_0.2 \
  --seed 0 \
  --steps 500000
```

Objective names are defined in `experiments/experiment.py`.

## 3. Generate random baselines and summary

After training, estimate the random-policy rewards (100 episodes **per environment**, not 100 training steps):

```bash
python -m experiments.random_baseline --episodes 100
```

Generate the comparison CSVs and learning-curve images:

```bash
python -m experiments.summarize --allow-partial
```

Use `--allow-partial` because this assignment intentionally uses fewer environments and timesteps than the full paper protocol.

Outputs are saved under:

```text
results/
└── modern/
    ├── random_baselines.csv
    ├── Hopper-v1/
    │   └── clip_0.2/
    │       └── seed_0/
    │           ├── config.json
    │           ├── progress.csv
    │           ├── episodes.monitor.csv
    │           ├── provenance.json
    │           └── result.json
    └── report/
        ├── run_scores.csv
        ├── table1_comparison.csv
        ├── report_metadata.json
        └── *_learning_curve.png
```

The final result uses the **mean reward from the last 100 completed episodes** of each run. The summarizer normalizes this against a random policy and the best observed result for each environment, then averages scores across the available runs. **Our normalized scores are not numerically comparable to the paper's scores** because the environments, training budget and normalization reference differ.

## 4. What we customized

| File | Change / purpose |
|---|---|
| `baselines/ppo1/pposgd_simple.py` | Extended the original PPO1 clipped policy loss to select **unclipped, clipped, fixed KL penalty and adaptive KL penalty** objectives; added KL penalty adaptation and related logging. The PPO trajectory collection, advantage estimation and optimization loop are reused. |
| `experiments/experiment.py` | Defines the three selected environments, 11 objective settings, paper reference scores and seeds `(0, 1, 2)`. |
| `experiments/train.py` | New training entry point; passes objective parameters and training hyperparameters into `pposgd_simple.learn`, then saves configuration and run results. |
| `experiments/batch.py` | Runs combinations of environments, objectives and seeds; supports dry runs and skipping matching completed jobs. |
| `experiments/env_adapter.py` | Adapts modern Gymnasium/MuJoCo to the older Baselines interface. The `-v1` experiment labels map to `-v4` environments with the modern backend. |
| `experiments/random_baseline.py` | Evaluates a random policy for reward normalization. |
| `experiments/summarize.py` | Aggregates run rewards, creates normalized Table 1 comparisons and learning-curve images. |
| `experiments/preflight.py` | Checks that the required environment and software setup works before training. |

### How the objective modification works

The original Baselines PPO1 already implemented the clipped surrogate. We retained the PPO trainer and added objective selection in `pposgd_simple.py`:

```python
ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac))
surr1 = ratio * atarg
surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg

if objective == 'clip':
    pol_surr = -tf.reduce_mean(tf.minimum(surr1, surr2))
elif objective == 'unclipped':
    pol_surr = -tf.reduce_mean(surr1)
else:
    pol_surr = -tf.reduce_mean(surr1) + kl_beta_ph * meankl
```

The **loss changes the gradients** used by the optimizer; the same underlying PPO learning loop trains the policy for every configuration. For adaptive KL, the penalty coefficient is adjusted according to the measured KL divergence.

The 11 configurations are: `unclipped`, `clip_0.1`, `clip_0.2`, `clip_0.3`, `adaptive_0.003`, `adaptive_0.01`, `adaptive_0.03`, `fixed_0.3`, `fixed_1`, `fixed_3`, and `fixed_10`.

## 5. Training settings and scope

| Setting | Value |
|---|---|
| Environments | Hopper, HalfCheetah, InvertedDoublePendulum |
| Policy objectives | 11 |
| Seeds | 0, 1, 2 |
| Timesteps requested per run | 500,000 |
| Trajectory batch (`horizon`) | 2,048 steps |
| Optimization epochs | 10 |
| Minibatch size | 64 |
| Learning rate | 0.0003 |
| Discount factor (`gamma`) | 0.99 |
| GAE (`lambda`) | 0.95 |
| Backend | `modern` (Gymnasium MuJoCo v4) |

The original paper studied more environments and other comparisons. This repository reproduces **only the Table 1 objective comparison at reduced scale**, not the full research paper. The modern MuJoCo v4 tasks are approximations of the historical v1 environments.

## 6. References

1. Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms*. https://arxiv.org/abs/1707.06347
2. OpenAI Baselines, PPO1 source: https://github.com/openai/baselines/blob/master/baselines/ppo1/pposgd_simple.py
3. Gymnasium MuJoCo documentation: https://gymnasium.farama.org/environments/mujoco/

**Note:** Do not overwrite or merge runs from different training budgets in the same output folder without checking the batch runner's result-file validation.
