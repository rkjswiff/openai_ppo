#!/usr/bin/env bash
set -euo pipefail
python -m pytest -q tests/test_objective_math.py tests/test_reporting.py
python -m experiments.batch --mode smoke --steps 8192
python -m experiments.random_baseline --envs Hopper-v1 --episodes 20
python -m experiments.summarize --allow-partial
