#!/usr/bin/env bash
set -euo pipefail
python -m experiments.batch --mode full --steps 1000000
python -m experiments.random_baseline --episodes 100
python -m experiments.summarize
