#!/usr/bin/env bash
# Idempotent dependency bootstrap for the Cloud Agent environment.
# Creates a project virtual environment and installs the package (with dev
# extras) in editable mode. Safe to run repeatedly.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e ".[dev]"

echo "Fraud intervention environment ready."
