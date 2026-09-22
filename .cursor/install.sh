#!/usr/bin/env bash
# Idempotent dependency bootstrap for the Cloud Agent environment.
#
# Cloud Agent VMs are ephemeral, single-purpose containers, so we install the
# package (with dev extras) into the system interpreter with pip. The
# `--break-system-packages` flag is required on Ubuntu's PEP 668 "externally
# managed" Python; it is safe here because the container is disposable.
# Re-running simply reinstalls the pinned versions, so this stays idempotent.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python3 -m pip install --break-system-packages --upgrade pip
python3 -m pip install --break-system-packages -e ".[dev]"

echo "Fraud intervention environment ready."
