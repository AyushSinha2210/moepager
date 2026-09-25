#!/usr/bin/env bash
# Tick PHASES.md task checkboxes by ID: scripts/tick.sh P2.1 P2.2
set -euo pipefail
for id in "$@"; do
  grep -q -- "- \[ \] $id " PHASES.md || { echo "no open task $id" >&2; exit 1; }
  sed -i "s/^- \[ \] $id /- [x] $id /" PHASES.md
done
