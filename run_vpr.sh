#!/usr/bin/env bash
set -euo pipefail

# Compatibility entrypoint: this independent package runs Trajectory, not PLD.
exec bash "$(dirname "$0")/run_trajectory.sh" "$@"
