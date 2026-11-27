#!/usr/bin/env bash
# Evaluate a ManiSkill task with RPent's API planner and DeepSeek Flash.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$RPENT_REPO_ROOT"

options=(--simulator maniskill --task-name PickCube-v1 --image_size 512)
if [[ -n "${RPENT_SAM3_ENDPOINT:-}" ]]; then
  options+=(--sam3-endpoint "$RPENT_SAM3_ENDPOINT")
fi
exec "${RPENT_PYTHON:-python}" -m rpent.cli.main --robot rl --planner api \
  --model deepseek:deepseek-flash --reasoning-effort high --no-strict-tools \
  "${options[@]}" "$@"
