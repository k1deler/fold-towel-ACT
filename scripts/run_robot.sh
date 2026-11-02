#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -t 0 ]] || { echo 'An interactive terminal is required; SSH use ssh -t'; exit 1; }
[[ $# == 0 ]] || { echo 'Use this task config; command overrides disabled'; exit 2; }
PY="$ROOT/inference/.venv/bin/python"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT/inference:$ROOT/inference/third_party/i2rt"
unset CKPT YAM_LEFT_CAM_SERIAL YAM_RIGHT_CAM_SERIAL
exec 9>"$ROOT/.run.lock"
flock -n 9 || { echo 'This fold-towel runner is active'; exit 1; }
"$PY" "$ROOT/scripts/validate_model.py" --check
cd "$ROOT/inference"
"$PY" -m yam_infer.tools.free_hw --check --hw-config configs/hardware.yaml
echo 'START initializes arms/cameras and MOVES to the checkpoint start pose before Enter.'
echo 'q+Enter and Ctrl+C perform MOVING home before torque off; they are not an emergency stop.'
read -r -p 'On-site operator ready? Type START: ' answer
[[ "$answer" == START ]] || exit 1
exec "$PY" -u -m yam_infer.run --models-config "$ROOT/configs/model.yaml" \
 --run-config "$ROOT/configs/runtime.yaml" --hw-config configs/hardware.yaml \
 --device cuda --goto-start task_mean --park home --record-dir "$ROOT/runs_infer"
