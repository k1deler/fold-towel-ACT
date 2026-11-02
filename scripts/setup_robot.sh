#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ $# == 1 ]] || { echo 'Usage: bash scripts/setup_robot.sh /absolute/path/to/ACT_model.pt'; exit 2; }
cd "$ROOT/inference"
export PYTHONDONTWRITEBYTECODE=1
if [[ ! -x .venv/bin/python ]]; then
  if command -v uv >/dev/null; then uv venv .venv --python /usr/bin/python3.10
  else /usr/bin/python3.10 -m venv .venv; fi
fi
PY="$ROOT/inference/.venv/bin/python"
install_deps() {
  if command -v uv >/dev/null; then uv pip install --python "$PY" "$@"
  else "$PY" -m pip install "$@"; fi
}
install_deps 'torch==2.7.1+cu128' 'torchvision==0.22.1+cu128' --index-url https://download.pytorch.org/whl/cu128
install_deps -r requirements.txt pytest --index-url https://pypi.org/simple
export PYTHONPATH="$ROOT/inference:$ROOT/inference/third_party/i2rt"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
"$PY" -m pytest -q -p no:cacheprovider
"$PY" "$ROOT/scripts/register_model.py" "$1"
"$PY" "$ROOT/scripts/validate_model.py"
echo 'ROBOT_ENVIRONMENT_PREPARED; no real device initialized'
