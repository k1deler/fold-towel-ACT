#!/usr/bin/env bash
# Setup and saved-input validation only. Never starts real cameras or arms.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
for script in scripts/*.sh; do bash -n "$script"; done
python3 - <<'PY'
from pathlib import Path
import hashlib,json
root=Path.cwd()
manifest=json.loads((root/'DEPLOY_FILES_SHA256.json').read_text())
for name,expected in manifest.items():
    p=(root/name).resolve()
    if not p.is_relative_to(root):raise SystemExit('Invalid file manifest path')
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    if h.hexdigest()!=expected:raise SystemExit('Bundle hash mismatch: '+name)
print('BUNDLE_FILES_VERIFIED')
PY
bash scripts/setup_robot.sh "$ROOT/artifacts/fold_towel_act_delta_seed42.pt"
echo 'FOLD_TOWEL_INSTALL_AND_OFFLINE_REPLAY_PASSED; no hardware started'
