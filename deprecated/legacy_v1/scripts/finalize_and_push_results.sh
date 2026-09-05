#!/usr/bin/env bash
# Validate, organize, commit, and push scripts plus experiment CSVs (never datasets).
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=/private/users/lym/venv/bin/python
cd "$ROOT"

for model in audioseal wavmark timbrewm voicemark wmcodec; do
    for k in 2 3 5 8; do
        path="results/evaluation/tamper_arbitrary_N1024_${model}_K${k}.csv"
        [[ -s "$path" ]]
        rows=$(($(wc -l < "$path") - 1))
        if (( rows != 6000 )); then
            echo "incomplete matched tamper output: $path rows=$rows" >&2
            exit 1
        fi
    done
done

"$PYTHON" scripts/publish_results_to_data.py

# Stage only source code and the canonical data snapshot. Runtime results,
# dataset/, cache/, logs, partial checkpoints, and local config stay out.
git add scripts
git add DATA_INVENTORY.md
git add data
if git ls-files --error-unmatch results >/dev/null 2>&1; then
    git rm -r --cached results
fi
if git diff --cached --quiet; then
    echo "nothing new to commit"
else
    git commit -m "Add matched-registry tamper controls and organized results"
fi
# gh 2.4's credential-helper protocol is incompatible with the Git version on
# this host.  Read the already-authorized gh credential locally and pass it to
# this one Git process as an in-memory HTTP header; never place it in the remote
# URL, repository, logs, or process arguments.
"$PYTHON" - <<'PY'
import base64
import os
import subprocess
from pathlib import Path

import yaml

config_dir = Path(os.environ.get("GH_CONFIG_DIR", Path.home() / ".config" / "gh"))
hosts = yaml.safe_load((config_dir / "hosts.yml").read_text())
token = hosts["github.com"]["oauth_token"]
auth = base64.b64encode(f"x-access-token:{token}".encode()).decode()
env = os.environ.copy()
env.pop("GIT_ASKPASS", None)
env.pop("SSH_ASKPASS", None)
env["GIT_CONFIG_COUNT"] = "1"
env["GIT_CONFIG_KEY_0"] = "http.extraHeader"
env["GIT_CONFIG_VALUE_0"] = f"Authorization: Basic {auth}"
subprocess.run(["git", "push", "origin", "main"], check=True, env=env)
PY
