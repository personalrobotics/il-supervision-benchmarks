#!/bin/bash
# Login node, once per protocol commit (no training): the clone and the two environments, each verified.
#   ILSB_COMMIT=<sha> bash tools/pusht_v2/slurm/setup.sh
source "$(dirname "$0")/env.sh"
UV=/usr/local/bin/uv
[ -d "$SRC" ] || git clone -q https://github.com/personalrobotics/il-supervision-benchmarks "$SRC"
git -C "$SRC" checkout -q "$COMMIT"
[ -z "$(git -C "$SRC" status --porcelain)" ] || { echo "the clone is dirty"; exit 1; }
cd "$SRC"
# The harness: uv.lock's versions exactly, with CPU PyTorch (it never uses a GPU), and this checkout itself.
$UV venv -q --python 3.12 "$HARNESS"
$UV export -q --frozen --no-hashes --no-emit-project > "$HARNESS/requirements.txt"
$UV pip install -q --python "$HARNESS/bin/python" --torch-backend cpu -r "$HARNESS/requirements.txt"
$UV pip install -q --python "$HARNESS/bin/python" --no-deps -e .
# The learner environment: the pinned freeze, then this checkout's plugins and learner.
$UV venv -q --python 3.12.15 "$LEROBOT"
$UV pip install -q --python "$LEROBOT/bin/python" -r tools/pusht_v2/surface_learner/lerobot-env.txt
$UV pip install -q --python "$LEROBOT/bin/python" --no-deps tools/pusht_v2/env_plugin tools/pusht_v2/ibc_plugin \
  tools/pusht_v2/surface_learner
$UV pip freeze --python "$LEROBOT/bin/python" | grep -v ' @ file://' | diff - <(grep -v ' @ file://' \
  tools/pusht_v2/surface_learner/lerobot-env.txt) && echo "learner environment matches the freeze"
