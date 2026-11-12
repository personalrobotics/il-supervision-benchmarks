# Sourced by every PushT surface job on Tillicum: one layout, one environment pair, paths the jobs agree on.
# BASE holds src/ilsb-<commit> (a clone at the protocol commit), envs/ and data/; runs go to BASE/runs/pusht-surface.
set -euo pipefail
BASE=${ILSB_BASE:-/gpfs/projects/toteconsolidation/siddh}
COMMIT=${ILSB_COMMIT:?set ILSB_COMMIT to the protocol commit}
SRC=$BASE/src/ilsb-$COMMIT
HARNESS=$BASE/envs/ilsb-surface-harness-$COMMIT
LEROBOT=$BASE/envs/ilsb-surface-lerobot-$COMMIT
export ILSB_LEROBOT_PYTHON=$LEROBOT/bin/python
export ILSB_PUSHT_DATASET=$BASE/data/pusht_keypoints-ace8c161
export UV_CACHE_DIR=$BASE/uv-cache UV_PYTHON_INSTALL_DIR=$BASE/uv-python
export LC_ALL=C
# Inductor runs `nvcc --version`; a non-searchable directory on PATH turns "not found" into a PermissionError.
export PATH=$LEROBOT/bin:/usr/local/bin:/usr/bin:/bin

header() {  # the run log's first lines: what ran, where, on what
  echo "command: $0 $*"
  echo "commit: $(git -C "$SRC" rev-parse HEAD) dirty: $(git -C "$SRC" status --porcelain | wc -l)"
  echo "host: $(hostname) job: ${SLURM_JOB_ID:-none} array: ${SLURM_ARRAY_TASK_ID:-none} start: $(date -Is)"
  nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv,noheader || true
  "$ILSB_LEROBOT_PYTHON" -c "import torch, lerobot; print('lerobot', lerobot.__version__, 'torch', torch.__version__)"
  "$HARNESS/bin/python" -c "import torch, ccil; print('harness torch', torch.__version__, 'ccil', ccil.__version__)"
}
