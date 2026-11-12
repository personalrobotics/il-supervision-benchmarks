#!/bin/bash
#SBATCH --job-name=ilsb-surface-units
#SBATCH --account=toteconsolidation
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:30:00
#SBATCH --output=%x-%A_%a.log
# One policy unit per array task, in the design's order (preflight.json); a task whose unit exists does nothing, and
# an interrupted one resumes from its newest checkpoint when resubmitted (only on an explicit decision to).
#   ILSB_COMMIT=<sha> ROOT=<run dir> SPEC=<spec> sbatch --export=ALL --array=0-27%8 tools/pusht_v2/slurm/units.sbatch
source "${SLURM_SUBMIT_DIR:-.}/tools/pusht_v2/slurm/env.sh"
header
cd "$SRC"
KEY=$("$HARNESS/bin/python" -c "import json, sys; print(list(json.load(open(sys.argv[1]))['units'])[int(sys.argv[2])])" \
  "$ROOT/stages/preflight.json" "$SLURM_ARRAY_TASK_ID")
echo "unit: $KEY"
"$HARNESS/bin/python" -m ilsb unit "$SRC/$SPEC" --root "$ROOT" --key "$KEY"
echo "end: $(date -Is) exit: $?"
