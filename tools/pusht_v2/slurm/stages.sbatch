#!/bin/bash
#SBATCH --job-name=ilsb-surface-stages
#SBATCH --account=toteconsolidation
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
# The stages around the policy array, on one GPU: STAGES="preflight" before it; STAGES="train audit evaluate
# evaluation_audit report" after it (train only collects the array's units: no --gpus).
#   ILSB_COMMIT=<sha> ROOT=<run dir> SPEC=<spec> STAGES="..." sbatch --export=ALL --output=$ROOT/%x-%j.log \
#     tools/pusht_v2/slurm/stages.sbatch   (from the clone; the log stays outside it)
source "${SLURM_SUBMIT_DIR:-.}/tools/pusht_v2/slurm/env.sh"
header
cd "$SRC"
for s in $STAGES; do
  echo "stage $s: $(date -Is)"
  "$HARNESS/bin/python" -m ilsb stage "$s" "$SRC/$SPEC" --root "$ROOT"
done
echo "end: $(date -Is)"
