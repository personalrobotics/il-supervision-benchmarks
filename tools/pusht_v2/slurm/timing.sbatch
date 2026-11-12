#!/bin/bash
#SBATCH --job-name=ilsb-surface-timing
#SBATCH --account=toteconsolidation
#SBATCH --partition=gpu-h200
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
# The off-study 20k timing unit (learner seed 99, never reused): wall time, s/step, peak GPU and host memory.
# Submit from the clone; the job's log goes to the run directory (a file in the clone would make the tree dirty):
#   mkdir -p $ROOT && ILSB_COMMIT=<sha> ROOT=$ROOT sbatch --export=ALL --output=$ROOT/%x-%j.log \
#     tools/pusht_v2/slurm/timing.sbatch
source "${SLURM_SUBMIT_DIR:-.}/tools/pusht_v2/slurm/env.sh"
header
SPEC=$SRC/tools/pusht_v2/surface-timing.json
H=$HARNESS/bin/python
cd "$SRC"
$H -m ilsb stage preflight "$SPEC" --root "$ROOT"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 30 > "$ROOT/nvidia-smi.csv" &
# the learner records wall time, s/step and peak host and GPU memory itself (result.json); nvidia-smi adds the device view
$H -m ilsb unit "$SPEC" --root "$ROOT" --key ccil/r1.0/a0.8/s99
echo "end: $(date -Is)"
