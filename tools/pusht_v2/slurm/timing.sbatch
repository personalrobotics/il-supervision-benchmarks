#!/bin/bash
#SBATCH --job-name=ilsb-surface-timing
#SBATCH --account=toteconsolidation
#SBATCH --partition=gpu-h200
#SBATCH --qos=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:55:00
#SBATCH --output=%x-%j.log
# The off-study 20k timing unit (learner seed 99, never reused): wall time, s/step, peak GPU and host memory.
#   ILSB_COMMIT=<sha> ROOT=<new run dir> sbatch --export=ALL tools/pusht_v2/slurm/timing.sbatch
source "${SLURM_SUBMIT_DIR:-.}/tools/pusht_v2/slurm/env.sh"
header
SPEC=$SRC/tools/pusht_v2/surface-timing.json
H=$HARNESS/bin/python
cd "$SRC"
$H -m ilsb stage preflight "$SPEC" --root "$ROOT"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 30 > "$ROOT/nvidia-smi.csv" &
/usr/bin/time -v $H -m ilsb unit "$SPEC" --root "$ROOT" --key ccil/r1.0/a0.8/s99 2> "$ROOT/time-v.txt"
echo "end: $(date -Is)"
