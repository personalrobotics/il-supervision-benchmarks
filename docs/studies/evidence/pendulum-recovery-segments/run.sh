#!/bin/bash
cd ~/code/il-supervision-benchmarks; export PATH=$HOME/.local/bin:$PATH
TS=$(date -u +%Y%m%dT%H%M%SZ); OUT=runs/$TS-recovery-segments; LOG=$OUT.log
cp ~/rseg-stage-a.sh $OUT.sh
tmux new-session -d -s ilsb-rseg-$TS "bash -c '{ echo \"command: uv run python tools/recovery_segments.py --out $OUT\"; echo \"commit: \$(git rev-parse HEAD) dirty_tracked: \$(git status --porcelain --untracked-files=no | wc -l)\"; echo \"host: \$(hostname) \$(uptime)\"; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader; free -g | head -2; uv run python -c \"import numpy, scipy, torch, ccil; print(\\\"numpy\\\", numpy.__version__, \\\"scipy\\\", scipy.__version__, \\\"torch\\\", torch.__version__, \\\"ccil\\\", ccil.__file__)\"; uv run python tools/recovery_segments.py --out $OUT; echo EXIT=\$?; } > $LOG 2>&1'"
echo $LOG ilsb-rseg-$TS
