export PATH=$HOME/.local/bin:$PATH
I=/home/siddh/code/il-supervision-benchmarks
OUT=$I/runs/20261102T160255Z-recovery-tube
cd /home/siddh/code/CCIL-rtube-6fbcce5
CMD1="uv run -q --extra torch python $I/tools/recovery_tube_dup.py --audit $OUT/audit.json --decision $OUT/decision.json --out $OUT/matched"
{ echo "command $CMD1"; echo "ccil commit $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l)"; echo "ilsb commit $(git -C $I rev-parse HEAD) dirty=$(git -C $I status --porcelain | wc -l)"; echo "host $(hostname) cpus $(nproc)"; free -g | head -2; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader; uptime; uv --version; echo "stall threshold 1800 s; completion marker EXIT= and matched/manifest.json"; }
uv run -q --extra torch python -c "import numpy, torch, sys; print('numpy', numpy.__version__, 'torch', torch.__version__, 'python', sys.version.split()[0])"
$CMD1 || exit $?
echo "$(date +%T) training done"
cd $I
CMD2="uv run python tools/recovery_tube.py evaluate --out $OUT --geometry-sha256 d0b0872a18ddbcd89a89afb4599ba3aa0504b667e2d42b56e3c5fc5a31ed7745 --cohort matched --resume"
CMD3="uv run python tools/recovery_tube.py decide --out $OUT"
echo "command $CMD2 && $CMD3"
$CMD2 && $CMD3
