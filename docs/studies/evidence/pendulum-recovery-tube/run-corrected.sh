cd /home/siddh/code/il-supervision-benchmarks
export PATH=$HOME/.local/bin:$PATH
OUT=/home/siddh/code/il-supervision-benchmarks/runs/20261102T160255Z-recovery-tube
CMD1="uv run python tools/recovery_tube.py evaluate --out $OUT --geometry-sha256 d0b0872a18ddbcd89a89afb4599ba3aa0504b667e2d42b56e3c5fc5a31ed7745 --cohort corrected --resume"
CMD2="uv run python tools/recovery_tube.py decide --out $OUT"
{ echo "command $CMD1 && $CMD2"; echo "commit $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l)"; echo "host $(hostname) cpus $(nproc)"; free -g | head -2; uptime; uv --version; echo "stall threshold 1800 s; heartbeat every 300 s; completion marker EXIT="; }
uv run python -c "import numpy, scipy, sys; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'python', sys.version.split()[0])"
$CMD1 && $CMD2
