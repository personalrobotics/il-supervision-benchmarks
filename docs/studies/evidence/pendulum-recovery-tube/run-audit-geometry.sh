cd /home/siddh/code/il-supervision-benchmarks
export PATH=$HOME/.local/bin:$PATH
OUT=/home/siddh/code/il-supervision-benchmarks/runs/20261102T160255Z-recovery-tube
C=/home/siddh/code/CCIL/runs
CMD1="uv run python tools/recovery_tube.py audit --out $OUT --capture runs/20261102T003101Z-capture-control --corrected $C/20261029T191145Z-corrected-study --corrected-eval $C/20261030T013609Z-corrected-eval --ccil-evidence /home/siddh/code/CCIL/docs/fixtures/evidence/pendulum-corrected-demonstrator --response-surface runs/20261031T141026Z-j-r-k"
CMD2="uv run python tools/recovery_tube.py geometry --out $OUT"
{ echo "command $CMD1 && $CMD2"; echo "commit $(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l)"; echo "host $(hostname) cpus $(nproc)"; free -g | head -2; uptime; uv --version; echo "stall threshold 1800 s; heartbeat every 300 s; completion marker EXIT= and geometry.json"; }
uv run python -c "import numpy, scipy, sys; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'python', sys.version.split()[0])"
$CMD1 && $CMD2
