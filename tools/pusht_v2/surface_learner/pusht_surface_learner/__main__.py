import sys
from pathlib import Path

from .train import rescore, train

if __name__ == "__main__":
    cmd, *args = sys.argv[1:]
    if cmd == "train":
        sys.exit(train(Path(args[0])))
    if cmd == "infonce":
        sys.exit(rescore(Path(args[0]), Path(args[1]), Path(args[2])))
    sys.exit(f"unknown command {cmd!r}: train UNIT_DIR | infonce UNIT_DIR CHECKPOINT OUT_JSON")
