"""The deterministic, atomic npz write of policy checkpoints: copied from ccil.schema.io._atomic_savez at CCIL
3b090bb, so a checkpoint is byte for byte the one CCIL's trainer writes."""
from __future__ import annotations

import io as io_
import os
import zipfile

import numpy as np


def _atomic_savez(path, arrays: dict) -> None:
    """Write an .npz byte-for-byte deterministically: np.savez stamps each member with
    the current time, so members are written in sorted order with a fixed timestamp."""
    tmp = f"{path}.tmp-{os.getpid()}.npz"
    try:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
            for name in sorted(arrays):
                buf = io_.BytesIO()
                np.lib.format.write_array(buf, np.asanyarray(arrays[name]), allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.external_attr = 0o644 << 16
                zf.writestr(info, buf.getvalue())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
