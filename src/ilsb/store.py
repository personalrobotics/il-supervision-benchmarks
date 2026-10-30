"""Content-addressed units: an entry is named by the SHA-256 of its complete input identity.

An entry holds identity.json (that identity), its files, and manifest.json (each file's SHA-256 and size). It is
written whole into a temporary directory and renamed into place, so an entry either exists complete or not at all.
Reuse needs the stored identity to equal the requested one and every file to match the manifest; any other state is
refused, never silently recomputed or overwritten. Entries may be shared across studies: equal identity is equal work.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path


class StoreError(RuntimeError):
    """A refused reuse: mismatched identity, a corrupted or incomplete entry."""


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def digest(obj) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    """Atomic JSON write (temporary file, then rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)

    def path(self, kind: str, identity: dict) -> Path:
        return self.root / kind / digest(identity)

    def get(self, kind: str, identity: dict) -> Path | None:
        """The verified entry for identity, None if absent; StoreError if present but not exactly this work."""
        d = self.path(kind, identity)
        if not d.exists():
            return None
        if not (d / "identity.json").exists() or not (d / "manifest.json").exists():
            raise StoreError(f"{d} is an incomplete entry")
        if json.loads((d / "identity.json").read_text()) != json.loads(canonical(identity)):
            raise StoreError(f"{d} holds a different identity than the one requested (mismatched reuse)")
        manifest = json.loads((d / "manifest.json").read_text())
        present = {p.name for p in d.iterdir()} - {"identity.json", "manifest.json"}
        if present != set(manifest):
            raise StoreError(f"{d} files {sorted(present)} differ from its manifest {sorted(manifest)}")
        for name, m in manifest.items():
            if sha256_file(d / name) != m["sha256"]:
                raise StoreError(f"{d / name} does not match its manifest (corrupted)")
        return d

    def put(self, kind: str, identity: dict, write) -> Path:
        """Create the entry: write(tmpdir) fills it; the manifest is computed and the directory renamed into place."""
        d = self.path(kind, identity)
        if d.exists():
            raise StoreError(f"{d} exists; use get()")
        d.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".{d.name}.", dir=d.parent))
        try:
            write(tmp)
            (tmp / "identity.json").write_text(canonical(identity) + "\n")
            files = sorted(p for p in tmp.iterdir() if p.name not in ("identity.json", "manifest.json"))
            if any(not p.is_file() or p.is_symlink() for p in files):
                raise StoreError("entries hold regular files only")
            (tmp / "manifest.json").write_text(canonical({p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                                                          for p in files}) + "\n")
            os.replace(tmp, d)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return d

    def ensure(self, kind: str, identity: dict, write) -> tuple[Path, bool]:
        """(entry, reused): the verified existing entry, or a new one."""
        got = self.get(kind, identity)
        return (got, True) if got is not None else (self.put(kind, identity, write), False)
