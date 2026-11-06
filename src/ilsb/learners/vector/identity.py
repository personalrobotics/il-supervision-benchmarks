"""Collision-resistant digests of transition identities, for deterministic per-sample randomness.

Copied from ccil.core.identity at CCIL 3b090bb. A transition is identified by (episode_id, frame), digested in
full: BLAKE2b-128 over a versioned encoding of the UTF-8 episode id (length-prefixed) and the frame (signed 64-bit).
The version bytes are CCIL's, so NoiseBC draws and v2 minibatch keys are the ones CCIL's trainers made.
"""
from __future__ import annotations

import hashlib
import struct

IDENTITY_VERSION = b"ccil.transition-identity/1"


def identity_digest(episode_id, frame) -> bytes:
    """16-byte digest of (episode_id, frame)."""
    ep = str(episode_id).encode("utf-8")
    payload = IDENTITY_VERSION + b"\0" + struct.pack(">Q", len(ep)) + ep + struct.pack(">q", int(frame))
    return hashlib.blake2b(payload, digest_size=16).digest()


def identity_words64(episode_id, frame) -> tuple[int, int]:
    """The digest as two unsigned 64-bit words (for counter hashing)."""
    return struct.unpack(">2Q", identity_digest(episode_id, frame))
