"""
The digest of a file a run was built from, and how a run record names the ones it carries.

A record's `dataset` block holds the digest of each such file under `<name>_sha256`; the name is
one of `Dataset.built_from`. One place writes that key and reads it back.

The digest is of the file's text, not of its bytes: every line ending counts as `\\n`. The
files are text - tables and forms - and git on Windows rewrites their line endings on checkout,
so the same file has two byte forms and a digest of the bytes would orphan a run every time the
form changed. `file_sha256` is the plain one, for a pack's manifest, whose files are copies that
must not change at all.
"""
import hashlib
from pathlib import Path
from typing import Dict, Mapping

SUFFIX = "_sha256"


def text(data: bytes) -> bytes:
    """`data` with every line ending as `\\n`."""
    return data.replace(b"\r\n", b"\n")


def text_sha256(data: bytes) -> str:
    return hashlib.sha256(text(data)).hexdigest()


def content_sha256(path: Path) -> str:
    return text_sha256(path.read_bytes())


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def named(digests: Mapping[str, str]) -> Dict[str, str]:
    return {f"{name}{SUFFIX}": digest for name, digest in digests.items()}


def recorded(block: Mapping[str, object]) -> Dict[str, str]:
    return {key[:-len(SUFFIX)]: digest for key, digest in block.items() if key.endswith(SUFFIX)}
