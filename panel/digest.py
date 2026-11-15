"""
The sha256 of a file, and how a run record names the ones it carries.

A record's `dataset` block holds the digest of each file the run was built from under
`<name>_sha256`; the name is one of `Dataset.built_from`. One place writes that key and reads it back.
"""
import hashlib
from pathlib import Path
from typing import Dict, Mapping

SUFFIX = "_sha256"


def sha256(path: Path) -> str:
    """The sha256 of a file's bytes, as hex."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def named(digests: Mapping[str, str]) -> Dict[str, str]:
    """Digests by name as a record's block carries them: `<name>_sha256` -> digest."""
    return {f"{name}{SUFFIX}": digest for name, digest in digests.items()}


def recorded(block: Mapping[str, object]) -> Dict[str, str]:
    """The digests a record's block carries, by name; whatever else the block holds is left out."""
    return {key[:-len(SUFFIX)]: digest for key, digest in block.items() if key.endswith(SUFFIX)}
