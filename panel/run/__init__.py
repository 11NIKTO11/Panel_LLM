"""A run: declared by names (`spec`), resolved into artifacts and calls (`resolve`), sent and saved (`execute`, `store`)."""
from .spec import RunSpec
from .resolve import Resolved, resolve
from .store import ResultStore, RunPaths
from .execute import execute
