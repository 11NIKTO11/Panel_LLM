"""
Runs read as kernels: a matrix from profiles to answers (`matrix`), read from saved runs
(`loaded`), its factors as a design (`factors`), its replicates (`replicates`), and one module
per analysis of it, after `CoRe/llm_pushforward`.

Every analysis is a function `(kernels, out) -> (table, figures)` over a `KernelSet` of one
run or several, raising `NotApplicable` when it has nothing to work on; `KERNEL_ANALYSES` names
them in the order a full pass runs them. What `panel.analysis` and its comparisons already do
(MANOVA, PERMANOVA, country and area shares, the expert map) is not repeated here.
"""
from .agreement import agreement
from .axes import axes
from .electorates import electorates
from .importance import importance
from .intervals import intervals
from .loaded import describe, kernel_of, load_kernels
from .matrix import Kernel, KernelSet, push
from .noise import noise
from .pairs import pairs
from .populations import populations
from .scales import scales
from .sensitivity import sensitivity
from .shapley import shapley
from .surrogate import surrogate
from .variance import variance

KERNEL_ANALYSES = {
    "noise": noise, "scales": scales, "pairs": pairs, "variance": variance, "importance": importance,
    "axes": axes, "shapley": shapley, "electorates": electorates, "populations": populations, "sensitivity": sensitivity,
    "surrogate": surrogate, "agreement": agreement, "intervals": intervals,
}
