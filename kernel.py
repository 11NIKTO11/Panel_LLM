"""
Read saved runs as kernels - a matrix from profiles to answers - and analyse them, from the command line.

    python kernel.py election_2021 gridsodretired_ gridsimple_  each run by any text only its stem contains
    python kernel.py election_2021 gridsodretired_ gridsimple_ --name grids
    python kernel.py election_2021 --set grids                  a named set of the topic's comparisons.json
    python kernel.py election_2021 gridsodretired_ --only pairs
    python kernel.py election_2021 legacy_OpenAI_gpt-4o_Prob --factors gender,age,region
    python kernel.py election_2021 gridsodretired_ --no-save

`python analyze.py <topic>` lists a topic's saved runs. Writes into
`results/<topic>/comparisons/<name>/`: tables to `analysis/`, figures to `plots/`, named
`kernel_<analysis>`, and `runs.json` saying which runs were read; the name is the set's, or the
topic unless given. `kernel.ipynb` is the same thing one block at a time.
"""
import argparse

from cli import add_output, add_run_set, run_each, run_set
from panel import paths
from panel.analysis.comparison import comparison_output, load_runs
from panel.analysis.kernel import KERNEL_ANALYSES, describe, load_kernels
from panel.run import ResultStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Read saved runs of one topic as kernels and analyse them.")
    add_run_set(parser)
    parser.add_argument("--only", choices=list(KERNEL_ANALYSES), help="one analysis instead of all")
    parser.add_argument("--factors", help="the profile's columns, comma-separated; those the description reads by default")
    parser.add_argument("--reference", help="for agreement and intervals: the label of the run the others are held against; for agreement also 'trained on claims'")
    parser.add_argument("--draws", type=int, help="for intervals and shapley: replicates; 200 and 20 by default")
    add_output(parser)
    args = parser.parse_args()

    texts, name = run_set(parser, args)
    store = ResultStore(paths.RESULTS_DIR, args.topic)
    runs = load_runs(store, texts, styled=not args.plain)
    kernels = load_kernels(runs, args.factors.split(",") if args.factors else None)
    out = comparison_output(store, name, runs, save=not args.no_save)
    draws = {"draws": args.draws} if args.draws else {}
    reference = {"reference": args.reference} if args.reference else {}
    print(describe(kernels))
    run_each(KERNEL_ANALYSES, args.only, kernels, out, {"agreement": reference, "intervals": {**draws, **reference}, "shapley": draws})


if __name__ == "__main__":
    main()
