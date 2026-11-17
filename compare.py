"""
Compare several saved runs of one topic from the command line.

`python analyze.py <topic>` lists a topic's saved runs. Writes into
`results/<topic>/comparisons/<name>/`: tables to `analysis/`, figures to `plots/`, named
`runs_<comparison>`, and `runs.json` saying which runs were compared; the name is the set's, or
`--name`, which runs named one by one need. `compare.ipynb` is the same thing one block at a time.
"""
import argparse

from cli import add_effects, add_output, add_run_set, effects_settings, run_each, run_set, suffix
from panel import paths
from panel.analysis.comparison import COMPARISONS, comparison_output, describe, load_runs
from panel.run import ResultStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved runs of one topic.")
    add_run_set(parser)
    parser.add_argument("--only", choices=list(COMPARISONS), help="one comparison instead of all")
    add_effects(parser)
    add_output(parser)
    args = parser.parse_args()

    texts, name = run_set(parser, args)
    store = ResultStore(paths.RESULTS_DIR, args.topic)
    runs = load_runs(store, texts, styled=not args.plain, weighted=not args.unweighted)
    out = comparison_output(store, name, runs, save=not args.no_save, suffix=suffix(args))
    print(describe(runs))
    run_each(COMPARISONS, args.only, runs, out, effects_settings(args))


if __name__ == "__main__":
    main()
