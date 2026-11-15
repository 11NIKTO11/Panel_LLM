"""
Analyze a saved run from the command line.

    python analyze.py                               lists every topic's saved runs
    python analyze.py election_2021                 lists that topic's saved runs
    python analyze.py election_2021 haiku-5-5       the run whose stem contains that text
    python analyze.py election_2021 haiku-5-5 --only country
    python analyze.py election_2021 haiku-5-5 --only effects --predictors gender,age,region
    python analyze.py election_2021 haiku-5-5 --only effects --delta 0.001
    python analyze.py election_2021 haiku-5-5 --unweighted   option shares not weighted by turnout
    python analyze.py election_2021 haiku-5-5 --no-save
    python analyze.py election_2021 haiku-5-5 --plain      ignore the topic's plot.json

Reads `results/<topic>/runs/<run>/` and writes tables to its `analysis/`, figures to its
`plots/`. `analyze.ipynb` is the same thing one block at a time.
"""
import argparse

from cli import add_effects, add_output, effects_settings, list_runs, run_each, suffix
from panel import paths
from panel.analysis import ANALYSES, Output, describe, find_stem, load_run
from panel.run import ResultStore
from panel.run.check import check, report
from panel.run.record import DELIVERED
from panel.topic import load_framing


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a saved run.")
    parser.add_argument("topic", nargs="?", help="a folder under results/")
    parser.add_argument("run", nargs="?", help="a stem, or any text only one stem contains")
    parser.add_argument("--only", choices=list(ANALYSES), help="one analysis instead of all")
    add_effects(parser)
    add_output(parser)
    args = parser.parse_args()

    if args.run is None:
        list_runs(args.topic)
        return

    store = ResultStore(paths.RESULTS_DIR, args.topic)
    run = load_run(store, find_stem(store, args.run), styled=not args.plain, weighted=not args.unweighted)
    out = Output(store.run_dir(run.stem), save=not args.no_save, suffix=suffix(args))
    print(describe(run))
    if DELIVERED in run.record:
        print(f"\nkept as a delivered table - nothing to check: {run.record[DELIVERED]['note']}")
    else:
        print(f"\n{report(check(store, run.stem, run.topic, load_framing(run.topic, run.spec.framing)), len(run.options))}")
    run_each(ANALYSES, args.only, run, out, effects_settings(args))


if __name__ == "__main__":
    main()
