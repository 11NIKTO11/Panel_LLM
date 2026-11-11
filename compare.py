"""
Compare several saved runs of one topic from the command line.

    python compare.py election_2021 haiku-4-5 haiku-5-5     each run by any text only its stem contains
    python compare.py election_2021 haiku-4-5 haiku-5-5 --name haiku
    python compare.py election_2021 haiku-4-5 haiku-5-5 --only distances
    python compare.py election_2021 haiku-4-5 haiku-5-5 --no-save
    python compare.py election_2021 haiku-4-5 haiku-5-5 --plain    ignore the topic's plot.json

`python analyze.py <topic>` lists a topic's saved runs. Writes tables to
`results/<topic>/analysis/` and figures to `results/<topic>/plots/`, named
`runs_<comparison>_<name>`; the name is the topic unless given. `compare.ipynb` is the same
thing one block at a time.
"""
import argparse

from panel import paths
from panel.analysis import NotApplicable, Output
from panel.analysis.comparison import COMPARISONS, describe, load_runs
from panel.run import ResultStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved runs of one topic.")
    parser.add_argument("topic", help="a folder under results/")
    parser.add_argument("runs", nargs="+", help="per run, a stem or any text only one stem contains")
    parser.add_argument("--name", help="what the written files are named after; the topic by default")
    parser.add_argument("--only", choices=list(COMPARISONS), help="one comparison instead of all")
    parser.add_argument("--predictors", help="for effects: the columns to test, comma-separated; the dataset's grouping columns and claims by default")
    parser.add_argument("--no-save", action="store_true", help="write no table and no figure")
    parser.add_argument("--plain", action="store_true", help="ignore the topic's plot.json")
    args = parser.parse_args()

    store = ResultStore(paths.RESULTS_DIR, args.topic)
    runs = load_runs(store, args.runs, styled=not args.plain)
    out = Output(store, args.name or args.topic, save=not args.no_save)
    settings = {"effects": {"predictors": args.predictors.split(",")}} if args.predictors else {}
    print(describe(runs))
    for name, comparison in COMPARISONS.items():
        if args.only in (None, name):
            try:
                table, _ = comparison(runs, out, **settings.get(name, {}))
            except NotApplicable as reason:
                print(f"\n{name}: skipped - {reason}")
                continue
            print(f"\n{name}\n{table.round(3).to_string()}")
    for path in out.written:
        print(f"-> {path}")


if __name__ == "__main__":
    main()
