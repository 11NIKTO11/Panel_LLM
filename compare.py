"""
Compare several saved runs of one topic from the command line.

    python compare.py election_2021 haiku-4-5 haiku-5-5     each run by any text only its stem contains
    python compare.py election_2021 haiku-4-5 haiku-5-5 --name haiku
    python compare.py election_2021 --set haiku             a named set of the topic's comparisons.json
    python compare.py election_2021 haiku-4-5 haiku-5-5 --only distances
    python compare.py election_2021 haiku-4-5 haiku-5-5 --no-save
    python compare.py election_2021 haiku-4-5 haiku-5-5 --plain    ignore the topic's plot.json

`python analyze.py <topic>` lists a topic's saved runs. Writes into
`results/<topic>/comparisons/<name>/`: tables to `analysis/`, figures to `plots/`, named
`runs_<comparison>`, and `runs.json` saying which runs were compared; the name is the set's, or
the topic unless given. `compare.ipynb` is the same thing one block at a time.
"""
import argparse

from panel import paths
from panel.analysis import NotApplicable
from panel.analysis.comparison import COMPARISONS, comparison_output, describe, load_comparisons, load_runs
from panel.run import ResultStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved runs of one topic.")
    parser.add_argument("topic", help="a folder under results/")
    parser.add_argument("runs", nargs="*", help="per run, a stem or any text only one stem contains")
    parser.add_argument("--set", help="a named set of the topic's comparisons.json, in place of runs; its labels name the runs, its name the files")
    parser.add_argument("--name", help="what the written files are named after; the set, or the topic, by default")
    parser.add_argument("--only", choices=list(COMPARISONS), help="one comparison instead of all")
    parser.add_argument("--predictors", help="for effects: the columns to test, comma-separated; the dataset's grouping columns and claims by default")
    parser.add_argument("--delta", type=float, help="for effects: what a zero share is replaced by before the log-ratio transform; 0.005 by default")
    parser.add_argument("--no-save", action="store_true", help="write no table and no figure")
    parser.add_argument("--plain", action="store_true", help="ignore the topic's plot.json")
    parser.add_argument("--unweighted", action="store_true", help="do not weight option shares by the probability of voting; files get the suffix _unweighted")
    args = parser.parse_args()

    named = load_comparisons(args.topic)
    if bool(args.set) == bool(args.runs) or (args.set and args.set not in named):
        parser.error(f"name either the topic's runs, or --set, one of {list(named)}")
    texts, name = (named[args.set].runs, args.set) if args.set else (args.runs, args.topic)
    store = ResultStore(paths.RESULTS_DIR, args.topic)
    runs = load_runs(store, texts, styled=not args.plain, weighted=not args.unweighted)
    out = comparison_output(store, args.name or name, runs, save=not args.no_save, suffix="_unweighted" if args.unweighted else "")
    settings = {"effects": {**({"predictors": args.predictors.split(",")} if args.predictors else {}), **({"delta": args.delta} if args.delta is not None else {})}}
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
