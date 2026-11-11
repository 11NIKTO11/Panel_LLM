"""
Analyze a saved run from the command line.

    python analyze.py                               lists every topic's saved runs
    python analyze.py election_2021                 lists that topic's saved runs
    python analyze.py election_2021 haiku-5-5       the run whose stem contains that text
    python analyze.py election_2021 haiku-5-5 --only country
    python analyze.py election_2021 haiku-5-5 --only effects --predictors gender,age,region
    python analyze.py election_2021 haiku-5-5 --no-save
    python analyze.py election_2021 haiku-5-5 --plain      ignore the topic's plot.json

Reads `results/<topic>/`, writes tables to `results/<topic>/analysis/` and figures to
`results/<topic>/plots/`. `analyze.ipynb` is the same thing one block at a time.
"""
import argparse

from panel import paths
from panel.analysis import ANALYSES, NotApplicable, Output, describe, find_stem, load_run
from panel.run import ResultStore, resolve
from panel.run.check import check, report
from panel.topic import topic_names


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a saved run.")
    parser.add_argument("topic", nargs="?", help="a folder under results/")
    parser.add_argument("run", nargs="?", help="a stem, or any text only one stem contains")
    parser.add_argument("--only", choices=list(ANALYSES), help="one analysis instead of all")
    parser.add_argument("--predictors", help="for effects: the columns to test, comma-separated; the dataset's grouping columns and claims by default")
    parser.add_argument("--no-save", action="store_true", help="write no table and no figure")
    parser.add_argument("--plain", action="store_true", help="default colours instead of the topic's plot.json")
    args = parser.parse_args()

    if args.run is None:
        for topic in [args.topic] if args.topic else topic_names():
            for stem in ResultStore(paths.RESULTS_DIR, topic).runs():
                print(f"  {topic:<22} {stem}")
        return

    store = ResultStore(paths.RESULTS_DIR, args.topic)
    run = load_run(store, find_stem(store, args.run), styled=not args.plain)
    out = Output(store, run.stem, save=not args.no_save)
    settings = {"effects": {"predictors": args.predictors.split(",")}} if args.predictors else {}
    print(describe(run))
    print(f"\n{report(check(resolve(run.spec), store), len(run.options))}")
    for name, analysis in ANALYSES.items():
        if args.only in (None, name):
            try:
                table, _ = analysis(run, out, **settings.get(name, {}))
            except NotApplicable as reason:
                print(f"\n{name}: skipped - {reason}")
                continue
            print(f"\n{name}\n{table.round(3).to_string()}")
    for path in out.written:
        print(f"-> {path}")


if __name__ == "__main__":
    main()
