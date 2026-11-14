"""
Hold saved runs against a model trained on the subjects' own answers, from the command line.

    python benchmark.py election_2021 haiku-5-5                 a run by any text only its stem contains
    python benchmark.py election_2021 haiku-4-5 haiku-5-5 --name haiku
    python benchmark.py election_2021 --set haiku               a named set of the topic's comparisons.json
    python benchmark.py election_2021 haiku-5-5 --only scores
    python benchmark.py election_2021 haiku-5-5 --predictors gender,age,region
    python benchmark.py election_2021 haiku-5-5 --no-save
    python benchmark.py election_2021 haiku-5-5 --plain         ignore the topic's plot.json

`python analyze.py <topic>` lists a topic's saved runs. Writes into
`results/<topic>/comparisons/<name>/`: tables to `analysis/`, figures to `plots/`, named
`benchmark_<benchmark>`, and `runs.json` saying which runs were scored; the name is the set's, or
the topic unless given. `benchmark.ipynb` is the same thing one block at a time.
"""
import argparse

from panel import paths
from panel.analysis.benchmark import BENCHMARKS, describe, load_ground
from panel.analysis.comparison import comparison_output, load_comparisons, load_runs
from panel.run import ResultStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark saved runs against a model trained on the subjects' own answers.")
    parser.add_argument("topic", help="a folder under results/")
    parser.add_argument("runs", nargs="*", help="per run, a stem or any text only one stem contains")
    parser.add_argument("--set", help="a named set of the topic's comparisons.json, in place of runs; its labels name the runs, its name the files")
    parser.add_argument("--name", help="what the written files are named after; the set, or the topic, by default")
    parser.add_argument("--only", choices=list(BENCHMARKS), help="one benchmark instead of all")
    parser.add_argument("--predictors", help="the columns the trained model is given, comma-separated; the dataset's grouping columns by default")
    parser.add_argument("--no-save", action="store_true", help="write no table and no figure")
    parser.add_argument("--plain", action="store_true", help="ignore the topic's plot.json")
    args = parser.parse_args()

    named = load_comparisons(args.topic)
    if bool(args.set) == bool(args.runs) or (args.set and args.set not in named):
        parser.error(f"name either the topic's runs, or --set, one of {list(named)}")
    texts, name = (named[args.set].runs, args.set) if args.set else (args.runs, args.topic)
    store = ResultStore(paths.RESULTS_DIR, args.topic)
    runs = load_runs(store, texts, styled=not args.plain)
    ground = load_ground(runs, args.predictors.split(",") if args.predictors else None)
    out = comparison_output(store, args.name or name, runs, save=not args.no_save)
    print(describe(ground))
    for name, benchmark in BENCHMARKS.items():
        if args.only in (None, name):
            table, _ = benchmark(ground, out)
            print(f"\n{name}\n{table.round(3).to_string()}")
    for path in out.written:
        print(f"-> {path}")


if __name__ == "__main__":
    main()
