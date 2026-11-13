"""
Hold saved runs against a model trained on the subjects' own answers, from the command line.

    python benchmark.py election_2021 haiku-5-5                 a run by any text only its stem contains
    python benchmark.py election_2021 haiku-4-5 haiku-5-5 --name haiku
    python benchmark.py --set haiku                             a named set of config/comparisons.json
    python benchmark.py election_2021 haiku-5-5 --only scores
    python benchmark.py election_2021 haiku-5-5 --predictors gender,age,region
    python benchmark.py election_2021 haiku-5-5 --no-save
    python benchmark.py election_2021 haiku-5-5 --plain         ignore the topic's plot.json

`python analyze.py <topic>` lists a topic's saved runs. Writes tables to
`results/<topic>/analysis/` and figures to `results/<topic>/plots/`, named
`benchmark_<name>_<name given>`; the name is the topic unless given. `benchmark.ipynb` is the
same thing one block at a time.
"""
import argparse

from panel import paths
from panel.analysis import Output
from panel.analysis.benchmark import BENCHMARKS, describe, load_ground
from panel.analysis.comparison import load_comparisons, load_runs
from panel.run import ResultStore

COMPARISONS_PATH = paths.CONFIG_DIR / "comparisons.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark saved runs against a model trained on the subjects' own answers.")
    parser.add_argument("topic", nargs="?", help="a folder under results/")
    parser.add_argument("runs", nargs="*", help="per run, a stem or any text only one stem contains")
    parser.add_argument("--set", help="a named set of config/comparisons.json, in place of a topic and runs; its labels name the runs, its name the files")
    parser.add_argument("--name", help="what the written files are named after; the set, or the topic, by default")
    parser.add_argument("--only", choices=list(BENCHMARKS), help="one benchmark instead of all")
    parser.add_argument("--predictors", help="the columns the trained model is given, comma-separated; the dataset's grouping columns by default")
    parser.add_argument("--no-save", action="store_true", help="write no table and no figure")
    parser.add_argument("--plain", action="store_true", help="ignore the topic's plot.json")
    args = parser.parse_args()

    if bool(args.set) == bool(args.topic):
        parser.error(f"name either a topic and its runs, or --set, one of {list(load_comparisons(COMPARISONS_PATH))}")
    if args.topic and not args.runs:
        parser.error("name the runs of the topic")
    chosen = load_comparisons(COMPARISONS_PATH)[args.set] if args.set else None
    topic, texts, name = (chosen.topic, chosen.runs, chosen.name) if chosen else (args.topic, args.runs, args.topic)
    store = ResultStore(paths.RESULTS_DIR, topic)
    ground = load_ground(load_runs(store, texts, styled=not args.plain), args.predictors.split(",") if args.predictors else None)
    out = Output(store, args.name or name, save=not args.no_save)
    print(describe(ground))
    for name, benchmark in BENCHMARKS.items():
        if args.only in (None, name):
            table, _ = benchmark(ground, out)
            print(f"\n{name}\n{table.round(3).to_string()}")
    for path in out.written:
        print(f"-> {path}")


if __name__ == "__main__":
    main()
