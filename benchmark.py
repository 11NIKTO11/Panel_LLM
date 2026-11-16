"""
Hold saved runs against a model trained on the subjects' own answers, from the command line.

    python benchmark.py election_2021 haiku-5-5 --name haiku            a run by any text only its stem contains
    python benchmark.py election_2021 haiku-4-5 haiku-5-5 --name haiku  several, scored beside each other
    python benchmark.py election_2021 --set grids                         a named set of the topic's comparisons.json
    python benchmark.py election_2021 haiku-5-5 --name haiku --only scores
    python benchmark.py election_2021 haiku-5-5 --name haiku --predictors gender,age,region
    python benchmark.py election_2021 haiku-5-5 --name haiku --no-save
    python benchmark.py election_2021 haiku-5-5 --name haiku --plain    ignore the topic's plot.json

`python analyze.py <topic>` lists a topic's saved runs. Writes into
`results/<topic>/comparisons/<name>/`: tables to `analysis/`, figures to `plots/`, named
`benchmark_<benchmark>`, and `runs.json` saying which runs were scored; the name is the set's, or
`--name`, which runs named one by one need. `benchmark.ipynb` is the same thing one block at a time.
"""
import argparse

from cli import add_output, add_run_set, run_each, run_set
from panel import paths
from panel.analysis.benchmark import BENCHMARKS, describe, load_ground
from panel.analysis.comparison import comparison_output, load_runs
from panel.run import ResultStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark saved runs against a model trained on the subjects' own answers.")
    add_run_set(parser)
    parser.add_argument("--only", choices=list(BENCHMARKS), help="one benchmark instead of all")
    parser.add_argument("--predictors", help="the columns the trained model is given, comma-separated; the dataset's grouping columns by default")
    add_output(parser)
    args = parser.parse_args()

    texts, name = run_set(parser, args)
    store = ResultStore(paths.RESULTS_DIR, args.topic)
    runs = load_runs(store, texts, styled=not args.plain)
    ground = load_ground(runs.runs, args.predictors.split(",") if args.predictors else None)
    out = comparison_output(store, name, runs, save=not args.no_save)
    print(describe(ground))
    run_each(BENCHMARKS, args.only, ground, out)


if __name__ == "__main__":
    main()
