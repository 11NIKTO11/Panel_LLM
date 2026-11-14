"""
Read saved runs as kernels - a matrix from profiles to answers - and analyse them, from the command line.

    python kernel.py election_2021 gridsod_ gridsimple          each run by any text only its stem contains
    python kernel.py election_2021 gridsod_ gridsimple --name grids
    python kernel.py election_2021 --set haiku                  a named set of the topic's comparisons.json
    python kernel.py election_2021 gridsod_ --only pairs
    python kernel.py election_2021 legacy_OpenAI_gpt-4o_Prob --factors gender,age,region
    python kernel.py election_2021 gridsod_ --no-save

`python analyze.py <topic>` lists a topic's saved runs. Writes into
`results/<topic>/comparisons/<name>/`: tables to `analysis/`, figures to `plots/`, named
`kernel_<analysis>`, and `runs.json` saying which runs were read; the name is the set's, or the
topic unless given. `kernel.ipynb` is the same thing one block at a time.
"""
import argparse

from panel import paths
from panel.analysis import NotApplicable
from panel.analysis.comparison import comparison_output, load_comparisons, load_runs
from panel.analysis.kernel import KERNEL_ANALYSES, describe, load_kernels
from panel.run import ResultStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Read saved runs of one topic as kernels and analyse them.")
    parser.add_argument("topic", help="a folder under results/")
    parser.add_argument("runs", nargs="*", help="per run, a stem or any text only one stem contains")
    parser.add_argument("--set", help="a named set of the topic's comparisons.json, in place of runs")
    parser.add_argument("--name", help="what the written files are named after; the set, or the topic, by default")
    parser.add_argument("--only", choices=list(KERNEL_ANALYSES), help="one analysis instead of all")
    parser.add_argument("--factors", help="the profile's columns, comma-separated; those the description reads by default")
    parser.add_argument("--reference", help="for agreement and intervals: the label of the run the others are held against; for agreement also 'trained on claims'")
    parser.add_argument("--draws", type=int, help="for intervals and shapley: replicates; 200 and 20 by default")
    parser.add_argument("--no-save", action="store_true", help="write no table and no figure")
    parser.add_argument("--plain", action="store_true", help="ignore the topic's plot.json")
    args = parser.parse_args()

    named = load_comparisons(args.topic)
    if bool(args.set) == bool(args.runs) or (args.set and args.set not in named):
        parser.error(f"name either the topic's runs, or --set, one of {list(named)}")
    texts, name = (named[args.set].runs, args.set) if args.set else (args.runs, args.topic)
    store = ResultStore(paths.RESULTS_DIR, args.topic)
    runs = load_runs(store, texts, styled=not args.plain)
    kernels = load_kernels(runs, args.factors.split(",") if args.factors else None)
    out = comparison_output(store, args.name or name, runs, save=not args.no_save)
    draws = {"draws": args.draws} if args.draws else {}
    reference = {"reference": args.reference} if args.reference else {}
    settings = {"agreement": reference, "intervals": {**draws, **reference}, "shapley": draws}
    print(describe(kernels))
    for analysis_name, analysis in KERNEL_ANALYSES.items():
        if args.only in (None, analysis_name):
            try:
                table, _ = analysis(kernels, out, **settings.get(analysis_name, {}))
            except NotApplicable as reason:
                print(f"\n{analysis_name}: skipped - {reason}")
                continue
            print(f"\n{analysis_name}\n{table.round(3).to_string()}")
    for path in out.written:
        print(f"-> {path}")


if __name__ == "__main__":
    main()
