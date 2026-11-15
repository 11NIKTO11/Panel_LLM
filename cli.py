"""
What the entry scripts share: the arguments several of them take, and how a pass over a set of
analyses is printed. Entry-layer code - it parses and prints, and leaves every decision to `panel`.
"""
import argparse
from typing import Any, Callable, Dict, List, Mapping, Tuple, Union

from panel import paths
from panel.analysis import NotApplicable, Output
from panel.analysis.comparison import load_comparisons
from panel.run import ResultStore
from panel.topic import topic_names


def list_runs(topic: Union[str, None]) -> None:
    """Print the saved runs of one topic, or of every topic."""
    for name in [topic] if topic else topic_names():
        for stem in ResultStore(paths.RESULTS_DIR, name).runs():
            print(f"  {name:<22} {stem}")


def add_run_set(parser: argparse.ArgumentParser) -> None:
    """The arguments that name several runs of one topic: the runs one by one, or a set of its comparisons.json."""
    parser.add_argument("topic", help="a folder under results/")
    parser.add_argument("runs", nargs="*", help="per run, a stem or any text only one stem contains")
    parser.add_argument("--set", help="a named set of the topic's comparisons.json, in place of runs; its labels name the runs, its name the files")
    parser.add_argument("--name", help="what the written files are named after; the set, or the topic, by default")


def run_set(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Tuple[Union[List[str], Dict[str, str]], str]:
    """The runs `add_run_set` named - texts, or label -> text for a set - and the name their files go under."""
    named = load_comparisons(args.topic)
    if bool(args.set) == bool(args.runs) or (args.set and args.set not in named):
        parser.error(f"name either the topic's runs, or --set, one of {list(named)}")
    return (named[args.set].runs, args.name or args.set) if args.set else (args.runs, args.name or args.topic)


def add_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-save", action="store_true", help="write no table and no figure")
    parser.add_argument("--plain", action="store_true", help="default colours instead of the topic's plot.json")


def add_effects(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--predictors", help="for effects: the columns to test, comma-separated; the dataset's grouping columns and claims by default")
    parser.add_argument("--delta", type=float, help="for effects: what a zero share is replaced by before the log-ratio transform; 0.005 by default")
    parser.add_argument("--unweighted", action="store_true", help="do not weight option shares by the probability of voting; files get the suffix _unweighted")


def effects_settings(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    """What `add_effects` set, as the keyword arguments of the `effects` analysis."""
    return {"effects": {**({"predictors": args.predictors.split(",")} if args.predictors else {}), **({"delta": args.delta} if args.delta is not None else {})}}


def suffix(args: argparse.Namespace) -> str:
    return "_unweighted" if args.unweighted else ""


def run_each(analyses: Mapping[str, Callable], only: Union[str, None], subject: Any, out: Output, settings: Union[Mapping[str, Dict[str, Any]], None] = None) -> None:
    """Run every analysis, or the one named, over `subject`; print each table, or why it was skipped, then what was written."""
    for name, analysis in analyses.items():
        if only in (None, name):
            try:
                table, _ = analysis(subject, out, **(settings or {}).get(name, {}))
            except NotApplicable as reason:
                print(f"\n{name}: skipped - {reason}")
                continue
            print(f"\n{name}\n{table.round(3).to_string()}")
    for path in out.written:
        print(f"-> {path}")
